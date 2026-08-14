"""
Evaluation metrics for the IoMT IDS Benchmarking Framework.

Computes all metrics specified in config.yaml under evaluation.metrics:
    - accuracy
    - precision_weighted, recall_weighted, f1_weighted
    - precision_macro, recall_macro, f1_macro
    - matthews_correlation_coefficient (MCC)
    - per_class_f1

Plus efficiency metrics (inference latency, throughput, memory, energy)
and confusion matrix visualization.

Design note — dual primary metric strategy:
    - CICIoMT2024 evaluation: weighted F1 is the primary ranking metric.
      "Weighted" averages per-class scores using class support as weight,
      reflecting real-world traffic distribution. This is what Optuna
      maximizes during HPO.
    - Cross-dataset generalization: macro F1 is the primary ranking metric.
      "Macro" gives equal weight to every class regardless of support,
      measuring whether the model generalizes across ALL attack families.
    - MCC is always reported as a complementary balanced measure that uses
      all four confusion matrix quadrants (TP, TN, FP, FN).
"""

import time
import os
import logging
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for a single model run.

    Computes both weighted-average and macro-average variants of precision,
    recall, and F1. Weighted metrics are the primary ranking criterion for
    CICIoMT2024 evaluation; macro metrics are primary for cross-dataset
    generalization. MCC is always reported as a complementary measure.

    Args:
        y_true: Ground-truth labels (n_samples,).
        y_pred: Predicted labels (n_samples,).
        y_proba: Predicted probabilities (n_samples, n_classes). Optional.
        class_names: Human-readable class names for per-class reporting.

    Returns:
        Dict with keys: accuracy, precision/recall/f1 (weighted and macro),
        mcc, per_class_f1, and classification_report_str.
    """
    metrics: Dict[str, Any] = {}

    # --- Accuracy ---
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # --- Weighted-average metrics (primary for CICIoMT2024) ---
    metrics["precision_weighted"] = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["recall_weighted"] = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    # --- Macro-average metrics (primary for cross-dataset generalization) ---
    metrics["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # --- Matthews Correlation Coefficient ---
    # MCC uses all four confusion matrix quadrants (TP, TN, FP, FN) and
    # ranges from -1 (total disagreement) to +1 (perfect prediction).
    # For multiclass, sklearn computes the multiclass generalization.
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # --- Per-class F1 scores ---
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class = f1_score(
        y_true, y_pred, labels=unique_classes, average=None, zero_division=0
    )
    if class_names is not None and len(class_names) == len(unique_classes):
        metrics["per_class_f1"] = {
            name: float(score) for name, score in zip(class_names, per_class)
        }
    else:
        metrics["per_class_f1"] = {
            str(cls): float(score) for cls, score in zip(unique_classes, per_class)
        }

    # --- Full classification report (text, for logging) ---
    target_names = class_names if class_names else [str(c) for c in unique_classes]
    metrics["classification_report_str"] = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    return metrics


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals for key classification metrics.

    Resamples the test set predictions with replacement n_bootstrap times
    and computes metrics on each resample to produce confidence intervals.

    This quantifies the stability of our point estimates without requiring
    computationally expensive repeated training runs. With large test sets
    (hundreds of thousands of samples), the resulting CIs are tight,
    confirming that the point estimates are reliable.

    Design decision: we chose bootstrap CIs over repeated runs because:
      - Repeated runs require re-running all 24 model-task combinations
        multiple times (days of GPU compute per run).
      - Bootstrap CIs capture test-set sampling variance, which is the
        primary source of uncertainty for large-scale fixed-split experiments.
      - With test sets of 100K+ samples, training variance is negligible
        compared to the model's systematic performance characteristics.

    Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap."

    Args:
        y_true: Ground-truth labels (n_samples,).
        y_pred: Predicted labels (n_samples,).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        confidence_level: Confidence level for CI (default 0.95 → 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys for each metric containing:
          - 'point': original point estimate
          - 'ci_lower': lower bound of CI
          - 'ci_upper': upper bound of CI
          - 'ci_width': width of CI (upper - lower)
          - 'std': standard deviation across bootstrap samples
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    alpha = 1 - confidence_level

    # Metrics to bootstrap
    metric_fns = {
        "f1_weighted": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "mcc": lambda yt, yp: matthews_corrcoef(yt, yp),
        "precision_weighted": lambda yt, yp: precision_score(yt, yp, average="weighted", zero_division=0),
        "recall_weighted": lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0),
    }

    # Compute point estimates
    point_estimates = {name: fn(y_true, y_pred) for name, fn in metric_fns.items()}

    # Bootstrap resampling
    bootstrap_scores = {name: [] for name in metric_fns}

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        for name, fn in metric_fns.items():
            try:
                score = fn(y_true_boot, y_pred_boot)
                bootstrap_scores[name].append(score)
            except Exception:
                # Some resamples might have degenerate class distributions
                bootstrap_scores[name].append(np.nan)

    # Compute CIs using percentile method
    results: Dict[str, Any] = {}
    for name in metric_fns:
        scores = np.array(bootstrap_scores[name])
        scores = scores[~np.isnan(scores)]  # drop any failed resamples

        if len(scores) == 0:
            continue

        ci_lower = float(np.percentile(scores, 100 * alpha / 2))
        ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

        results[name] = {
            "point": float(point_estimates[name]),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_upper - ci_lower,
            "std": float(np.std(scores)),
        }

    results["_meta"] = {
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "n_test_samples": n_samples,
        "random_state": random_state,
    }

    logger.info(
        f"Bootstrap CIs ({confidence_level:.0%}, {n_bootstrap} resamples): "
        f"F1(w)={results['f1_weighted']['point']:.4f} "
        f"[{results['f1_weighted']['ci_lower']:.4f}, {results['f1_weighted']['ci_upper']:.4f}], "
        f"MCC={results['mcc']['point']:.4f} "
        f"[{results['mcc']['ci_lower']:.4f}, {results['mcc']['ci_upper']:.4f}]"
    )

    return results


def _measure_gpu_energy(
    model: Any,
    X_test: np.ndarray,
    n_runs: int = 5,
) -> Optional[float]:
    """Measure average GPU power draw during inference using nvidia-smi.

    Runs inference n_runs times while sampling GPU power at ~100ms intervals.
    Returns average watts during inference, or None if GPU power monitoring
    is unavailable (CPU-only models, nvidia-smi not found, etc.).

    This is a best-effort measurement. All models run on identical hardware,
    so relative comparisons are valid even if absolute values have noise.

    Args:
        model: A fitted BaseModel instance with predict().
        X_test: Test features.
        n_runs: Number of inference passes for stable measurement.

    Returns:
        Average GPU power in watts, or None if unavailable.
    """
    try:
        # Check if nvidia-smi is available
        subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    power_readings = []
    for _ in range(n_runs):
        # Start nvidia-smi power sampling in background (100ms interval)
        smi_proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # Run inference
            model.predict(X_test)
        finally:
            smi_proc.terminate()
            try:
                stdout, _ = smi_proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                smi_proc.kill()
                stdout, _ = smi_proc.communicate()

        # Parse power readings from nvidia-smi output
        for line in stdout.decode().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    power_readings.append(float(line))
                except ValueError:
                    continue

    if not power_readings:
        return None

    return float(np.mean(power_readings))


def compute_efficiency_metrics(
    model: Any,
    X_test: np.ndarray,
    preprocessing_time_per_sample_ms: float = 0.0,
    n_runs: int = 10,
) -> Dict[str, float]:
    """Measure model efficiency: latency, throughput, memory, energy, params.

    Args:
        model: A fitted BaseModel instance with predict() and
               measure_inference_latency() methods.
        X_test: Test features (n_samples, n_features).
        preprocessing_time_per_sample_ms: Average preprocessing time per
            sample (in ms), for end-to-end latency calculation.
        n_runs: Number of inference runs for stable timing.

    Returns:
        Dict with efficiency metrics matching config.yaml evaluation.efficiency.
    """
    efficiency: Dict[str, float] = {}

    # --- Latency and throughput (from BaseModel helper) ---
    latency = model.measure_inference_latency(X_test, n_runs=n_runs)
    efficiency["inference_latency_ms_per_sample"] = latency[
        "mean_latency_ms_per_sample"
    ]
    efficiency["batch_throughput_samples_per_sec"] = latency[
        "throughput_samples_per_sec"
    ]

    # --- End-to-end latency (preprocessing + model prediction) ---
    efficiency["end_to_end_latency_ms_per_sample"] = (
        latency["mean_latency_ms_per_sample"] + preprocessing_time_per_sample_ms
    )

    # --- Training time ---
    if hasattr(model, "training_time_seconds") and model.training_time_seconds:
        efficiency["training_time_seconds"] = model.training_time_seconds

    # --- Model complexity ---
    if hasattr(model, "get_n_params"):
        efficiency["model_parameter_count"] = model.get_n_params()

    # --- Peak memory (best effort via tracemalloc) ---
    try:
        import tracemalloc

        tracemalloc.start()
        model.predict(X_test)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        efficiency["peak_memory_mb_inference"] = peak / (1024 * 1024)
    except Exception:
        efficiency["peak_memory_mb_inference"] = -1.0

    # --- GPU energy (best effort via nvidia-smi power monitoring) ---
    # Measures average GPU power draw (watts) during inference.
    # Returns -1.0 if GPU monitoring is unavailable (CPU-only models).
    gpu_watts = _measure_gpu_energy(model, X_test, n_runs=min(n_runs, 5))
    if gpu_watts is not None:
        efficiency["gpu_power_watts_inference"] = gpu_watts
        # Compute energy per sample: watts * seconds = joules
        total_time_sec = latency["total_inference_time_seconds"]
        n_samples = X_test.shape[0]
        efficiency["energy_joules_per_sample"] = (
            gpu_watts * total_time_sec / n_samples
        )
    else:
        efficiency["gpu_power_watts_inference"] = -1.0
        efficiency["energy_joules_per_sample"] = -1.0

    return efficiency


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
) -> str:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Tick labels for each class.
        output_path: File path for the saved PNG.
        title: Plot title.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        The output_path string (for convenience in result dicts).
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Confusion matrix saved to {output_path}")
    return output_path
