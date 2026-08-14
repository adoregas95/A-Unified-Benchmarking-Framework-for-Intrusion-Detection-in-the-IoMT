"""
SHAP analysis module for model interpretability.

Implements the two-stage SHAP approach from config.yaml:
    Stage 1: Lightweight global SHAP feature importance for ALL 8 models
    Stage 2: Detailed SHAP analysis (force/waterfall plots, dependence) for
             the top model per family (3 models selected by model_selection.py)

Explainer selection by model family (based on literature synthesis of 11 papers):
    - Tree-based (RF, XGBoost, LightGBM, CatBoost): TreeSHAP — exact Shapley
      values in O(TLD) time, polynomial in leaves. Used by Sohail, Alani,
      Alsharaiah, Nugraha et al.
    - Deep learning (CNN1D, BiLSTM): DeepSHAP (DeepExplainer) — combines
      DeepLIFT with Shapley values. Used by Kalakoti et al. on CICIoMT2024.
    - Transformers (FTTransformer, SAINT): KernelSHAP — model-agnostic
      approximation using weighted linear regression on sampled coalitions.
      Used by Abououf et al. Falls back to KernelSHAP for any model without
      a specialized explainer.

Literature basis: Nugraha et al. (two-stage XAI-IDS framework), Kalakoti et al.
(faithfulness/sensitivity/complexity metrics), Barredo Arrieta et al. (XAI
taxonomy), Montavon et al. (explanation quality). Integrated Gradients was
explicitly rejected for IDS per Nugraha et al. due to computational cost and
incompatibility with real-time requirements.

XAI quality metrics:
    - Faithfulness: Spearman rank correlation between SHAP importance rank and
      prediction degradation on iterative feature removal.
    - Stability: Spearman rank correlation of global feature importance across
      multiple bootstrap data subsets.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model family → explainer type mapping
# ---------------------------------------------------------------------------
MODEL_EXPLAINER_MAP = {
    "RandomForest": "tree",
    "XGBoost": "tree",
    "LightGBM": "tree",
    "CatBoost": "tree",
    "CNN1D": "deep",
    "BiLSTM": "deep",
    "FTTransformer": "kernel",
    "SAINT": "kernel",
}

# Maximum background samples for KernelSHAP (computational budget control)
KERNEL_SHAP_BACKGROUND_SIZE = 100
# Maximum test samples for SHAP computation (Stage 1 lightweight vs Stage 2 full)
STAGE1_MAX_SAMPLES = 500
STAGE2_MAX_SAMPLES = 2000
# Number of bootstrap iterations for stability measurement
STABILITY_N_BOOTSTRAPS = 5
STABILITY_BOOTSTRAP_FRAC = 0.8


def _get_explainer_type(model_name: str) -> str:
    """Resolve which SHAP explainer to use for a given model.

    Supports an environment variable override for models with known
    native-level crashes (e.g., CatBoost multiclass TreeSHAP segfault):
        export SHAP_FORCE_KERNEL=CatBoost,BiLSTM

    Args:
        model_name: Name of the model (e.g., "XGBoost", "CNN1D").

    Returns:
        One of "tree", "deep", or "kernel".
    """
    force_kernel = os.environ.get("SHAP_FORCE_KERNEL", "")
    if force_kernel and model_name in [s.strip() for s in force_kernel.split(",")]:
        logger.info(
            f"SHAP_FORCE_KERNEL override active: using KernelSHAP for {model_name}"
        )
        return "kernel"
    return MODEL_EXPLAINER_MAP.get(model_name, "kernel")


def _probe_tree_shap_safe(underlying_model: Any, X_probe: np.ndarray) -> bool:
    """Test TreeSHAP in a subprocess to detect native-level crashes (segfaults).

    CatBoost multiclass models can segfault inside TreeExplainer due to a
    known SHAP library issue with CatBoost's symmetric tree representation.
    Python's try/except cannot catch segfaults, so we run a 1-sample probe
    in a child process. If the child crashes (exit code != 0), we know
    TreeSHAP is unsafe and the caller should fall back to KernelSHAP.

    Args:
        underlying_model: The raw tree model (sklearn, xgboost, catboost, etc.).
        X_probe: A small sample array (1-2 rows) to test with.

    Returns:
        True if TreeSHAP is safe; False if the probe crashed.
    """
    import pickle
    import subprocess
    import sys
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            data_path = os.path.join(tmpdir, "data.npy")

            with open(model_path, "wb") as f:
                pickle.dump(underlying_model, f)
            np.save(data_path, X_probe[:1])

            script = (
                "import pickle, numpy as np, shap, sys\n"
                f"with open(r'{model_path}', 'rb') as f:\n"
                "    m = pickle.load(f)\n"
                f"d = np.load(r'{data_path}')\n"
                "ex = shap.TreeExplainer(m)\n"
                "ex.shap_values(d)\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                timeout=120,
                capture_output=True,
            )
            return result.returncode == 0
    except Exception as e:
        logger.warning(f"TreeSHAP safety probe failed with exception: {e}")
        return False


def _create_explainer(
    model: Any,
    model_name: str,
    X_background: np.ndarray,
) -> Any:
    """Create the appropriate SHAP explainer for a model.

    Args:
        model: Trained model instance (BaseModel subclass).
        model_name: Model name for explainer resolution.
        X_background: Background dataset for KernelSHAP / DeepSHAP.

    Returns:
        A shap.Explainer instance.

    Raises:
        RuntimeError: If explainer creation fails for all strategies.
    """
    import shap

    explainer_type = _get_explainer_type(model_name)

    if explainer_type == "tree":
        try:
            # TreeSHAP: exact Shapley values for tree-based models
            # Access the underlying sklearn/xgb/lgb/catboost estimator
            underlying = getattr(model, "model", model)

            # Safety probe: run TreeSHAP on 1 sample in a subprocess to
            # catch native segfaults (known issue with CatBoost multiclass).
            if not _probe_tree_shap_safe(underlying, X_background):
                logger.warning(
                    f"TreeSHAP safety probe crashed for {model_name}. "
                    f"Falling back to KernelSHAP."
                )
                explainer_type = "kernel"
            else:
                explainer = shap.TreeExplainer(underlying)
                logger.info(f"Created TreeExplainer for {model_name}")
                return explainer
        except Exception as e:
            logger.warning(
                f"TreeExplainer failed for {model_name}: {e}. "
                f"Falling back to KernelSHAP."
            )
            explainer_type = "kernel"

    if explainer_type == "deep":
        try:
            # DeepSHAP: DeepLIFT + Shapley for neural networks
            import torch

            underlying = getattr(model, "model", model)
            # DeepExplainer needs torch tensors as background
            if isinstance(X_background, np.ndarray):
                bg_tensor = torch.FloatTensor(X_background)
                if next(underlying.parameters()).is_cuda:
                    bg_tensor = bg_tensor.cuda()
            else:
                bg_tensor = X_background

            explainer = shap.DeepExplainer(underlying, bg_tensor)
            logger.info(f"Created DeepExplainer for {model_name}")
            return explainer
        except Exception as e:
            logger.warning(
                f"DeepExplainer failed for {model_name}: {e}. "
                f"Falling back to KernelSHAP."
            )
            explainer_type = "kernel"

    # KernelSHAP: model-agnostic fallback
    # Uses predict_proba as the model function for richer explanations
    def predict_fn(X):
        return model.predict_proba(X)

    explainer = shap.KernelExplainer(predict_fn, X_background)
    logger.info(f"Created KernelExplainer for {model_name}")
    return explainer


def _compute_shap_values(
    explainer: Any,
    X_explain: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Compute SHAP values, handling multiclass output shape.

    For binary/multiclass, SHAP may return shape (n_samples, n_features, n_classes).
    We reduce to (n_samples, n_features) by taking the mean absolute across classes,
    which gives a single importance score per feature per sample.

    Args:
        explainer: A SHAP explainer instance.
        X_explain: Samples to explain (n_samples, n_features).
        model_name: For logging.

    Returns:
        SHAP values of shape (n_samples, n_features).
    """
    import shap

    logger.info(
        f"Computing SHAP values for {model_name} on {X_explain.shape[0]} samples..."
    )
    t0 = time.time()

    explainer_type = _get_explainer_type(model_name)

    if explainer_type == "deep":
        import torch
        import torch.nn as nn

        X_tensor = torch.FloatTensor(X_explain)
        # Move to same device as model
        try:
            device = next(explainer.model.parameters()).device
            X_tensor = X_tensor.to(device)
        except Exception:
            pass

        # cuDNN LSTM backward only works in training mode, but we must
        # disable dropout to keep behaviour deterministic (no random
        # masking).  Save and restore original state afterward.
        was_training = explainer.model.training
        dropout_states = {}
        explainer.model.train()
        for name, module in explainer.model.named_modules():
            if isinstance(module, nn.Dropout):
                dropout_states[name] = module.p
                module.p = 0.0

        try:
            # check_additivity=False: DeepLIFT approximation can have
            # large residuals for multiclass CNNs/LSTMs — a known SHAP
            # limitation.  The feature importance ranking remains valid.
            shap_values = explainer.shap_values(
                X_tensor, check_additivity=False
            )
        finally:
            # Restore original model state
            for name, module in explainer.model.named_modules():
                if name in dropout_states:
                    module.p = dropout_states[name]
            if not was_training:
                explainer.model.eval()
    else:
        shap_values = explainer.shap_values(X_explain)

    elapsed = time.time() - t0
    logger.info(f"SHAP values computed in {elapsed:.1f}s for {model_name}")

    # Handle multiclass: shape may be (n_classes, n_samples, n_features)
    # or list of arrays, or (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # List of arrays, one per class — stack and take mean abs
        stacked = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)
        shap_values = np.mean(np.abs(stacked), axis=-1)
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes)
        shap_values = np.mean(np.abs(shap_values), axis=-1)
    # else: already (n_samples, n_features) — binary or regression

    return shap_values


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Rank features by mean absolute SHAP value (global importance).

    Args:
        shap_values: SHAP values of shape (n_samples, n_features).
        feature_names: Feature names matching the columns.

    Returns:
        List of (feature_name, mean_abs_shap) sorted descending by importance.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranking = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranking


# ---------------------------------------------------------------------------
# Stage 1: Lightweight global SHAP for all models
# ---------------------------------------------------------------------------

def run_stage1_shap(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    max_samples: int = STAGE1_MAX_SAMPLES,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Stage 1: Lightweight global SHAP feature importance for a single model.

    Applied to ALL 8 models. Computes mean |SHAP| per feature and generates
    a bar plot of global importance rankings.

    Args:
        model: Trained BaseModel instance.
        model_name: Model identifier (e.g., "XGBoost").
        X_test: Test features (n_samples, n_features).
        feature_names: Feature names.
        output_dir: Directory for saving plots and results.
        max_samples: Cap on samples to explain (controls compute cost).
        random_state: For reproducible subsampling.

    Returns:
        Dict with keys:
            - global_importance: List[(feature_name, mean_abs_shap)]
            - shap_values: np.ndarray (n_explained_samples, n_features)
            - computation_time_seconds: float
            - explainer_type: str
            - n_samples_explained: int
    """
    import shap

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    t0 = time.time()

    # Subsample test set if needed
    n = X_test.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_explain = X_test[idx]
        logger.info(f"Stage 1 SHAP: subsampled {max_samples}/{n} test samples")
    else:
        X_explain = X_test

    # Create background data for KernelSHAP/DeepSHAP
    bg_size = min(KERNEL_SHAP_BACKGROUND_SIZE, X_explain.shape[0])
    bg_idx = rng.choice(X_explain.shape[0], size=bg_size, replace=False)
    X_background = X_explain[bg_idx]

    # Create explainer and compute values
    explainer = _create_explainer(model, model_name, X_background)
    shap_values = _compute_shap_values(explainer, X_explain, model_name)

    # Global importance ranking
    importance = get_global_feature_importance(shap_values, feature_names)

    computation_time = time.time() - t0

    # --- Generate bar plot of global importance ---
    _save_global_importance_bar_plot(
        importance,
        model_name=model_name,
        output_path=os.path.join(output_dir, f"{model_name}_shap_global_bar.png"),
        top_k=20,
    )

    # --- Generate SHAP summary plot (beeswarm) ---
    try:
        _apply_pub_style()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=feature_names,
            show=False,
            max_display=20,
            cmap=_BEESWARM_CMAP,
            alpha=0.7,
            plot_size=None,  # Use our own figure size
        )
        # Polish the current axes after SHAP draws into them
        cur_ax = plt.gca()
        cur_ax.set_title(f"SHAP Feature Impact — {model_name}",
                         fontweight="bold", fontsize=13, pad=12)
        cur_ax.set_xlabel("SHAP value (impact on model output)",
                          fontweight="bold", fontsize=11)
        cur_ax.spines["top"].set_visible(False)
        cur_ax.spines["right"].set_visible(False)
        cur_ax.tick_params(axis="both", labelsize=10)

        summary_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close("all")
        logger.info(f"SHAP summary plot saved to {summary_path}")
    except Exception as e:
        logger.warning(f"Could not generate SHAP summary plot: {e}")

    result = {
        "global_importance": importance,
        "shap_values": shap_values,
        "computation_time_seconds": computation_time,
        "explainer_type": _get_explainer_type(model_name),
        "n_samples_explained": X_explain.shape[0],
    }

    logger.info(
        f"Stage 1 SHAP for {model_name}: {computation_time:.1f}s, "
        f"{X_explain.shape[0]} samples, top feature = {importance[0][0]}"
    )

    return result


# ---------------------------------------------------------------------------
# Stage 2: Full SHAP analysis for top model per family
# ---------------------------------------------------------------------------

def run_stage2_shap(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    output_dir: str,
    max_samples: int = STAGE2_MAX_SAMPLES,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Stage 2: Detailed SHAP analysis for a top model per family.

    Applied to 3 models (best tree-based, best DL, best transformer).
    Generates force/waterfall plots for individual instances, dependence
    plots for top features, and computes XAI quality metrics.

    Args:
        model: Trained BaseModel instance.
        model_name: Model identifier.
        X_test: Test features.
        y_test: True labels (for instance selection — correct, misclassified).
        feature_names: Feature names.
        class_names: Class label names.
        output_dir: Directory for saving plots and results.
        max_samples: Cap on samples for SHAP computation.
        random_state: For reproducibility.

    Returns:
        Dict with keys:
            - global_importance: List[(feature_name, mean_abs_shap)]
            - shap_values: np.ndarray
            - instance_explanations: List[Dict] (force/waterfall for key instances)
            - faithfulness_score: float (XAI quality)
            - stability_score: float (XAI quality)
            - computation_time_seconds: float
    """
    import shap

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    t0 = time.time()

    # Subsample
    n = X_test.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_explain = X_test[idx]
        y_explain = y_test[idx]
    else:
        X_explain = X_test
        y_explain = y_test
        idx = np.arange(n)

    # Background data
    bg_size = min(KERNEL_SHAP_BACKGROUND_SIZE, X_explain.shape[0])
    bg_idx = rng.choice(X_explain.shape[0], size=bg_size, replace=False)
    X_background = X_explain[bg_idx]

    # Compute SHAP values
    explainer = _create_explainer(model, model_name, X_background)
    shap_values = _compute_shap_values(explainer, X_explain, model_name)

    # Global importance
    importance = get_global_feature_importance(shap_values, feature_names)

    # --- Instance-level explanations ---
    instance_explanations = _generate_instance_explanations(
        model, model_name, X_explain, y_explain, shap_values,
        feature_names, class_names, output_dir, rng,
    )

    # --- Dependence plots for top 5 features ---
    _save_dependence_plots(
        shap_values, X_explain, feature_names, importance[:5],
        model_name, output_dir,
    )

    # --- XAI quality metrics ---
    faithfulness = compute_faithfulness(
        model, X_explain, shap_values, feature_names,
    )
    stability = compute_stability(
        model, model_name, X_explain, feature_names, rng,
    )

    computation_time = time.time() - t0

    # Global plots (same as Stage 1 but on more samples)
    _save_global_importance_bar_plot(
        importance, model_name,
        os.path.join(output_dir, f"{model_name}_shap_global_bar.png"),
        top_k=20,
    )

    result = {
        "global_importance": importance,
        "shap_values": shap_values,
        "subsample_idx": idx,
        "instance_explanations": instance_explanations,
        "faithfulness_score": faithfulness,
        "stability_score": stability,
        "computation_time_seconds": computation_time,
        "explainer_type": _get_explainer_type(model_name),
        "n_samples_explained": X_explain.shape[0],
    }

    logger.info(
        f"Stage 2 SHAP for {model_name}: {computation_time:.1f}s, "
        f"faithfulness={faithfulness:.4f}, stability={stability:.4f}"
    )

    return result


# ---------------------------------------------------------------------------
# XAI quality metrics (Kalakoti et al. framework)
# ---------------------------------------------------------------------------

def compute_faithfulness(
    model: Any,
    X_test: np.ndarray,
    shap_values: np.ndarray,
    feature_names: List[str],
    n_steps: int = 10,
) -> float:
    """Compute faithfulness correlation (Kalakoti et al.).

    Measures correlation between SHAP importance rank and prediction
    degradation when features are iteratively removed (set to zero).

    Process:
        1. Rank features by mean |SHAP| (descending importance).
        2. Iteratively mask the top-k features (k = 1, 2, ..., n_steps).
        3. At each step, compute mean prediction confidence drop.
        4. Faithfulness = Spearman correlation between step number and
           confidence drop. High correlation means SHAP correctly
           identifies the features that matter most.

    Args:
        model: Trained model with predict_proba().
        X_test: Test features (n_samples, n_features).
        shap_values: SHAP values (n_samples, n_features).
        feature_names: Feature names.
        n_steps: Number of feature removal steps.

    Returns:
        Faithfulness score in [-1, 1]. Higher is better.
    """
    from scipy.stats import spearmanr

    # Get global importance ranking
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_order = np.argsort(-mean_abs)  # most important first

    # Original predictions — confidence for predicted class
    try:
        proba_orig = model.predict_proba(X_test)
        pred_classes = np.argmax(proba_orig, axis=1)
        orig_confidence = proba_orig[np.arange(len(pred_classes)), pred_classes]
    except Exception:
        logger.warning("predict_proba failed; faithfulness returns 0.0")
        return 0.0

    n_features = X_test.shape[1]
    step_size = max(1, n_features // n_steps)
    degradation = []

    for step in range(1, n_steps + 1):
        n_mask = min(step * step_size, n_features)
        features_to_mask = importance_order[:n_mask]

        X_masked = X_test.copy()
        X_masked[:, features_to_mask] = 0.0

        try:
            proba_masked = model.predict_proba(X_masked)
            masked_confidence = proba_masked[
                np.arange(len(pred_classes)), pred_classes
            ]
            mean_drop = np.mean(orig_confidence - masked_confidence)
        except Exception:
            mean_drop = 0.0

        degradation.append(mean_drop)

    # Spearman correlation: step index vs degradation
    # Perfect faithfulness: removing more important features causes more degradation
    steps = np.arange(1, n_steps + 1)
    if np.std(degradation) < 1e-12:
        return 0.0

    corr, _ = spearmanr(steps, degradation)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_stability(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    feature_names: List[str],
    rng: np.random.RandomState,
    n_bootstraps: int = STABILITY_N_BOOTSTRAPS,
    bootstrap_frac: float = STABILITY_BOOTSTRAP_FRAC,
) -> float:
    """Compute explanation stability via bootstrap resampling.

    Measures how consistent the global feature importance rankings are
    across different data subsets. High stability means explanations are
    not sensitive to the specific test samples used.

    Process:
        1. Draw n_bootstraps subsets of X_test (each bootstrap_frac %).
        2. Compute SHAP global importance for each subset.
        3. Stability = mean pairwise Spearman rank correlation of
           importance rankings across all bootstrap pairs.

    Args:
        model: Trained model.
        model_name: For explainer selection.
        X_test: Test features.
        feature_names: Feature names.
        rng: Random state.
        n_bootstraps: Number of bootstrap iterations.
        bootstrap_frac: Fraction of samples per bootstrap.

    Returns:
        Stability score in [-1, 1]. Higher is better.
    """
    from scipy.stats import spearmanr

    n = X_test.shape[0]
    bootstrap_size = int(n * bootstrap_frac)
    if bootstrap_size < 10:
        logger.warning("Too few samples for stability bootstrap; returning 0.0")
        return 0.0

    rankings = []

    for i in range(n_bootstraps):
        idx = rng.choice(n, size=bootstrap_size, replace=True)
        X_boot = X_test[idx]

        # Create background from this bootstrap
        bg_size = min(KERNEL_SHAP_BACKGROUND_SIZE, bootstrap_size)
        bg_idx = rng.choice(bootstrap_size, size=bg_size, replace=False)
        X_bg = X_boot[bg_idx]

        try:
            explainer = _create_explainer(model, model_name, X_bg)
            # Use a small subset for speed
            explain_size = min(100, bootstrap_size)
            explain_idx = rng.choice(bootstrap_size, size=explain_size, replace=False)
            sv = _compute_shap_values(explainer, X_boot[explain_idx], model_name)
            mean_abs = np.mean(np.abs(sv), axis=0)
            rankings.append(mean_abs)
        except Exception as e:
            logger.warning(f"Stability bootstrap {i} failed: {e}")
            continue

    if len(rankings) < 2:
        return 0.0

    # Pairwise Spearman rank correlations
    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            corr, _ = spearmanr(rankings[i], rankings[j])
            if not np.isnan(corr):
                correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 0.0


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# Publication-quality style settings
_PUB_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

# Color palette for plots (navy-to-teal gradient for bars)
_BAR_COLOR_PRIMARY = "#1B365D"      # Navy (matches dissertation theme)
_BAR_COLOR_ACCENT = "#2E86AB"       # Steel blue
_POS_SHAP_COLOR = "#1B365D"         # Navy for positive SHAP contributions
_NEG_SHAP_COLOR = "#C0392B"         # Crimson for negative SHAP contributions
_BEESWARM_CMAP = "RdBu_r"          # Red (high) → Blue (low), reversed


def _apply_pub_style():
    """Apply publication-quality matplotlib + seaborn style settings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(
            style="ticks",
            context="paper",
            font="serif",
            rc=_PUB_STYLE,
        )
    except ImportError:
        # Seaborn not available — fall back to pure matplotlib
        plt.rcParams.update(_PUB_STYLE)


def _save_global_importance_bar_plot(
    importance: List[Tuple[str, float]],
    model_name: str,
    output_path: str,
    top_k: int = 20,
) -> None:
    """Save horizontal bar plot of top-k features by mean |SHAP|."""
    _apply_pub_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    top = importance[:top_k]
    names = [t[0] for t in top][::-1]
    values = [t[1] for t in top][::-1]

    # Create gradient colors: lighter for lower values, darker for higher
    max_val = max(values) if values else 1.0
    norm_vals = [v / max_val for v in values]
    cmap = LinearSegmentedColormap.from_list(
        "importance", [_BAR_COLOR_ACCENT, _BAR_COLOR_PRIMARY]
    )
    colors = [cmap(nv) for nv in norm_vals]

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.32)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.3,
                   height=0.7)

    ax.set_xlabel("Mean |SHAP value|", fontweight="bold")
    ax.set_title(f"Global Feature Importance — {model_name}",
                 fontweight="bold", pad=12)

    # Subtle grid on x-axis only
    ax.xaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color="#444444")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Global importance bar plot saved to {output_path}")


def _save_dependence_plots(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    top_features: List[Tuple[str, float]],
    model_name: str,
    output_dir: str,
) -> None:
    """Save SHAP dependence plots for top features."""
    import shap
    _apply_pub_style()
    import matplotlib.pyplot as plt

    for feat_name, _ in top_features:
        try:
            feat_idx = feature_names.index(feat_name)
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(
                feat_idx,
                shap_values,
                X_explain,
                feature_names=feature_names,
                show=False,
                ax=ax,
                alpha=0.5,
            )
            ax.set_title(f"SHAP Dependence — {model_name}: {feat_name}",
                         fontweight="bold", fontsize=13, pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=10)

            dep_path = os.path.join(
                output_dir,
                f"{model_name}_shap_dep_{feat_name.replace(' ', '_')}.png",
            )
            fig.savefig(dep_path, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Dependence plot for {feat_name} failed: {e}")


def _generate_instance_explanations(
    model: Any,
    model_name: str,
    X_explain: np.ndarray,
    y_explain: np.ndarray,
    shap_values: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    output_dir: str,
    rng: np.random.RandomState,
    n_instances: int = 10,
) -> List[Dict[str, Any]]:
    """Generate per-instance SHAP explanations for interesting samples.

    Selects a mix of correctly classified and misclassified instances
    to show how the model's reasoning differs between successes and failures.

    Args:
        model: Trained model.
        model_name: For labeling.
        X_explain: Features of explained samples.
        y_explain: True labels.
        shap_values: Precomputed SHAP values.
        feature_names: Feature names.
        class_names: Class label names.
        output_dir: Where to save plots.
        rng: Random state.
        n_instances: How many instances to explain.

    Returns:
        List of dicts with instance-level explanation metadata.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = model.predict(X_explain)
    correct_mask = predictions == y_explain
    misclassified_mask = ~correct_mask

    # Select ~half correct, ~half misclassified (or as many as available)
    n_correct = min(n_instances // 2, int(correct_mask.sum()))
    n_misclass = min(n_instances - n_correct, int(misclassified_mask.sum()))

    selected_idx = []
    if n_correct > 0:
        correct_indices = np.where(correct_mask)[0]
        selected_idx.extend(
            rng.choice(correct_indices, size=n_correct, replace=False).tolist()
        )
    if n_misclass > 0:
        misclass_indices = np.where(misclassified_mask)[0]
        selected_idx.extend(
            rng.choice(misclass_indices, size=n_misclass, replace=False).tolist()
        )

    explanations = []
    for i, sample_idx in enumerate(selected_idx):
        sv = shap_values[sample_idx]
        true_label = int(y_explain[sample_idx])
        pred_label = int(predictions[sample_idx])
        is_correct = true_label == pred_label

        # Top contributing features for this instance
        abs_sv = np.abs(sv)
        top_idx = np.argsort(-abs_sv)[:10]
        top_features = [
            (feature_names[j], float(sv[j])) for j in top_idx
        ]

        # Save waterfall-style bar plot for this instance
        try:
            _apply_pub_style()
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_names_top = [feature_names[j] for j in top_idx][::-1]
            feat_vals_top = [float(sv[j]) for j in top_idx][::-1]
            colors = [_NEG_SHAP_COLOR if v < 0 else _POS_SHAP_COLOR
                      for v in feat_vals_top]
            ax.barh(feat_names_top, feat_vals_top, color=colors,
                    edgecolor="white", linewidth=0.3, height=0.65)
            ax.axvline(x=0, color="#333333", linewidth=0.8)

            true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
            pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
            status = "CORRECT" if is_correct else "MISCLASSIFIED"

            ax.set_title(
                f"{model_name} — Instance #{sample_idx} [{status}]\n"
                f"True: {true_name}  |  Predicted: {pred_name}",
                fontweight="bold", fontsize=12, pad=10,
            )
            ax.set_xlabel("SHAP value (impact on prediction)",
                          fontweight="bold", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()

            plot_path = os.path.join(
                output_dir,
                f"{model_name}_instance_{i}_{status.lower()}.png",
            )
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Instance plot {i} failed: {e}")
            plot_path = None

        explanations.append({
            "instance_idx": int(sample_idx),
            "true_label": true_label,
            "predicted_label": pred_label,
            "is_correct": is_correct,
            "top_features": top_features,
            "plot_path": plot_path,
        })

    return explanations


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline for a model
# ---------------------------------------------------------------------------

def run_shap_analysis(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    stage: int = 1,
    y_test: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Entry point for SHAP analysis — dispatches to Stage 1 or Stage 2.

    Args:
        model: Trained BaseModel instance.
        model_name: Model identifier.
        X_test: Test features.
        feature_names: Feature names.
        output_dir: Output directory for plots/results.
        stage: 1 for lightweight global, 2 for full detailed analysis.
        y_test: True labels (required for Stage 2).
        class_names: Class names (required for Stage 2).
        random_state: For reproducibility.

    Returns:
        Stage-specific results dict.
    """
    if stage == 1:
        return run_stage1_shap(
            model, model_name, X_test, feature_names,
            output_dir, random_state=random_state,
        )
    elif stage == 2:
        if y_test is None or class_names is None:
            raise ValueError(
                "Stage 2 SHAP requires y_test and class_names."
            )
        return run_stage2_shap(
            model, model_name, X_test, y_test, feature_names,
            class_names, output_dir, random_state=random_state,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
