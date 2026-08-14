"""
Cross-dataset generalization evaluation for the IoMT IDS Framework.

Evaluates models trained on CICIoMT2024 against three cross-dataset targets:
    - CIC-BoT-IoT
    - CIC-IoT-DIAD-2024
    - CIC-ToN-IoT

Zero-shot transfer: the model is evaluated directly on the cross-dataset
target without any retraining or fine-tuning. This tests whether the
patterns learned on CICIoMT2024 generalize to completely unseen network
environments captured by different teams at different times.

Applicable tasks:
    - Task 2 (binary: Benign vs Attack) — all 3 targets
    - Task 6 (families) — shared families only; novel classes analyzed separately
    - Task 19 — NOT applicable (CICIoMT2024-specific individual attack types)

Primary metric for cross-dataset: Macro F1 (equal weight to all classes,
since class distributions differ drastically between datasets).

Design principles:
    - The cross-dataset target is loaded in "full_test" mode — the ENTIRE
      dataset becomes the test set (no train/test split).
    - Preprocessing uses the PRIMARY dataset's scaler and feature selection
      indices to ensure identical feature transformations.
    - For Task 6, only shared families are included in the primary evaluation.
      Novel classes (families that exist in the target but not in CICIoMT2024)
      are analyzed separately to understand how the model handles unseen
      attack categories.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path for clean imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.data_loader import load_dataset
from evaluation.metrics import compute_all_metrics, save_confusion_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared family labels between CICIoMT2024 and each cross-dataset target
# ---------------------------------------------------------------------------
CICIOMT2024_FAMILIES = {"Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"}

SHARED_FAMILIES = {
    "CIC-BoT-IoT": {"Benign", "DDoS", "DoS", "Recon"},
    "CIC-IoT-DIAD-2024": {"Benign", "DDoS", "DoS", "Recon", "Spoofing"},
    "CIC-ToN-IoT": {"Benign", "DDoS", "DoS", "Recon"},
}

NOVEL_FAMILIES = {
    "CIC-BoT-IoT": {"Theft"},
    "CIC-IoT-DIAD-2024": {"BruteForce", "Mirai", "Web-Based"},
    "CIC-ToN-IoT": {"Injection", "MITM", "Password", "Ransomware", "XSS", "Backdoor"},
}


def load_cross_dataset_target(
    dataset: str,
    task: int,
    data_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load a cross-dataset target in full_test mode.

    The entire target dataset is returned as test data. No train/test
    split is performed because the model was trained on CICIoMT2024.

    Args:
        dataset: One of "CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT".
        task: Classification task (2 or 6). Task 19 is not applicable.
        data_root: Optional path to data/ directory.

    Returns:
        X_test: Test features DataFrame (76 features, full dataset).
        y_test: Test labels Series.
        metadata: Dataset metadata dict.

    Raises:
        ValueError: If task is 19 or dataset is the primary dataset.
    """
    if task == 19:
        raise ValueError(
            "Task 19 (individual attack types) is NOT applicable for "
            "cross-dataset generalization. Use task 2 (binary) or task 6 "
            "(families)."
        )

    if dataset == "CICIoMT2024":
        raise ValueError(
            "CICIoMT2024 is the primary training dataset, not a "
            "cross-dataset target. Use one of: CIC-BoT-IoT, "
            "CIC-IoT-DIAD-2024, CIC-ToN-IoT."
        )

    logger.info(f"Loading cross-dataset target: {dataset}, task={task}")
    X_train, y_train, X_test, y_test, metadata = load_dataset(
        dataset=dataset,
        task=task,
        data_root=data_root,
        cross_dataset_mode="full_test",
    )

    # Verify full_test mode: training set should be empty
    assert len(X_train) == 0, (
        f"Expected empty training set in full_test mode, got {len(X_train)} rows"
    )

    logger.info(
        f"  Loaded {len(X_test)} samples, {X_test.shape[1]} features, "
        f"{metadata['n_classes']} classes: {metadata['class_names']}"
    )

    return X_test, y_test, metadata


def preprocess_cross_dataset(
    X_target: pd.DataFrame,
    y_target: pd.Series,
    primary_metadata: Dict[str, Any],
    primary_cache_dir: str,
    primary_task: int,
    target_dataset: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Apply primary dataset's preprocessing to cross-dataset target.

    Uses the scaler and feature selection indices from the primary dataset
    (CICIoMT2024) preprocessing to transform the cross-dataset target
    identically. This ensures the model receives features in the same
    scale and order as during training.

    Args:
        X_target: Raw features from cross-dataset target (DataFrame).
        y_target: Raw labels from cross-dataset target (Series).
        primary_metadata: Metadata from the primary preprocessing cache.
        primary_cache_dir: Path to preprocessing/cache.
        primary_task: The task the primary model was trained on (2 or 6).
        target_dataset: Name of the cross-dataset target.

    Returns:
        X_processed: Preprocessed feature array.
        y_encoded: Integer-encoded labels.
        shared_class_names: Names of classes present in both datasets.
        filter_info: Dict with filtering statistics.
    """
    from preprocessing.preprocessing import clean_data, load_preprocessed_data

    # Load primary dataset metadata to get scaler info and feature names
    _, _, _, _, _, _, p_meta = load_preprocessed_data(
        primary_cache_dir, "CICIoMT2024", primary_task
    )
    primary_feature_names = p_meta["feature_names"]

    logger.info(
        f"Primary model expects {len(primary_feature_names)} features: "
        f"{primary_feature_names[:5]}..."
    )

    # --- Step 1: Feature alignment ---
    # Ensure the target has exactly the features the primary model expects.
    # All four datasets share the same 76 CICFlowMeter features after
    # column name standardization in data_loader.py. Feature selection
    # (MI threshold) may have reduced this to fewer features.
    available_features = X_target.columns.tolist()
    missing_features = [f for f in primary_feature_names if f not in available_features]
    if missing_features:
        logger.warning(
            f"Cross-dataset target is missing {len(missing_features)} features: "
            f"{missing_features}. Filling with zeros."
        )
        for f in missing_features:
            X_target[f] = 0.0

    # Select only the features used by the primary model, in the same order
    X_aligned = X_target[primary_feature_names].copy()

    # --- Step 2: Data cleaning (replace Inf/NaN) ---
    # Use column-wise medians from the target data itself for cleaning
    # (this is safe because cleaning is just replacing invalid values,
    # not fitting a model-dependent transformation)
    #
    # Safety: force all columns to numeric first. Some cross-dataset CSVs
    # have mixed-type columns where label strings bleed into feature cells
    # due to malformed rows (e.g., CIC-BoT-IoT columns 19/59).
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce")
    X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan)
    col_medians = X_aligned.median()
    X_aligned = X_aligned.fillna(col_medians)
    X_arr = X_aligned.values.astype(np.float32)

    # --- Step 3: Feature scaling ---
    # We need to apply the SAME scaler that was fit on CICIoMT2024 training data.
    # The scaler parameters are stored in the preprocessing cache metadata.
    scaler_type = p_meta.get("scaling", "robust")
    logger.info(f"Applying {scaler_type} scaling from primary dataset")

    # Reconstruct scaler from metadata. The scaler must come from the
    # primary (source) dataset to avoid information leakage — fitting a
    # fresh scaler on the target data would use target statistics that are
    # unavailable in a true zero-shot deployment scenario (Giannakidis
    # et al., 2025; standard ML practice per Hastie et al., 2009).
    scaler_params = p_meta.get("scaler_params", None)
    if scaler_params is not None:
        X_scaled = _apply_saved_scaler(X_arr, scaler_type, scaler_params)
    else:
        raise RuntimeError(
            "scaler_params not found in primary metadata. "
            "Run 'python3 scripts/backfill_scaler_params.py' first to "
            "extract scaler parameters from the CICIoMT2024 training data."
        )

    # --- Step 4: Label encoding and filtering ---
    # For task 6, filter to shared families only
    task = primary_task
    if task == 6:
        shared = SHARED_FAMILIES.get(target_dataset, set())
        novel = NOVEL_FAMILIES.get(target_dataset, set())

        shared_mask = y_target.isin(shared)
        novel_mask = y_target.isin(novel)

        n_shared = int(shared_mask.sum())
        n_novel = int(novel_mask.sum())
        n_total = len(y_target)

        logger.info(
            f"Task 6 label filtering: {n_shared}/{n_total} samples in shared "
            f"families ({shared}), {n_novel} in novel families ({novel})"
        )

        # Primary evaluation: shared families only
        X_shared = X_scaled[shared_mask.values]
        y_shared = y_target[shared_mask].reset_index(drop=True)

        # Novel family analysis (separate)
        X_novel = X_scaled[novel_mask.values] if n_novel > 0 else np.array([])
        y_novel = y_target[novel_mask].reset_index(drop=True) if n_novel > 0 else pd.Series([])
    else:
        # Task 2 (binary): all samples have Benign or Attack label
        X_shared = X_scaled
        y_shared = y_target
        X_novel = np.array([])
        y_novel = pd.Series([])
        n_shared = len(y_target)
        n_novel = 0
        shared = {"Benign", "Attack"}
        novel = set()

    # Encode labels using the primary dataset's class ordering
    # to ensure consistent label indexing
    from sklearn.preprocessing import LabelEncoder
    primary_classes = p_meta.get("label_classes", None)
    le = LabelEncoder()
    if primary_classes is not None:
        le.classes_ = np.array(primary_classes)
    else:
        le.fit(y_shared)

    # Only encode labels that exist in the primary encoder
    try:
        y_encoded = le.transform(y_shared)
    except ValueError as e:
        # Some target labels not in primary encoder — filter them out
        logger.warning(f"Label encoding issue: {e}. Filtering unknown labels.")
        known_mask = y_shared.isin(le.classes_)
        X_shared = X_shared[known_mask.values]
        y_shared = y_shared[known_mask].reset_index(drop=True)
        y_encoded = le.transform(y_shared)

    shared_class_names = list(le.classes_)

    filter_info = {
        "total_samples": len(y_target),
        "shared_samples": n_shared,
        "novel_samples": n_novel,
        "shared_families": sorted(shared),
        "novel_families": sorted(novel),
        "X_novel": X_novel,
        "y_novel": y_novel,
        "label_encoder": le,
    }

    return X_shared, y_encoded, shared_class_names, filter_info


def _apply_saved_scaler(
    X: np.ndarray,
    scaler_type: str,
    scaler_params: Dict[str, Any],
) -> np.ndarray:
    """Apply saved scaler parameters to transform data.

    Args:
        X: Raw features (n_samples, n_features).
        scaler_type: Type of scaler ("robust", "minmax", "standard").
        scaler_params: Dict with scaler-specific parameters.

    Returns:
        Scaled features.
    """
    if scaler_type == "robust":
        center = np.array(scaler_params["center_"])
        scale = np.array(scaler_params["scale_"])
        scale[scale == 0] = 1.0  # avoid division by zero
        return (X - center) / scale
    elif scaler_type == "minmax":
        data_min = np.array(scaler_params["data_min_"])
        data_max = np.array(scaler_params["data_max_"])
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0
        return (X - data_min) / data_range
    elif scaler_type == "standard":
        mean = np.array(scaler_params["mean_"])
        std = np.array(scaler_params["scale_"])
        std[std == 0] = 1.0
        return (X - mean) / std
    else:
        logger.warning(f"Unknown scaler type '{scaler_type}', returning raw data")
        return X


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------

def evaluate_zero_shot(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_name: str,
    target_dataset: str,
    task: int,
    output_dir: str,
    shared_class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model on cross-dataset target (zero-shot transfer).

    No retraining or fine-tuning — the model is used as-is from its
    CICIoMT2024 training.

    Args:
        model: Trained BaseModel instance (loaded from checkpoint).
        X_test: Preprocessed test features from cross-dataset target.
        y_test: Integer-encoded labels.
        class_names: Class names for reporting (all primary classes).
        model_name: Model identifier.
        target_dataset: Name of cross-dataset target.
        task: Classification task (2 or 6).
        output_dir: Where to save results.
        shared_class_names: For task 6, the subset of class_names that
            actually exist in the target dataset. Macro F1 is computed
            over these classes only (avoiding deflation from absent
            classes whose F1 is trivially zero).

    Returns:
        Dict with test metrics, confusion matrix path, and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Zero-shot evaluation: {model_name} on {target_dataset} "
        f"(task={task}, {X_test.shape[0]} samples)"
    )

    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    # Compute metrics — macro F1 is the primary metric for cross-dataset
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=class_names,
    )

    # For task 6, compute corrected macro F1 over shared classes only.
    # Some primary classes (e.g. MQTT, Spoofing) may not exist in the
    # target dataset, giving them F1=0 by construction and deflating
    # the macro average. The corrected metric averages only over classes
    # that are genuinely present in the target.
    if shared_class_names is not None and len(shared_class_names) < len(class_names):
        shared_f1_values = [
            metrics["per_class_f1"].get(c, 0.0) for c in shared_class_names
        ]
        metrics["f1_macro_shared"] = float(
            sum(shared_f1_values) / len(shared_f1_values)
        ) if shared_f1_values else 0.0
        metrics["n_shared_classes"] = len(shared_class_names)
        metrics["shared_class_names"] = shared_class_names
        logger.info(
            f"  F1(macro, {len(shared_class_names)} shared): "
            f"{metrics['f1_macro_shared']:.4f}  "
            f"[all {len(class_names)}: {metrics['f1_macro']:.4f}]"
        )

    logger.info(
        f"  F1(macro): {metrics['f1_macro']:.4f}, "
        f"F1(weighted): {metrics['f1_weighted']:.4f}, "
        f"MCC: {metrics['mcc']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    # Save confusion matrix
    cm_path = os.path.join(
        output_dir,
        f"{model_name}_{target_dataset}_task{task}_confusion_matrix.png",
    )
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_path=cm_path,
        title=f"Zero-Shot: {model_name} → {target_dataset} (Task {task})",
    )

    result = {
        "model": model_name,
        "target_dataset": target_dataset,
        "source_dataset": "CICIoMT2024",
        "task": task,
        "transfer_type": "zero_shot",
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "test_metrics": {
            k: v for k, v in metrics.items()
            if k != "classification_report_str"
        },
        "inference_time_seconds": inference_time,
        "confusion_matrix_path": cm_path,
        "classification_report": metrics.get("classification_report_str", ""),
    }

    # Save results JSON
    results_path = os.path.join(
        output_dir,
        f"{model_name}_{target_dataset}_task{task}_results.json",
    )
    # Make a serializable copy
    result_save = {k: v for k, v in result.items() if k != "classification_report"}
    with open(results_path, "w") as f:
        json.dump(result_save, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Save classification report text
    report_path = os.path.join(
        output_dir,
        f"{model_name}_{target_dataset}_task{task}_report.txt",
    )
    with open(report_path, "w") as f:
        f.write(f"Zero-Shot Transfer: {model_name} → {target_dataset}\n")
        f.write(f"Task: {task}, Source: CICIoMT2024\n")
        f.write(f"Test samples: {X_test.shape[0]}\n\n")
        f.write(metrics.get("classification_report_str", ""))

    return result


def analyze_novel_classes(
    model: Any,
    X_novel: np.ndarray,
    y_novel: pd.Series,
    model_name: str,
    target_dataset: str,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    """Analyze how the model handles novel attack classes.

    Novel classes are attack families in the cross-dataset target that
    do not exist in CICIoMT2024 (e.g., Ransomware, XSS in CIC-ToN-IoT).
    The model has never seen these classes during training, so it must
    assign them to one of its known classes. This analysis reveals:
        - Which known classes absorb novel attacks
        - Prediction confidence distribution for novel samples
        - Whether the model can at least detect them as attacks (task 2 level)

    Args:
        model: Trained model.
        X_novel: Features of novel-class samples.
        y_novel: True family labels (strings) for novel samples.
        model_name: Model identifier.
        target_dataset: Cross-dataset target name.
        output_dir: Where to save analysis.

    Returns:
        Dict with novel class analysis results, or None if no novel samples.
    """
    if len(X_novel) == 0 or len(y_novel) == 0:
        logger.info(f"No novel classes in {target_dataset}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Analyzing {len(y_novel)} novel-class samples from {target_dataset}: "
        f"{y_novel.value_counts().to_dict()}"
    )

    # Predictions on novel samples
    y_pred = model.predict(X_novel)

    try:
        y_proba = model.predict_proba(X_novel)
        max_confidence = np.max(y_proba, axis=1)
        mean_confidence = float(np.mean(max_confidence))
    except Exception:
        max_confidence = None
        mean_confidence = None

    # Which known classes absorb novel attacks?
    novel_families = y_novel.unique()
    absorption = {}
    for family in novel_families:
        family_mask = y_novel == family
        family_preds = y_pred[family_mask.values]
        unique_preds, counts = np.unique(family_preds, return_counts=True)
        total = int(counts.sum())
        absorption[family] = {
            int(pred): int(count)
            for pred, count in zip(unique_preds, counts)
        }
        # Log the dominant prediction class
        dominant_idx = np.argmax(counts)
        logger.info(
            f"  {family} ({total} samples) → most often predicted as "
            f"class {int(unique_preds[dominant_idx])} "
            f"({int(counts[dominant_idx])}/{total} = "
            f"{counts[dominant_idx]/total:.1%})"
        )

    result = {
        "target_dataset": target_dataset,
        "model": model_name,
        "n_novel_samples": int(len(y_novel)),
        "novel_families": list(novel_families),
        "absorption_map": absorption,
        "mean_prediction_confidence": mean_confidence,
    }

    # Save analysis
    analysis_path = os.path.join(
        output_dir,
        f"{model_name}_{target_dataset}_novel_analysis.json",
    )
    with open(analysis_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Novel class analysis saved to {analysis_path}")

    return result


# ---------------------------------------------------------------------------
# Main pipeline: full cross-dataset evaluation for one model + target
# ---------------------------------------------------------------------------

def run_cross_dataset_evaluation(
    model: Any,
    model_name: str,
    target_dataset: str,
    task: int,
    primary_cache_dir: str,
    output_dir: str,
    data_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete cross-dataset generalization evaluation.

    This is the main entry point for evaluating one trained model on one
    cross-dataset target. It handles:
        1. Loading the cross-dataset target (full_test mode)
        2. Applying primary dataset preprocessing (scaling, feature selection)
        3. Filtering to shared labels (task 6)
        4. Zero-shot evaluation
        5. Novel class analysis (task 6)
        6. Saving all results

    Args:
        model: Trained BaseModel instance (loaded from CICIoMT2024 checkpoint).
        model_name: Model identifier (e.g., "XGBoost").
        target_dataset: Cross-dataset target name.
        task: Classification task (2 or 6).
        primary_cache_dir: Path to preprocessing/cache (for primary dataset
            scaler and feature selection info).
        output_dir: Root output directory for cross-dataset results.
        data_root: Optional path to data/ directory.

    Returns:
        Dict with all results including metrics, novel analysis, and paths.
    """
    run_dir = os.path.join(
        output_dir, "cross_dataset", target_dataset, f"task_{task}", model_name
    )
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Cross-Dataset Evaluation: {model_name}")
    logger.info(f"  Source: CICIoMT2024 (task={task})")
    logger.info(f"  Target: {target_dataset}")
    logger.info("=" * 60)

    t0 = time.time()

    # --- Step 1: Load target dataset ---
    X_target, y_target, target_metadata = load_cross_dataset_target(
        dataset=target_dataset,
        task=task,
        data_root=data_root,
    )

    # --- Step 2: Apply primary preprocessing and filter labels ---
    X_processed, y_encoded, shared_class_names, filter_info = (
        preprocess_cross_dataset(
            X_target=X_target,
            y_target=y_target,
            primary_metadata=target_metadata,
            primary_cache_dir=primary_cache_dir,
            primary_task=task,
            target_dataset=target_dataset,
        )
    )

    # --- Step 3: Zero-shot evaluation on shared classes ---
    # For task 6, pass the truly shared class names so macro F1 can be
    # computed over shared classes only (excluding absent families whose
    # F1 is trivially 0, which would deflate the macro average).
    truly_shared = sorted(filter_info.get("shared_families", []))
    eval_result = evaluate_zero_shot(
        model=model,
        X_test=X_processed,
        y_test=y_encoded,
        class_names=shared_class_names,
        model_name=model_name,
        target_dataset=target_dataset,
        task=task,
        output_dir=run_dir,
        shared_class_names=truly_shared if task == 6 else None,
    )

    # --- Step 4: Novel class analysis (task 6 only) ---
    novel_result = None
    if task == 6 and filter_info.get("novel_samples", 0) > 0:
        novel_result = analyze_novel_classes(
            model=model,
            X_novel=filter_info["X_novel"],
            y_novel=filter_info["y_novel"],
            model_name=model_name,
            target_dataset=target_dataset,
            output_dir=run_dir,
        )

    total_time = time.time() - t0

    # --- Combine results ---
    full_result = {
        **eval_result,
        "novel_class_analysis": novel_result,
        "filter_info": {
            "total_samples": filter_info["total_samples"],
            "shared_samples": filter_info["shared_samples"],
            "novel_samples": filter_info["novel_samples"],
            "shared_families": filter_info["shared_families"],
            "novel_families": filter_info["novel_families"],
        },
        "total_time_seconds": total_time,
    }

    # Append to cross-dataset master CSV
    _append_to_cross_dataset_csv(
        os.path.join(output_dir, "cross_dataset", "cross_dataset_results.csv"),
        eval_result,
    )

    f1_display = eval_result["test_metrics"].get(
        "f1_macro_shared", eval_result["test_metrics"]["f1_macro"]
    )
    logger.info(
        f"Cross-dataset evaluation complete in {total_time:.1f}s: "
        f"{model_name} → {target_dataset} task={task} "
        f"F1(macro)={f1_display:.4f}"
    )

    return full_result


def _append_to_cross_dataset_csv(
    csv_path: str,
    result: Dict[str, Any],
) -> None:
    """Append a single row to the cross-dataset master CSV."""
    import csv

    test = result["test_metrics"]
    row = {
        "model": result["model"],
        "source_dataset": result["source_dataset"],
        "target_dataset": result["target_dataset"],
        "task": result["task"],
        "transfer_type": result["transfer_type"],
        "n_test_samples": result["n_test_samples"],
        "n_features": result["n_features"],
        # Macro metrics (primary for cross-dataset)
        "f1_macro": test["f1_macro"],
        "f1_macro_shared": test.get("f1_macro_shared", test["f1_macro"]),
        "precision_macro": test["precision_macro"],
        "recall_macro": test["recall_macro"],
        # Weighted metrics (secondary)
        "f1_weighted": test["f1_weighted"],
        "precision_weighted": test["precision_weighted"],
        "recall_weighted": test["recall_weighted"],
        # Other metrics
        "accuracy": test["accuracy"],
        "mcc": test["mcc"],
        # Timing
        "inference_time_seconds": result.get("inference_time_seconds", ""),
    }

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Appended cross-dataset row to {csv_path}")


# ---------------------------------------------------------------------------
# Batch runner: evaluate one model across ALL cross-dataset targets
# ---------------------------------------------------------------------------

def run_all_cross_dataset(
    model: Any,
    model_name: str,
    task: int,
    primary_cache_dir: str,
    output_dir: str,
    data_root: Optional[str] = None,
    target_datasets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate one model on all cross-dataset targets for a given task.

    Convenience wrapper that calls run_cross_dataset_evaluation for each
    target dataset.

    Args:
        model: Trained BaseModel instance.
        model_name: Model identifier.
        task: Classification task (2 or 6).
        primary_cache_dir: Preprocessing cache directory.
        output_dir: Root output directory.
        data_root: Optional data directory path.
        target_datasets: List of target names. Defaults to all three.

    Returns:
        Dict mapping target_dataset → evaluation result dict.
    """
    if target_datasets is None:
        target_datasets = ["CIC-BoT-IoT", "CIC-IoT-DIAD-2024", "CIC-ToN-IoT"]

    all_results = {}
    for dataset in target_datasets:
        try:
            result = run_cross_dataset_evaluation(
                model=model,
                model_name=model_name,
                target_dataset=dataset,
                task=task,
                primary_cache_dir=primary_cache_dir,
                output_dir=output_dir,
                data_root=data_root,
            )
            all_results[dataset] = result
        except Exception as e:
            logger.error(
                f"Cross-dataset evaluation failed for {model_name} → {dataset}: {e}",
                exc_info=True,
            )
            all_results[dataset] = {"error": str(e)}

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info(f"Cross-Dataset Summary: {model_name} (task={task})")
    logger.info("-" * 70)
    logger.info(f"{'Target':<25} {'F1(macro)':>10} {'F1(weighted)':>12} {'MCC':>8} {'Acc':>8}")
    logger.info("-" * 70)
    for ds, res in all_results.items():
        if "error" in res:
            logger.info(f"{ds:<25} {'ERROR':>10}")
        else:
            m = res["test_metrics"]
            logger.info(
                f"{ds:<25} {m['f1_macro']:>10.4f} {m['f1_weighted']:>12.4f} "
                f"{m['mcc']:>8.4f} {m['accuracy']:>8.4f}"
            )
    logger.info("=" * 70)

    return all_results
