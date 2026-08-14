"""
Preprocessing pipeline for the IoMT IDS Unified Benchmarking Framework.

Handles data cleaning (Inf/NaN), feature scaling, label encoding,
class imbalance treatment, validation splitting, and caching.

This module sits between data_loader.py (raw data) and the model training
pipeline. Feature selection (MI-based) is handled separately in
feature_selection.py and can be applied after preprocessing.

Pipeline flow:
    data_loader.load_dataset()     # raw CSVs → DataFrames (76 features)
    → clean_data()                 # replace Inf/NaN with training medians
    → scale_data()                 # normalize features (MinMax/Standard/Robust)
    → encode_labels()              # string labels → integer indices
    → select_features_mi_threshold()  # cumulative MI thresholding (optional)
    → handle_imbalance()           # SMOTE on training data only (optional)
    → save_preprocessed_data()     # cache to NPZ for fast reload

Usage:
    from preprocessing.preprocessing import preprocess_pipeline

    result = preprocess_pipeline(
        dataset="CICIoMT2024",
        task=2,
        scaling="minmax",
        imbalance_method="smotetomek",
        val_ratio=0.15,
        mi_threshold=0.90,   # retain features covering 90% of total MI
        cache_dir="preprocessing/cache",
    )
    # result dict contains: X_train, y_train, X_val, y_val, X_test, y_test,
    # feature_names, metadata, scaler, label_encoder, preprocessing_config
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,    # z-score: (x - mean) / std → mean=0, std=1
    MinMaxScaler,      # linear: (x - min) / (max - min) → range [0, 1]
    RobustScaler,      # IQR-based: (x - median) / IQR → outlier-resistant
    LabelEncoder,      # string labels → integer indices (e.g., "DDoS" → 1)
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Import our data loader from the same package
from .data_loader import load_dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Clean CICFlowMeter features: replace Inf with NaN, then impute.

    CICFlowMeter produces Inf values in flow rate columns (Flow Bytes/s,
    Flow Packets/s, Fwd Packets/s, Bwd Packets/s) when flow duration is 0
    because it computes bytes/duration and packets/duration. Division by zero
    yields Inf, which breaks all downstream ML operations.

    Strategy:
        1. Replace +/-Inf with NaN (so we can use pandas fillna)
        2. Compute column medians from TRAINING SET ONLY
        3. Fill NaN with those training medians in all splits

    Why training medians only? Using test/val data to compute fill values
    would leak information from the evaluation sets into training, giving
    artificially inflated performance (data leakage).

    Args:
        X_train: Training features (DataFrame with 76 CICFlowMeter columns)
        X_test: Test features
        X_val: Optional validation features

    Returns:
        Cleaned copies of the input DataFrames (same order as args)
    """
    # Work on copies to avoid modifying the caller's DataFrames
    X_train = X_train.copy()
    X_test = X_test.copy()
    if X_val is not None:
        X_val = X_val.copy()

    # Step 1: Convert all Inf/-Inf to NaN so we can handle them uniformly
    # np.inf and -np.inf would cause errors in scaling and model training
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_val is not None:
        X_val.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 2: Compute column-wise medians from training data ONLY
    # Median is preferred over mean because CICFlowMeter features are
    # heavily skewed (e.g., most Flow Bytes/s are small, a few are huge)
    train_medians = X_train.median()

    # Log how many values we're fixing (useful for debugging / reporting)
    train_nan_count = X_train.isna().sum().sum()
    test_nan_count = X_test.isna().sum().sum()
    if train_nan_count > 0 or test_nan_count > 0:
        logger.info(
            f"Cleaning: {train_nan_count} NaN/Inf in train, "
            f"{test_nan_count} in test"
        )

    # Step 3: Replace all NaN values with the training median for that column
    X_train.fillna(train_medians, inplace=True)
    X_test.fillna(train_medians, inplace=True)
    if X_val is not None:
        X_val.fillna(train_medians, inplace=True)

    # Safety net: if an entire column was NaN (all Inf), the median itself
    # is NaN, so fillna above wouldn't fix those. Fill any remaining with 0.
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    if X_val is not None:
        X_val.fillna(0, inplace=True)

    if X_val is not None:
        return X_train, X_test, X_val
    return X_train, X_test


# =============================================================================
# Feature Scaling
# =============================================================================

def scale_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    method: str = "minmax",
) -> Tuple:
    """
    Scale features using the specified method.

    The scaler is fit ONLY on training data to prevent data leakage.
    Test and validation data are transformed using the training statistics.

    Supported methods:
        - "minmax":   Scales each feature to [0, 1] using min and max.
                      Good default for neural networks (bounded input range).
                      Sensitive to outliers (a single extreme value compresses
                      the rest of the data into a narrow band).

        - "standard": Subtracts mean, divides by std (z-score normalization).
                      Centers data at 0 with unit variance.
                      Also sensitive to outliers (outliers inflate std).

        - "robust":   Subtracts median, divides by IQR (interquartile range).
                      Much more resistant to outliers because median and IQR
                      are not affected by extreme values. Particularly well-
                      suited for CICFlowMeter data where flow rate features
                      can have extreme outliers from very short flows.

    Args:
        X_train: Training features as numpy array (n_samples, n_features)
        X_test: Test features as numpy array
        X_val: Optional validation features as numpy array
        method: One of "minmax", "standard", or "robust"

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, [X_val_scaled,] scaler)
        The scaler object is returned so it can be saved for inference time.
    """
    # Select the appropriate scaler based on the method parameter
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        # RobustScaler uses median and IQR (Q1 to Q3) instead of mean/std.
        # This makes it robust to outliers, which are common in network
        # traffic data (e.g., a DDoS burst creating extreme packet counts).
        scaler = RobustScaler()
    else:
        raise ValueError(
            f"Unknown scaling method: '{method}'. "
            f"Use 'minmax', 'standard', or 'robust'."
        )

    # fit_transform: learns statistics from training data AND transforms it
    X_train_scaled = scaler.fit_transform(X_train)

    # transform: applies the SAME learned statistics to test/val data
    # This ensures no information leaks from test/val into preprocessing
    X_test_scaled = scaler.transform(X_test)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler

    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# Label Encoding
# =============================================================================

def encode_labels(
    y_train: pd.Series,
    y_test: pd.Series,
    y_val: Optional[pd.Series] = None,
) -> Tuple:
    """
    Encode string labels to integer indices for model training.

    Example: ["Benign", "DDoS", "DoS"] → [0, 1, 2]

    The encoder is fit on the union of ALL splits to ensure every class
    gets a consistent integer mapping, even if a rare class only appears
    in the test set (e.g., some attack types may appear in test but not
    in the particular training fold).

    Args:
        y_train: Training labels as strings (e.g., "Benign", "DDoS", ...)
        y_test: Test labels as strings
        y_val: Optional validation labels as strings

    Returns:
        Tuple of (y_train_encoded, y_test_encoded, [y_val_encoded,] label_encoder)
        The LabelEncoder is returned for later inverse_transform (int → string).
    """
    le = LabelEncoder()

    # Fit on ALL labels to ensure consistent encoding across splits
    all_labels = pd.concat([y_train, y_test])
    if y_val is not None:
        all_labels = pd.concat([all_labels, y_val])
    le.fit(all_labels)

    # Transform each split using the shared encoding
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    if y_val is not None:
        y_val_enc = le.transform(y_val)
        return y_train_enc, y_test_enc, y_val_enc, le

    return y_train_enc, y_test_enc, le


# =============================================================================
# Class Imbalance Handling
# =============================================================================

def handle_imbalance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "smoteenn",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance in training data using resampling.

    CRITICAL: Applied ONLY to training data. Validation and test sets are
    left untouched so they reflect the real-world class distribution.
    Applying SMOTE to val/test would create artificial samples that don't
    exist in reality, giving misleadingly optimistic evaluation metrics.

    Supported methods:
        - "smoteenn" (default): Two-phase approach:
              Phase 1 (SMOTE): Generates synthetic minority samples by
              interpolating between existing minority-class neighbors.
              Phase 2 (ENN): Edited Nearest Neighbors removes any sample
              whose class label disagrees with a majority of its k nearest
              neighbors. More aggressive boundary cleaning than Tomek links.
              Validated by Riyadi et al. (2025) as the best method on
              CICIoMT2024 (XGB+SMOTEENN: 99.811% F1).

        - "smotetomek": SMOTE + Tomek links cleanup. Less aggressive
              boundary cleaning — only removes directly adjacent
              opposite-class pairs. Computationally similar to SMOTEENN.

        - "smote": SMOTE oversampling only (no boundary cleanup).
              Faster but may leave noisy boundary samples.

        - "none": No resampling. Return training data as-is.
              Appropriate for models with built-in class weighting
              (e.g., XGBoost's scale_pos_weight, CatBoost's auto_class_weights)
              or when we want to evaluate how each model handles imbalance
              natively. Also useful as a baseline comparison.

    Args:
        X_train: Training features (scaled numpy array)
        y_train: Training labels (integer-encoded numpy array)
        method: Resampling strategy ("smoteenn", "smotetomek", "smote", or "none")
        random_state: Random seed for reproducibility

    Returns:
        Resampled (X_train, y_train) — shapes may differ from input
    """
    # "none" is valid — skip resampling entirely
    if method == "none" or method is None:
        return X_train, y_train

    # SMOTE requires at least k_neighbors+1 samples per class to find
    # enough neighbors for interpolation. If a class is very small, we
    # reduce k_neighbors to avoid a ValueError from sklearn.
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    min_class = unique[counts.argmin()]

    if min_count < 6:
        # Reduce k_neighbors so SMOTE can still work with tiny classes
        k_neighbors = max(1, min_count - 1)
        logger.warning(
            f"Class {min_class} has only {min_count} samples. "
            f"Setting SMOTE k_neighbors={k_neighbors} (default is 5)."
        )
    else:
        k_neighbors = 5  # sklearn default

    if method == "smoteenn":
        # SMOTE + Edited Nearest Neighbors: oversampling + aggressive boundary
        # cleaning. ENN removes any sample whose class disagrees with a majority
        # of its k nearest neighbors — more thorough than Tomek links.
        # Validated by Riyadi et al. (2025) as the best-performing method on
        # CICIoMT2024 (XGB+SMOTEENN: 99.811% vs XGB+SMOTETomek: 99.770%).
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        enn = EditedNearestNeighbours(n_neighbors=3)
        resampler = SMOTEENN(smote=smote, enn=enn, random_state=random_state)

    elif method == "smotetomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        resampler = SMOTETomek(smote=smote, random_state=random_state)

    elif method == "smote":
        from imblearn.over_sampling import SMOTE
        resampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

    else:
        raise ValueError(
            f"Unknown imbalance method: '{method}'. "
            f"Use 'smoteenn', 'smotetomek', 'smote', or 'none'."
        )

    logger.info(
        f"Resampling training data ({method}): "
        f"{len(y_train)} samples, {len(unique)} classes"
    )

    # fit_resample: analyzes class distribution, generates synthetic samples
    # (SMOTE), then optionally removes Tomek links (SMOTETomek)
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    logger.info(
        f"After resampling: {len(y_resampled)} samples "
        f"(was {len(y_train)}, delta={len(y_resampled) - len(y_train):+d})"
    )

    return X_resampled, y_resampled


# =============================================================================
# Validation Split
# =============================================================================

def create_validation_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a stratified validation split from the training data.

    This carves out a portion of the training data to serve two purposes:

    1. Hyperparameter Optimization (HPO): Optuna tries many model configs
       and needs to evaluate each one. It trains on the remaining training
       data and measures performance on this validation set. Without it,
       Optuna would have to use the test set, which would leak test info
       into the model selection process.

    2. Early Stopping: Neural networks (CNN1D, BiLSTM) and transformers
       (FTTransformer, SAINT) train for multiple epochs. We monitor
       validation loss/F1 each epoch and stop when it stops improving
       (patience=10 epochs). This prevents overfitting — the model learns
       the training data's noise rather than its patterns.

    "Stratified" means the class proportions in the validation set match
    the original training set. If training has 70% Benign / 30% Attack,
    the validation set will also be ~70/30. This ensures the validation
    score is representative of performance on each class.

    Important: This split happens BEFORE class imbalance handling. The
    validation set keeps the natural (imbalanced) distribution because
    that's what the model will encounter in production.

    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        val_ratio: Fraction of training data to hold out (default 15%)
        random_state: Random seed for reproducible splits

    Returns:
        X_train_split: Reduced training features (85% of original)
        y_train_split: Reduced training labels
        X_val: Validation features (15% of original)
        y_val: Validation labels
    """
    # train_test_split with stratify ensures proportional class representation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,    # 15% goes to validation
        stratify=y_train,       # maintain class proportions
        random_state=random_state,
    )
    return X_tr, y_tr, X_val, y_val


# =============================================================================
# K-Fold Cross-Validation Splits
# =============================================================================

def create_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified k-fold cross-validation splits.

    K-fold cross-validation divides the data into k equal parts (folds).
    For each iteration, one fold is held out as validation while the
    remaining k-1 folds are used for training. This gives k different
    train/val splits, allowing us to estimate model variance.

    "Stratified" means each fold maintains the original class proportions.
    Without stratification, a fold might accidentally contain no samples
    of a rare attack type, giving misleading performance estimates.

    Returns index arrays (not data copies) to save memory — with 6.7M
    rows in CICIoMT2024, copying the data 5 times would be prohibitive.

    Note: Our primary pipeline uses the simple holdout validation split
    (create_validation_split) rather than k-fold, because running HPO
    with k-fold would multiply compute time by k. K-fold is available
    here for final model evaluation if needed.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label vector (n_samples,)
        n_splits: Number of folds (default 5)
        random_state: Random seed for reproducible fold assignments

    Returns:
        List of (train_indices, val_indices) numpy arrays, one per fold.
        Usage: X_train_fold = X[train_idx], X_val_fold = X[val_idx]
    """
    # StratifiedKFold ensures each fold has approximately the same
    # percentage of samples of each target class as the complete set
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,          # shuffle before splitting (important!)
        random_state=random_state,
    )

    # Collect all fold indices as a list of (train, val) tuples
    splits = []
    for train_idx, val_idx in skf.split(X, y):
        splits.append((train_idx, val_idx))
    return splits


# =============================================================================
# Save / Load Preprocessed Data
# =============================================================================

def _make_cache_key(
    dataset: str,
    task: int,
    scaling: str,
    imbalance: str,
    val_ratio: float,
    mi_threshold: Union[float, str],
) -> str:
    """
    Generate a deterministic cache key for a preprocessing configuration.

    Creates an MD5 hash from the config parameters so we can quickly
    check if cached data matches the current settings without comparing
    the actual arrays.
    """
    key_str = (
        f"{dataset}_task{task}_{scaling}_{imbalance}_val{val_ratio}"
        f"_mi{mi_threshold}"
    )
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def save_preprocessed_data(
    output_dir: str,
    dataset_name: str,
    task: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    metadata: Dict[str, Any],
    fold_idx: Optional[int] = None,
) -> str:
    """
    Save preprocessed data splits to disk as compressed NPZ files.

    Why cache preprocessed data?
    - Loading 72 CSVs and running the full pipeline takes minutes.
    - Once preprocessed, we reload in seconds via numpy's binary format.
    - Multiple model training runs can reuse the same preprocessed data.
    - Ensures exact reproducibility (same preprocessed arrays every time).

    Directory structure:
        output_dir/
        └── {dataset_name}/
            └── task_{task}/
                ├── data.npz          ← compressed numpy arrays
                └── metadata.json     ← config + statistics (human-readable)

    If fold_idx is provided (for k-fold experiments):
        output_dir/{dataset_name}/task_{task}/fold_{fold_idx}/

    Args:
        output_dir: Root output directory (e.g., "preprocessing/cache")
        dataset_name: Dataset identifier (e.g., "CICIoMT2024")
        task: Classification task (2, 6, or 19)
        X_train, y_train: Training data (possibly resampled)
        X_val, y_val: Validation data (natural distribution)
        X_test, y_test: Test data (natural distribution)
        feature_names: Column names after feature selection
        metadata: Dict with preprocessing config and dataset stats
        fold_idx: Optional fold number for k-fold experiments

    Returns:
        Path to the directory where files were saved
    """
    # Build the directory path based on dataset and task
    save_dir = os.path.join(output_dir, dataset_name, f"task_{task}")
    if fold_idx is not None:
        save_dir = os.path.join(save_dir, f"fold_{fold_idx}")

    # Create the directory tree if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save all arrays in a single compressed NPZ file
    # NPZ is numpy's native binary format — fast to save/load, and
    # savez_compressed applies zlib compression (typically 3-5x smaller)
    np.savez_compressed(
        os.path.join(save_dir, "data.npz"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    # Save metadata as JSON for human readability and debugging
    # Filter out non-serializable objects (scaler, label_encoder)
    meta_to_save = {
        k: v for k, v in metadata.items()
        if k != "label_encoder" and k != "scaler"
    }
    meta_to_save["feature_names"] = feature_names

    # Extract scaler parameters as plain numbers for cross-dataset reuse.
    # The scaler object itself isn't JSON-serializable, but its fitted
    # statistics are just numpy arrays that we convert to lists.
    scaler = metadata.get("scaler")
    if scaler is not None:
        scaler_params = {}
        if hasattr(scaler, "center_"):
            # RobustScaler: center_ = median, scale_ = IQR
            scaler_params["center_"] = scaler.center_.tolist()
            scaler_params["scale_"] = scaler.scale_.tolist()
        if hasattr(scaler, "data_min_"):
            # MinMaxScaler: data_min_, data_max_, data_range_
            scaler_params["data_min_"] = scaler.data_min_.tolist()
            scaler_params["data_max_"] = scaler.data_max_.tolist()
        if hasattr(scaler, "mean_"):
            # StandardScaler: mean_, scale_ (= std)
            scaler_params["mean_"] = scaler.mean_.tolist()
            scaler_params["scale_"] = scaler.scale_.tolist()
        meta_to_save["scaler_params"] = scaler_params

    # Extract label encoder classes for cross-dataset label mapping
    label_encoder = metadata.get("label_encoder")
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        meta_to_save["label_classes"] = label_encoder.classes_.tolist()

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta_to_save, f, indent=2, default=str)

    logger.info(f"Saved preprocessed data to {save_dir}")
    return save_dir


def load_preprocessed_data(
    data_dir: str,
    dataset_name: str,
    task: int,
    fold_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load previously cached preprocessed data from disk.

    This is the fast path — instead of re-running the full pipeline
    (load CSVs → clean → scale → encode → resample), we load the
    pre-computed arrays directly from NPZ files in seconds.

    Args:
        data_dir: Root preprocessed data directory (e.g., "preprocessing/cache")
        dataset_name: Dataset identifier (e.g., "CICIoMT2024")
        task: Classification task (2, 6, or 19)
        fold_idx: Optional fold index for k-fold experiments

    Returns:
        X_train, y_train: Training data arrays
        X_val, y_val: Validation data arrays
        X_test, y_test: Test data arrays
        metadata: Dict with preprocessing config and dataset stats
    """
    # Reconstruct the path where save_preprocessed_data stored the files
    load_dir = os.path.join(data_dir, dataset_name, f"task_{task}")
    if fold_idx is not None:
        load_dir = os.path.join(load_dir, f"fold_{fold_idx}")

    npz_path = os.path.join(load_dir, "data.npz")
    meta_path = os.path.join(load_dir, "metadata.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"No preprocessed data found at {npz_path}. "
            f"Run preprocess_pipeline() first."
        )

    # np.load reads the compressed NPZ and returns a dict-like object
    data = np.load(npz_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
        metadata,
    )


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================

def preprocess_pipeline(
    dataset: str,
    task: int,
    data_root: Optional[str] = None,
    scaling: str = "minmax",
    imbalance_method: str = "smotetomek",
    val_ratio: float = 0.15,
    mi_threshold: Union[float, str] = 0.90,
    cache_dir: Optional[str] = None,
    random_state: int = 42,
    force_reprocess: bool = False,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete preprocessing pipeline for a dataset + task combination.

    This is the main entry point that orchestrates all preprocessing steps
    in the correct order. It ensures no data leakage by always fitting
    transformations on training data only and applying them to val/test.

    Pipeline steps (in order):
        1. Load raw data → DataFrames with 76 CICFlowMeter features
        2. Validation split → carve out 15% of training for HPO/early stopping
        3. Clean data → replace Inf/NaN with training medians
        4. Scale features → normalize to comparable ranges (fit on train only)
        5. Feature selection → cumulative MI thresholding (retain features
           covering mi_threshold fraction of total MI; "all" skips selection)
        6. Encode labels → string class names to integers
        7. Handle imbalance → SMOTETomek on training data only

    Args:
        dataset: Dataset name ("CICIoMT2024", "CIC-BoT-IoT",
                 "CIC-IoT-DIAD-2024", or "CIC-ToN-IoT")
        task: Classification task (2=binary, 6=families, 19=individual)
        data_root: Path to the data/ directory containing dataset folders
        scaling: Feature scaling method ("minmax", "standard", or "robust")
        imbalance_method: Class balancing ("smotetomek", "smote", or "none")
        val_ratio: Fraction of training data for validation (default 0.15)
        mi_threshold: Fraction of total MI to retain (e.g. 0.90 = 90%).
                      Use "all" to skip feature selection entirely.
                      Adaptive per task — different tasks will retain different
                      numbers of features based on MI distribution.
        cache_dir: Directory to save/load preprocessed NPZ files (None=no cache)
        random_state: Global random seed for all random operations
        force_reprocess: If True, rerun pipeline even if cache exists

    Returns:
        Dict containing:
            X_train (ndarray): Training features, shape (n_train, n_features)
            y_train (ndarray): Training labels, shape (n_train,)
            X_val (ndarray): Validation features (natural distribution)
            y_val (ndarray): Validation labels
            X_test (ndarray): Test features (natural distribution)
            y_test (ndarray): Test labels
            feature_names (list): Feature column names after selection
            metadata (dict): Dataset info (n_classes, class_names, etc.)
            scaler: Fitted scaler object (for inference-time preprocessing)
            label_encoder: Fitted LabelEncoder (for int → string conversion)
            preprocessing_config (dict): All config parameters used
    """
    logger.info(
        f"Preprocessing: {dataset} | task={task} | scaling={scaling} | "
        f"imbalance={imbalance_method} | mi_threshold={mi_threshold}"
    )

    # ── Step 1: Load raw data ──────────────────────────────────────────────
    # data_loader.load_dataset() reads CSVs, standardizes column names,
    # maps labels to the requested task granularity, and returns DataFrames
    logger.info("Step 1/7: Loading raw data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, metadata = load_dataset(
        dataset=dataset, task=task, data_root=data_root
    )
    feature_names = metadata["feature_names"]

    logger.info(
        f"  Loaded: {len(X_train_raw)} train + {len(X_test_raw)} test samples, "
        f"{len(feature_names)} features, {metadata['n_classes']} classes"
    )

    # ── Debug subsample (optional) ────────────────────────────────────────
    # When debug_mode is True, reduce to a tiny slice BEFORE any heavy
    # transformations to enable fast end-to-end pipeline validation.
    if debug_mode:
        from config.debug import subsample_data, DEBUG_SAMPLES_PER_CLASS
        logger.warning(
            f"DEBUG MODE: subsampling to {DEBUG_SAMPLES_PER_CLASS} per class"
        )
        X_train_raw, y_train_raw = subsample_data(
            X_train_raw, y_train_raw,
            max_per_class=DEBUG_SAMPLES_PER_CLASS,
            random_state=random_state,
        )
        X_test_raw, y_test_raw = subsample_data(
            X_test_raw, y_test_raw,
            max_per_class=DEBUG_SAMPLES_PER_CLASS // 2,  # smaller test set
            random_state=random_state,
        )

    # ── Step 2: Create validation split (before any transformations) ───────
    # We split BEFORE cleaning/scaling so that all transformations are
    # fit on the reduced training set (without val), preventing leakage
    logger.info(f"Step 2/7: Creating validation split ({val_ratio:.0%})...")
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=val_ratio,      # 15% → validation
        stratify=y_train_raw,     # preserve class proportions
        random_state=random_state,
    )

    logger.info(
        f"  Split: {len(X_tr_df)} train, {len(X_val_df)} val, "
        f"{len(X_test_raw)} test"
    )

    # ── Step 3: Clean data (Inf/NaN) ──────────────────────────────────────
    # CICFlowMeter's flow rate columns produce Inf when flow duration = 0
    logger.info("Step 3/7: Cleaning data (Inf/NaN handling)...")
    X_tr_clean, X_test_clean, X_val_clean = clean_data(
        X_tr_df, X_test_raw, X_val_df
    )

    # ── Step 4: Scale features ────────────────────────────────────────────
    # Scaler is fit on training data only, then applied to val/test
    logger.info(f"Step 4/7: Scaling features ({scaling})...")
    X_tr_scaled, X_test_scaled, X_val_scaled, scaler = scale_data(
        X_tr_clean.values,        # .values converts DataFrame → numpy array
        X_test_clean.values,
        X_val_clean.values,
        method=scaling,
    )

    # ── Step 5: Feature selection (cumulative MI thresholding, optional) ──
    # When mi_threshold is a float (e.g., 0.90), retain features whose
    # cumulative MI accounts for that fraction of total MI. This is adaptive:
    # binary classification may concentrate MI in fewer features than
    # 19-class, so the threshold naturally selects different counts per task.
    # When "all", skip selection and use all 76 features.
    if mi_threshold != "all" and isinstance(mi_threshold, (int, float)):
        logger.info(
            f"Step 5/7: Selecting features by cumulative MI "
            f"(threshold={mi_threshold:.0%})..."
        )
        from .feature_selection import select_features_mi_threshold

        # MI computation needs integer labels, so we temporarily encode them
        le_temp = LabelEncoder()
        y_tr_temp = le_temp.fit_transform(y_tr)

        # Compute MI scores on training data and select by cumulative threshold
        selected_indices, selected_names, mi_scores, selection_info = (
            select_features_mi_threshold(
                X_tr_scaled, y_tr_temp, feature_names,
                threshold=float(mi_threshold),
                random_state=random_state,
            )
        )

        # Apply the same feature selection to ALL splits (train, val, test)
        # Only the indices selected from training MI scores are used —
        # this prevents data leakage from val/test MI distributions.
        X_tr_scaled = X_tr_scaled[:, selected_indices]
        X_val_scaled = X_val_scaled[:, selected_indices]
        X_test_scaled = X_test_scaled[:, selected_indices]
        feature_names = selected_names

        logger.info(
            f"  Selected {selection_info['n_selected']} / "
            f"{selection_info['n_original']} features "
            f"(covering {selection_info['retained_fraction']:.1%} of total MI)"
        )
    else:
        logger.info("Step 5/7: Using all features (no selection)")
        mi_scores = None
        selection_info = None

    # ── Step 6: Encode labels ─────────────────────────────────────────────
    # Convert string class names to integers for model training
    logger.info("Step 6/7: Encoding labels...")
    y_tr_enc, y_test_enc, y_val_enc, label_encoder = encode_labels(
        y_tr, y_test_raw, y_val
    )

    logger.info(
        f"  Classes: {list(label_encoder.classes_)} "
        f"→ {list(range(len(label_encoder.classes_)))}"
    )

    # ── Step 7: Handle class imbalance (training data only) ───────────────
    # SMOTE/SMOTETomek is applied ONLY to training data. Val/test keep
    # their natural distribution to give honest evaluation metrics.
    logger.info(f"Step 7/7: Handling class imbalance ({imbalance_method})...")
    X_tr_final, y_tr_final = handle_imbalance(
        X_tr_scaled, y_tr_enc,
        method=imbalance_method,
        random_state=random_state,
    )

    # ── Cache results (optional) ──────────────────────────────────────────
    # Save to disk so subsequent runs can reload in seconds instead of
    # re-running the full pipeline
    if cache_dir is not None:
        save_preprocessed_data(
            output_dir=cache_dir,
            dataset_name=dataset,
            task=task,
            X_train=X_tr_final,
            y_train=y_tr_final,
            X_val=X_val_scaled,
            y_val=y_val_enc,
            X_test=X_test_scaled,
            y_test=y_test_enc,
            feature_names=feature_names,
            metadata={
                **metadata,  # spread original metadata (dataset, n_classes, etc.)
                "scaling": scaling,
                "imbalance_method": imbalance_method,
                "val_ratio": val_ratio,
                "mi_threshold": mi_threshold,
                "n_features_selected": len(feature_names),
                "selection_info": selection_info,
                "train_samples_after_resampling": len(y_tr_final),
                "scaler": scaler,              # needed for scaler_params extraction
                "label_encoder": label_encoder, # needed for label_classes extraction
            },
        )

    # ── Build result dict ─────────────────────────────────────────────────
    result = {
        "X_train": X_tr_final,       # resampled training features
        "y_train": y_tr_final,       # resampled training labels (integers)
        "X_val": X_val_scaled,       # validation features (natural distribution)
        "y_val": y_val_enc,          # validation labels (integers)
        "X_test": X_test_scaled,     # test features (natural distribution)
        "y_test": y_test_enc,        # test labels (integers)
        "feature_names": feature_names,  # column names after selection
        "metadata": metadata,        # dataset info (n_classes, class_names, etc.)
        "scaler": scaler,            # fitted scaler for inference time
        "label_encoder": label_encoder,  # for converting int predictions → strings
        "preprocessing_config": {    # record exactly what was done
            "dataset": dataset,
            "task": task,
            "scaling": scaling,
            "imbalance_method": imbalance_method,
            "val_ratio": val_ratio,
            "mi_threshold": mi_threshold,
            "random_state": random_state,
        },
    }

    logger.info(
        f"Preprocessing complete: "
        f"X_train={X_tr_final.shape}, X_val={X_val_scaled.shape}, "
        f"X_test={X_test_scaled.shape}"
    )

    return result
