"""
Preprocessing module for the IoMT IDS Unified Benchmarking Framework.

Main components:
    - data_loader: Raw dataset loading with feature alignment
    - preprocessing: Cleaning, scaling, imbalance handling, caching
    - feature_selection: MI-based feature selection
"""

from .data_loader import load_dataset
from .preprocessing import (
    preprocess_pipeline,
    clean_data,
    scale_data,
    encode_labels,
    handle_imbalance,
    create_validation_split,
    create_kfold_splits,
    save_preprocessed_data,
    load_preprocessed_data,
)
from .feature_selection import (
    select_features_mi_threshold,
    select_features_mi,       # legacy interface for ablation scripts
    compute_mi_scores,
)

__all__ = [
    # Data loading
    "load_dataset",
    # Full pipeline
    "preprocess_pipeline",
    # Individual steps
    "clean_data",
    "scale_data",
    "encode_labels",
    "handle_imbalance",
    "create_validation_split",
    "create_kfold_splits",
    "save_preprocessed_data",
    "load_preprocessed_data",
    # Feature selection
    "select_features_mi_threshold",
    "select_features_mi",
    "compute_mi_scores",
]
