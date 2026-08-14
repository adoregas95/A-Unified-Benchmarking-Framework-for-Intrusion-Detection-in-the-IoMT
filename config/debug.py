"""
Debug mode configuration for the IoMT IDS Framework.

When --debug is passed to any script, these settings override the normal
configuration to enable a fast end-to-end pipeline run with tiny data slices.
The purpose is to validate that the entire pipeline (preprocessing → training
→ cross-dataset → explainability → report) works without errors before
committing real GPU-hours to full-scale experiments.

All debug outputs are routed to a separate directory tree under debug/ to
prevent any contamination of real results.

Usage in scripts:
    from config.debug import apply_debug_overrides, DEBUG_OUTPUT

    if args.debug:
        config = apply_debug_overrides(config)
        output_dir = DEBUG_OUTPUT["results_dir"]
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug output directory tree (mirrors the real output structure)
# ---------------------------------------------------------------------------
DEBUG_ROOT = "debug"

DEBUG_OUTPUT = {
    "results_dir": os.path.join(DEBUG_ROOT, "results"),
    "logs_dir": os.path.join(DEBUG_ROOT, "logs"),
    "plots_dir": os.path.join(DEBUG_ROOT, "results", "plots"),
    "checkpoints_dir": os.path.join(DEBUG_ROOT, "logs", "checkpoints"),
    "cache_dir": os.path.join(DEBUG_ROOT, "preprocessing", "cache"),
    "storage_dir": os.path.join(DEBUG_ROOT, "logs", "optuna_dbs"),
}

# ---------------------------------------------------------------------------
# Debug data settings
# ---------------------------------------------------------------------------
# Number of samples to keep PER CLASS after loading raw data.
# This keeps the data balanced and small enough for fast iteration.
# 200 per class × 19 classes = 3,800 max rows for Task 19.
DEBUG_SAMPLES_PER_CLASS = 200

# For SMOTEENN/SMOTETomek: reduce k_neighbors to avoid errors with tiny classes
# (SMOTE's default k=5 requires at least 6 samples per minority class)
DEBUG_SMOTE_K_NEIGHBORS = 3

# ---------------------------------------------------------------------------
# Debug training settings
# ---------------------------------------------------------------------------
DEBUG_HPO_TRIALS = 2          # 2 Optuna trials instead of 50-100
DEBUG_MAX_EPOCHS = 3          # 3 epochs for DL/Transformer instead of 50
DEBUG_EARLY_STOPPING_PATIENCE = 2
DEBUG_BATCH_SIZES = [64]      # Single small batch size (no search)

# ---------------------------------------------------------------------------
# Debug explainability settings
# ---------------------------------------------------------------------------
DEBUG_SHAP_MAX_SAMPLES = 50       # Instead of 500/2000
DEBUG_KERNEL_BACKGROUND = 20      # Instead of 100
DEBUG_LIME_NUM_SAMPLES = 100      # Instead of 5000
DEBUG_LIME_NUM_INSTANCES = 5      # Instead of 20
DEBUG_STABILITY_BOOTSTRAPS = 2    # Instead of 5

# ---------------------------------------------------------------------------
# Debug cross-dataset settings
# ---------------------------------------------------------------------------
DEBUG_CROSS_DATASET_MAX_SAMPLES = 500  # Cap target dataset size


def apply_debug_overrides(config: dict) -> dict:
    """Apply debug overrides to the project config dict.

    Modifies the config IN PLACE and returns it. This keeps the same
    config structure so all downstream code works unchanged — only the
    values are smaller/faster.

    Args:
        config: Parsed config.yaml dict.

    Returns:
        The same dict with debug overrides applied.
    """
    logger.warning("=" * 60)
    logger.warning("  DEBUG MODE ACTIVE — using tiny data slices")
    logger.warning("  All outputs go to: %s/", DEBUG_ROOT)
    logger.warning("=" * 60)

    # --- Output paths ---
    config["output"] = {
        "results_dir": DEBUG_OUTPUT["results_dir"],
        "logs_dir": DEBUG_OUTPUT["logs_dir"],
        "plots_dir": DEBUG_OUTPUT["plots_dir"],
        "checkpoints_dir": DEBUG_OUTPUT["checkpoints_dir"],
    }

    # --- Preprocessing ---
    config["preprocessing"]["cache_dir"] = DEBUG_OUTPUT["cache_dir"]
    # Imbalance handling: keep enabled but with smaller k_neighbors
    # (The actual subsampling happens at data loading time, not here)

    # --- HPO ---
    config["hpo"]["storage_dir"] = DEBUG_OUTPUT["storage_dir"]
    config["hpo"]["budgets"] = {
        "tree_based": DEBUG_HPO_TRIALS,
        "deep_learning": DEBUG_HPO_TRIALS,
        "transformers": DEBUG_HPO_TRIALS,
    }

    # --- Training ---
    config["training"]["max_epochs"] = {
        "deep_learning": DEBUG_MAX_EPOCHS,
        "transformers": DEBUG_MAX_EPOCHS,
    }
    config["training"]["early_stopping"]["patience"] = DEBUG_EARLY_STOPPING_PATIENCE
    config["training"]["batch_sizes"] = DEBUG_BATCH_SIZES

    return config


def subsample_data(X, y, max_per_class: int = DEBUG_SAMPLES_PER_CLASS,
                   random_state: int = 42):
    """Subsample data to at most max_per_class samples per class.

    Works with both numpy arrays and pandas objects. Returns the same
    types as the inputs.

    Args:
        X: Features (numpy array or DataFrame).
        y: Labels (numpy array or Series).
        max_per_class: Maximum samples per class.
        random_state: Random seed for reproducibility.

    Returns:
        X_sub, y_sub: Subsampled features and labels.
    """
    import numpy as np

    rng = np.random.RandomState(random_state)

    # Convert to numpy for indexing, remember original types
    if hasattr(y, 'values'):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)

    classes = np.unique(y_arr)
    indices = []

    for cls in classes:
        cls_indices = np.where(y_arr == cls)[0]
        if len(cls_indices) > max_per_class:
            chosen = rng.choice(cls_indices, size=max_per_class, replace=False)
        else:
            chosen = cls_indices
        indices.append(chosen)

    indices = np.sort(np.concatenate(indices))

    # Subsample preserving original types
    if hasattr(X, 'iloc'):
        X_sub = X.iloc[indices].reset_index(drop=True)
    else:
        X_sub = X[indices]

    if hasattr(y, 'iloc'):
        y_sub = y.iloc[indices].reset_index(drop=True)
    else:
        y_sub = y[indices]

    n_original = len(y_arr)
    n_sampled = len(indices)
    logger.info(
        f"Debug subsample: {n_original} → {n_sampled} samples "
        f"({len(classes)} classes, max {max_per_class}/class)"
    )

    return X_sub, y_sub


def ensure_debug_dirs():
    """Create the debug output directory tree."""
    for path in DEBUG_OUTPUT.values():
        os.makedirs(path, exist_ok=True)
    logger.info(f"Debug directories created under {DEBUG_ROOT}/")
