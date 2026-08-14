"""
Feature selection using Mutual Information with cumulative threshold.

Design Decision (see docs/DESIGN_DECISIONS.md):
    The proposal originally described a 3-stage pipeline (variance threshold →
    Pearson correlation pruning → Random Forest importance). During implementation,
    we pivoted to MI-based selection with cumulative thresholding because:
      1. MI is model-agnostic — avoids biasing toward tree-friendly features when
         benchmarking across tree-based, DL, and transformer models.
      2. Cumulative MI thresholding is adaptive — it naturally retains more features
         when information is spread across many, and fewer when concentrated.
      3. It eliminates the need for arbitrary fixed k-values, producing a single
         defensible feature set per task ("features covering 90% of total MI").

Usage:
    from preprocessing.feature_selection import select_features_mi_threshold
    indices, names, scores = select_features_mi_threshold(
        X_train, y_train, feature_names, threshold=0.90
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import mutual_info_classif


def compute_mi_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute Mutual Information scores for all features against the target.

    MI measures the statistical dependence between each feature and the class
    label. Higher MI means the feature carries more information about which
    class a sample belongs to. MI is model-agnostic — it makes no assumptions
    about the relationship being linear or tree-structured.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Integer-encoded label vector (n_samples,).
        n_neighbors: Number of neighbors for MI estimation (default 3).
                     Higher values give smoother but slower estimates.
        random_state: Seed for reproducibility (MI estimation uses randomness).

    Returns:
        scores: Array of MI scores, one per feature. Non-negative; zero means
                the feature is statistically independent of the target.
    """
    return mutual_info_classif(
        X, y, n_neighbors=n_neighbors, random_state=random_state
    )


def select_features_mi_threshold(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.90,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> Tuple[List[int], List[str], np.ndarray, Dict]:
    """
    Select features using cumulative MI thresholding.

    How it works:
      1. Compute MI score for each feature against the target variable.
      2. Sort features by MI score in descending order (most informative first).
      3. Compute the cumulative sum of MI scores as a fraction of total MI.
      4. Keep all features needed to reach the specified threshold (e.g., 90%).

    This is adaptive — binary classification may concentrate information in
    fewer features, while 19-class classification may spread it across more.
    The result is a single defensible feature set per task without needing
    to sweep over arbitrary k values.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Integer-encoded label vector (n_samples,).
        feature_names: List of feature names matching X's columns.
        threshold: Fraction of total MI to retain (default 0.90 = 90%).
                   Higher values keep more features (0.95 → conservative pruning),
                   lower values are more aggressive (0.85 → aggressive pruning).
        n_neighbors: MI estimation parameter (see compute_mi_scores).
        random_state: Seed for reproducibility.

    Returns:
        selected_indices: Column indices of retained features (sorted by MI rank).
        selected_names: Names of retained features (sorted by MI rank).
        scores: Raw MI scores for ALL original features (for logging/analysis).
        selection_info: Dictionary with selection metadata:
            - total_mi: Sum of all MI scores.
            - retained_mi: Sum of MI scores for selected features.
            - retained_fraction: Actual fraction retained (>= threshold).
            - n_selected: Number of features selected.
            - n_original: Original number of features.
            - threshold_used: The threshold parameter that was used.
            - cumulative_fractions: Array of cumulative MI fractions (for plots).
    """
    # Step 1: Compute MI scores for all features
    scores = compute_mi_scores(X, y, n_neighbors, random_state)

    # Step 2: Sort features by MI score (highest first)
    ranked_indices = np.argsort(scores)[::-1]
    ranked_scores = scores[ranked_indices]

    # Step 3: Compute cumulative MI as fraction of total
    total_mi = ranked_scores.sum()

    # Edge case: if total MI is zero (e.g., all features are independent
    # of the target), return all features rather than selecting nothing
    if total_mi == 0:
        return (
            list(range(len(feature_names))),
            list(feature_names),
            scores,
            {
                "total_mi": 0.0,
                "retained_mi": 0.0,
                "retained_fraction": 1.0,
                "n_selected": len(feature_names),
                "n_original": len(feature_names),
                "threshold_used": threshold,
                "cumulative_fractions": np.ones(len(feature_names)),
            },
        )

    cumulative_mi = np.cumsum(ranked_scores)
    cumulative_fractions = cumulative_mi / total_mi

    # Step 4: Find how many features are needed to reach the threshold.
    # np.searchsorted finds the first index where cumulative_fractions >= threshold.
    # We add 1 because we want to INCLUDE that feature (it pushes us over).
    n_selected = int(np.searchsorted(cumulative_fractions, threshold) + 1)

    # Ensure at least 1 feature is selected (safety check)
    n_selected = max(1, min(n_selected, len(feature_names)))

    # Step 5: Extract the selected feature indices and names
    selected_indices = ranked_indices[:n_selected].tolist()
    selected_names = [feature_names[i] for i in selected_indices]

    # Build metadata dictionary for logging and reproducibility
    retained_mi = float(ranked_scores[:n_selected].sum())
    selection_info = {
        "total_mi": float(total_mi),
        "retained_mi": retained_mi,
        "retained_fraction": retained_mi / total_mi,
        "n_selected": n_selected,
        "n_original": len(feature_names),
        "threshold_used": threshold,
        "cumulative_fractions": cumulative_fractions.tolist(),
    }

    return selected_indices, selected_names, scores, selection_info


# ---------------------------------------------------------------------------
# Legacy function kept for backward compatibility (e.g., ablation scripts)
# ---------------------------------------------------------------------------
def select_features_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> Tuple[List[int], List[str], np.ndarray]:
    """
    Select top-k features using Mutual Information (legacy interface).

    This function is preserved for backward compatibility with ablation
    scripts that may still sweep over fixed k values. For primary experiments,
    use select_features_mi_threshold() instead.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Integer-encoded label vector (n_samples,).
        feature_names: List of feature names.
        k: Number of features to select.
        n_neighbors: MI estimation parameter.
        random_state: Random state for reproducibility.

    Returns:
        selected_indices: Indices of selected features.
        selected_names: Names of selected features.
        scores: MI scores for all features.
    """
    scores = compute_mi_scores(X, y, n_neighbors, random_state)

    # Rank features by MI score (descending)
    ranked_indices = np.argsort(scores)[::-1]

    # Select top-k (capped at total feature count)
    k = min(k, len(feature_names))
    selected_indices = ranked_indices[:k].tolist()
    selected_names = [feature_names[i] for i in selected_indices]

    return selected_indices, selected_names, scores
