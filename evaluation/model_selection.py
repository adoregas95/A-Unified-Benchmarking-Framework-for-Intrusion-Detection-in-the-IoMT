"""
Multi-criteria model selection for the IoMT IDS Benchmarking Framework.

Implements the composite scoring formula for selecting the best overall model:

    Final Score = W_perf * performance + W_eff * efficiency + W_xai * explainability

where:
    - W_perf = 0.60  (composite of F1-weighted, recall-weighted, and MCC)
    - W_eff  = 0.25  (normalized composite of inference-time efficiency)
    - W_xai  = 0.15  (normalized composite of faithfulness + stability)

Design rationale:
    Detection performance is paramount for a security system (60%). Efficiency
    determines deployability on resource-constrained IoMT gateways (25%).
    Explainability builds operator trust and supports forensic analysis (15%).

The performance composite uses three sub-metrics, each capturing a distinct
dimension of classification quality for intrusion detection:
    - F1-weighted: overall classification quality, balanced by class support.
    - Recall-weighted: attack detection rate — how many threats are caught.
    - MCC (Matthews Correlation Coefficient): robust to class imbalance,
      accounts for all four confusion matrix quadrants.
Each sub-metric is min-max normalized and averaged, mirroring the efficiency
composite approach.

The efficiency composite uses ONLY inference-time metrics (not training time)
because IoMT deployment cares about per-flow detection cost, not one-time
training cost. Each sub-metric is min-max normalized across all models so
that the most efficient model gets 1.0 and the least efficient gets 0.0.
Each model's composite averages only its *available* sub-metrics, because
energy consumption is only measurable for PyTorch-based models (DL and
transformers) via GPU power monitoring. Tree-based models average over
three sub-metrics (latency, memory, parameters); neural models over four
(adding energy).

The explainability composite averages faithfulness (do explanations reflect
actual model behavior?) and stability (are explanations consistent across
runs?), both normalized to [0, 1].
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default weights — can be overridden from config.yaml
DEFAULT_WEIGHTS = {
    "performance": 0.60,
    "efficiency": 0.25,
    "explainability": 0.15,
}

# Performance sub-metrics used in the composite.
# All are "higher is better" — no inversion needed.
PERFORMANCE_METRICS = [
    "f1_weighted",
    "recall_weighted",
    "mcc",
]

# Efficiency sub-metrics used in the composite.
# All are "lower is better" — they get inverted during normalization.
EFFICIENCY_METRICS = [
    "inference_latency_ms_per_sample",
    "peak_memory_mb_inference",
    "model_parameter_count",
    "energy_joules_per_sample",
]


def _normalize_min_max(
    values: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    """Min-max normalize an array to [0, 1].

    Args:
        values: Raw metric values (one per model).
        invert: If True, lower raw values produce higher normalized scores
                (used for "lower is better" metrics like latency).

    Returns:
        Normalized array in [0, 1]. If all values are identical, returns
        array of 1.0 (all models tied).
    """
    arr = np.array(values, dtype=float)

    # Filter out sentinel values (-1.0 means "unavailable")
    valid_mask = arr >= 0
    if not valid_mask.any():
        # All unavailable — return zeros (this metric contributes nothing)
        return np.zeros_like(arr)

    valid_vals = arr[valid_mask]
    vmin, vmax = valid_vals.min(), valid_vals.max()

    if vmax - vmin < 1e-12:
        # All valid values are identical — perfect tie
        result = np.where(valid_mask, 1.0, 0.0)
    else:
        normalized = (arr - vmin) / (vmax - vmin)
        if invert:
            normalized = 1.0 - normalized
        # Unavailable metrics get 0.0 (worst score)
        result = np.where(valid_mask, normalized, 0.0)

    return result


def compute_performance_composite(
    results_df: pd.DataFrame,
) -> np.ndarray:
    """Compute normalized performance composite score for each model.

    For each performance sub-metric (F1-weighted, recall-weighted, MCC),
    models are min-max normalized (higher is better). The composite is the
    mean of available sub-metric scores.

    Args:
        results_df: DataFrame with one row per model and columns matching
                    PERFORMANCE_METRICS. Missing columns are skipped.

    Returns:
        Array of composite performance scores in [0, 1], one per model.
    """
    n_models = len(results_df)
    sub_scores = []

    for metric in PERFORMANCE_METRICS:
        if metric not in results_df.columns:
            logger.debug(f"Performance metric '{metric}' not in results, skipping")
            continue

        values = results_df[metric].values
        normalized = _normalize_min_max(values, invert=False)
        sub_scores.append(normalized)

    if not sub_scores:
        logger.warning("No performance metrics available — returning zeros")
        return np.zeros(n_models)

    # Mean of available sub-metric scores
    return np.mean(sub_scores, axis=0)


def compute_efficiency_composite(
    results_df: pd.DataFrame,
) -> np.ndarray:
    """Compute normalized efficiency composite score for each model.

    For each efficiency sub-metric, models are min-max normalized with
    inversion (lower latency/memory/params/energy = higher score). The
    composite is the per-model mean of that model's *available* sub-metric
    scores.

    This per-model averaging is critical because some metrics may be
    unavailable for certain model families. For example, energy consumption
    is measured via PyTorch's GPU power monitoring and is therefore only
    available for deep-learning and transformer models, not tree-based ones.
    A naive global average would penalize tree-based models with 0.0 for
    energy, unfairly reducing their efficiency score. Instead, each model's
    composite averages only the sub-metrics it actually has.

    Args:
        results_df: DataFrame with one row per model and columns matching
                    EFFICIENCY_METRICS. Missing columns are skipped.

    Returns:
        Array of composite efficiency scores in [0, 1], one per model.
    """
    n_models = len(results_df)
    norm_list = []   # normalized scores per metric
    valid_list = []  # boolean masks per metric

    for metric in EFFICIENCY_METRICS:
        if metric not in results_df.columns:
            logger.debug(f"Efficiency metric '{metric}' not in results, skipping")
            continue

        values = results_df[metric].values
        # Skip metrics where all values are unavailable
        if (values < 0).all():
            logger.debug(f"All values for '{metric}' are unavailable, skipping")
            continue

        valid_mask = values >= 0
        normalized = _normalize_min_max(values, invert=True)
        norm_list.append(normalized)
        valid_list.append(valid_mask)

    if not norm_list:
        logger.warning("No efficiency metrics available — returning zeros")
        return np.zeros(n_models)

    # Stack into matrices: shape (n_metrics, n_models)
    norm_matrix = np.array(norm_list)
    valid_matrix = np.array(valid_list)

    # Per-model mean over that model's valid metrics only
    valid_counts = valid_matrix.sum(axis=0).astype(float)
    valid_counts = np.maximum(valid_counts, 1)  # avoid division by zero
    masked_norms = np.where(valid_matrix, norm_matrix, 0.0)
    composite = masked_norms.sum(axis=0) / valid_counts

    return composite


def compute_explainability_composite(
    faithfulness_scores: np.ndarray,
    stability_scores: np.ndarray,
) -> np.ndarray:
    """Compute normalized explainability composite score for each model.

    Averages faithfulness and stability, both normalized to [0, 1].

    Faithfulness: correlation between feature importance rank and prediction
    degradation when features are removed. Higher = explanations are more
    faithful to actual model behavior.

    Stability: Spearman rank correlation of feature importance rankings
    across different random seeds or data subsets. Higher = more consistent
    explanations.

    Args:
        faithfulness_scores: Raw faithfulness scores, one per model.
        stability_scores: Raw stability scores, one per model.

    Returns:
        Array of composite explainability scores in [0, 1].
    """
    faith_norm = _normalize_min_max(faithfulness_scores, invert=False)
    stab_norm = _normalize_min_max(stability_scores, invert=False)
    return (faith_norm + stab_norm) / 2.0


def compute_final_scores(
    results_df: pd.DataFrame,
    faithfulness_scores: Optional[np.ndarray] = None,
    stability_scores: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute the multi-criteria final score for each model.

    Final Score = W_perf * perf_composite + W_eff * eff_composite + W_xai * xai_composite

    The performance composite averages normalized F1-weighted, recall-weighted,
    and MCC scores (see PERFORMANCE_METRICS). This mirrors the efficiency
    composite approach.

    Args:
        results_df: DataFrame with one row per model. Must contain:
            - Columns from PERFORMANCE_METRICS (f1_weighted, recall_weighted, mcc)
            - Efficiency columns from EFFICIENCY_METRICS (as available)
            - A "model" column for identification
        faithfulness_scores: Faithfulness scores per model. If None,
            explainability weight is redistributed to performance.
        stability_scores: Stability scores per model. If None,
            explainability weight is redistributed to performance.
        weights: Dict with keys "performance", "efficiency", "explainability".
            Defaults to DEFAULT_WEIGHTS (0.60, 0.25, 0.15).

    Returns:
        Copy of results_df with added columns: 'perf_score', 'eff_score',
        'xai_score', 'final_score', 'rank'.
    """
    weights = weights or DEFAULT_WEIGHTS.copy()
    df = results_df.copy()
    n_models = len(df)

    # --- Performance composite (F1-weighted + recall-weighted + MCC) ---
    df["perf_score"] = compute_performance_composite(df)

    # --- Efficiency composite ---
    df["eff_score"] = compute_efficiency_composite(df)

    # --- Explainability composite ---
    if faithfulness_scores is not None and stability_scores is not None:
        df["xai_score"] = compute_explainability_composite(
            faithfulness_scores, stability_scores,
        )
    else:
        # Explainability not yet computed — redistribute weight to performance
        logger.info(
            "Explainability scores not available. Redistributing %.0f%% "
            "weight to performance.",
            weights["explainability"] * 100,
        )
        weights["performance"] += weights["explainability"]
        weights["explainability"] = 0.0
        df["xai_score"] = 0.0

    # --- Final composite score ---
    df["final_score"] = (
        weights["performance"] * df["perf_score"]
        + weights["efficiency"] * df["eff_score"]
        + weights["explainability"] * df["xai_score"]
    )

    # --- Rank (1 = best) ---
    df["rank"] = df["final_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank")

    logger.info(
        "Model ranking (weights: perf=%.0f%%, eff=%.0f%%, xai=%.0f%%):",
        weights["performance"] * 100,
        weights["efficiency"] * 100,
        weights["explainability"] * 100,
    )
    for _, row in df.iterrows():
        logger.info(
            "  #%d %s — final=%.4f (perf=%.4f, eff=%.4f, xai=%.4f)",
            row["rank"],
            row.get("model", "?"),
            row["final_score"],
            row["perf_score"],
            row["eff_score"],
            row["xai_score"],
        )

    return df


def select_best_per_family(
    ranked_df: pd.DataFrame,
    family_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Select the best model from each family for Stage 2 explainability.

    Args:
        ranked_df: Output of compute_final_scores() with 'model' and
                   'final_score' columns.
        family_map: Dict mapping model name to family. Defaults to the
                    standard 3-family grouping.

    Returns:
        Dict mapping family name to best model name.
    """
    if family_map is None:
        family_map = {
            "RandomForest": "tree_based",
            "XGBoost": "tree_based",
            "LightGBM": "tree_based",
            "CatBoost": "tree_based",
            "CNN1D": "deep_learning",
            "BiLSTM": "deep_learning",
            "FTTransformer": "transformers",
            "SAINT": "transformers",
        }

    df = ranked_df.copy()
    df["family"] = df["model"].map(family_map)

    best = {}
    for family, group in df.groupby("family"):
        top = group.sort_values("final_score", ascending=False).iloc[0]
        best[family] = top["model"]
        logger.info(f"Best {family}: {top['model']} (score={top['final_score']:.4f})")

    return best


# ---------------------------------------------------------------------------
# Sensitivity analysis — vary weights by ±10% (committee recommendation)
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    results_df: pd.DataFrame,
    faithfulness_scores: Optional[np.ndarray] = None,
    stability_scores: Optional[np.ndarray] = None,
    base_weights: Optional[Dict[str, float]] = None,
    delta: float = 0.10,
) -> pd.DataFrame:
    """Vary MCDM weights by ±delta and check ranking stability.

    For each of the three pillars, we increase its weight by delta and
    decrease the other two proportionally (so weights still sum to 1.0),
    then decrease it by delta and increase the others. This produces
    6 perturbed weight vectors plus the baseline = 7 scenarios.

    Args:
        results_df: DataFrame with one row per model (same as compute_final_scores).
        faithfulness_scores: Faithfulness scores per model (or None).
        stability_scores: Stability scores per model (or None).
        base_weights: Baseline weights dict. Defaults to DEFAULT_WEIGHTS.
        delta: Perturbation magnitude (default 0.10 = ±10%).

    Returns:
        DataFrame with columns: scenario, w_perf, w_eff, w_xai,
        top_model, top_score, ranking_matches_baseline (bool).
    """
    base_weights = base_weights or DEFAULT_WEIGHTS.copy()
    pillars = ["performance", "efficiency", "explainability"]

    # Compute baseline ranking
    baseline_df = compute_final_scores(
        results_df,
        faithfulness_scores, stability_scores,
        weights=base_weights.copy(),
    )
    baseline_ranking = baseline_df.sort_values("rank")["model"].tolist()
    baseline_top = baseline_ranking[0]

    scenarios = []

    # Baseline scenario
    scenarios.append({
        "scenario": "baseline",
        "w_perf": base_weights["performance"],
        "w_eff": base_weights["efficiency"],
        "w_xai": base_weights["explainability"],
        "top_model": baseline_top,
        "top_score": baseline_df.loc[
            baseline_df["model"] == baseline_top, "final_score"
        ].values[0],
        "full_ranking": baseline_ranking,
        "ranking_matches_baseline": True,
    })

    # Perturbed scenarios: increase/decrease each pillar by delta
    for pillar in pillars:
        for direction in [+1, -1]:
            perturbed = base_weights.copy()
            shift = direction * delta

            # Adjust target pillar
            new_val = perturbed[pillar] + shift
            if new_val < 0.0 or new_val > 1.0:
                continue  # Skip invalid weight

            # Redistribute the shift proportionally across the other two
            others = [p for p in pillars if p != pillar]
            other_total = sum(perturbed[p] for p in others)

            if other_total < 1e-12:
                continue  # Can't redistribute

            for other_p in others:
                proportion = perturbed[other_p] / other_total
                perturbed[other_p] -= shift * proportion

            perturbed[pillar] = new_val

            # Ensure no negative weights after redistribution
            if any(perturbed[p] < 0 for p in pillars):
                continue

            # Compute ranking with perturbed weights
            label = f"{pillar[:4]}{'+'if direction>0 else '-'}{int(delta*100)}%"
            try:
                perturbed_df = compute_final_scores(
                    results_df,
                    faithfulness_scores, stability_scores,
                    weights=perturbed.copy(),
                )
                perturbed_ranking = perturbed_df.sort_values("rank")["model"].tolist()
                perturbed_top = perturbed_ranking[0]

                scenarios.append({
                    "scenario": label,
                    "w_perf": perturbed["performance"],
                    "w_eff": perturbed["efficiency"],
                    "w_xai": perturbed["explainability"],
                    "top_model": perturbed_top,
                    "top_score": perturbed_df.loc[
                        perturbed_df["model"] == perturbed_top, "final_score"
                    ].values[0],
                    "full_ranking": perturbed_ranking,
                    "ranking_matches_baseline": perturbed_ranking == baseline_ranking,
                })
            except Exception as e:
                logger.error(f"Sensitivity scenario '{label}' failed: {e}")

    summary_df = pd.DataFrame(scenarios)

    # Log summary
    n_stable = summary_df["ranking_matches_baseline"].sum()
    n_total = len(summary_df)
    logger.info(
        f"Sensitivity analysis: {n_stable}/{n_total} scenarios preserve "
        f"baseline ranking (top model: {baseline_top})"
    )
    for _, row in summary_df.iterrows():
        match_str = "SAME" if row["ranking_matches_baseline"] else "CHANGED"
        logger.info(
            f"  {row['scenario']:12s}  "
            f"w=({row['w_perf']:.2f}/{row['w_eff']:.2f}/{row['w_xai']:.2f})  "
            f"top={row['top_model']}  [{match_str}]"
        )

    return summary_df
