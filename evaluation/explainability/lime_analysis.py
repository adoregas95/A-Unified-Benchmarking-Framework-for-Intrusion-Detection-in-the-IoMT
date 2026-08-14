"""
LIME analysis module for local model interpretability.

Implements Stage 2 local explanations using LIME (Local Interpretable
Model-agnostic Explanations). Applied to the top model per family (3 models)
selected by model_selection.select_best_per_family().

LIME generates per-instance explanations by:
    1. Perturbing the input around the instance of interest.
    2. Querying the model on perturbed samples.
    3. Fitting a weighted linear surrogate model.
    4. Extracting feature contributions from the surrogate.

SHAP-LIME cross-validation (Nugraha et al.):
    For each explained instance, compare SHAP's top-ℓ features with LIME's
    top-ℓ features. The consistency coefficient κ = |S_SHAP ∩ S_LIME| / ℓ.
    κ is computed at two values of ℓ: ℓ=5 (primary, following Nugraha et al.)
    and ℓ=10 (supplementary, sensitivity check). If κ_l5 < 0.50 for an
    instance, it is flagged for manual review.

Literature basis: Kalakoti et al. (SHAP + LIME on CICIoMT2024 with
transformers), Nugraha et al. (SHAP-LIME cross-validation with κ coefficient),
XAI-XGBoost (SHAP + LIME on WUSTL-EHMS-2020).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default number of perturbed samples for LIME surrogate model
DEFAULT_NUM_LIME_SAMPLES = 5000
# Number of instances to explain in Stage 2
DEFAULT_NUM_INSTANCES = 20
# Top-ℓ features for SHAP-LIME consistency check
# Primary ℓ=5 follows Nugraha et al.; supplementary ℓ=10 shows sensitivity
CONSISTENCY_TOP_L_PRIMARY = 5
CONSISTENCY_TOP_L_SUPPLEMENTARY = 10
# Consistency threshold below which instances are flagged (applied to primary)
CONSISTENCY_KAPPA_THRESHOLD = 0.50


def run_lime_analysis(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    output_dir: str,
    shap_values: Optional[np.ndarray] = None,
    num_lime_samples: int = DEFAULT_NUM_LIME_SAMPLES,
    num_instances: int = DEFAULT_NUM_INSTANCES,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run LIME analysis and SHAP-LIME cross-validation for a model.

    This is a Stage 2 function — applied only to the top model per family.

    Args:
        model: Trained BaseModel instance with predict_proba().
        model_name: Model identifier.
        X_train: Training features (used as LIME's background distribution).
        X_test: Test features.
        y_test: True labels for test set.
        feature_names: Feature names.
        class_names: Class label names.
        output_dir: Directory for saving plots and results.
        shap_values: Precomputed SHAP values for X_test (from Stage 2 SHAP).
            If provided, SHAP-LIME cross-validation is computed.
        num_lime_samples: Number of perturbed samples per LIME explanation.
        num_instances: Number of test instances to explain.
        random_state: For reproducibility.

    Returns:
        Dict with keys:
            - instance_explanations: List[Dict] per instance
            - consistency_scores_l5: List[float] (κ at ℓ=5 per instance)
            - consistency_scores_l10: List[float] (κ at ℓ=10 per instance)
            - mean_consistency_l5: float (mean κ at ℓ=5)
            - mean_consistency_l10: float (mean κ at ℓ=10)
            - flagged_instances: List[int] (indices with κ_l5 < threshold)
            - computation_time_seconds: float
    """
    import lime
    import lime.lime_tabular

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    t0 = time.time()

    # Initialize LIME explainer with training data distribution
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
        discretize_continuous=True,
    )

    # Select instances: mix of correct and misclassified predictions
    predictions = model.predict(X_test)
    correct_mask = predictions == y_test
    misclassified_mask = ~correct_mask

    n_correct = min(num_instances // 2, int(correct_mask.sum()))
    n_misclass = min(num_instances - n_correct, int(misclassified_mask.sum()))

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

    # If we couldn't fill the quota, add random instances
    remaining = num_instances - len(selected_idx)
    if remaining > 0:
        available = list(set(range(X_test.shape[0])) - set(selected_idx))
        if available:
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected_idx.extend(extra.tolist())

    logger.info(
        f"LIME explaining {len(selected_idx)} instances for {model_name} "
        f"({n_correct} correct, {n_misclass} misclassified)"
    )

    # --- Generate LIME explanations ---
    instance_explanations = []
    consistency_scores_l5 = []
    consistency_scores_l10 = []
    flagged_instances = []

    for i, sample_idx in enumerate(selected_idx):
        instance = X_test[sample_idx]
        true_label = int(y_test[sample_idx])
        pred_label = int(predictions[sample_idx])
        is_correct = true_label == pred_label

        try:
            explanation = explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=len(feature_names),
                num_samples=num_lime_samples,
                top_labels=1,
            )

            # Extract feature contributions for the predicted class
            lime_features = explanation.as_list(label=pred_label)
            # lime_features is a list of (feature_rule_str, weight) tuples

            # Map LIME feature rules back to feature names and weights
            lime_importance = _parse_lime_features(
                lime_features, feature_names,
            )

            # Save HTML explanation
            html_path = os.path.join(
                output_dir,
                f"{model_name}_lime_instance_{i}.html",
            )
            try:
                explanation.save_to_file(html_path)
            except Exception as e:
                logger.warning(f"Could not save LIME HTML for instance {i}: {e}")
                html_path = None

            # Save bar plot
            plot_path = _save_lime_bar_plot(
                lime_importance, model_name, sample_idx,
                true_label, pred_label, is_correct,
                class_names, output_dir, i,
            )

            # --- SHAP-LIME cross-validation (Nugraha et al.) ---
            # Compute κ at both ℓ=5 (primary) and ℓ=10 (supplementary)
            kappa_l5 = None
            kappa_l10 = None
            if shap_values is not None:
                kappa_l5 = _compute_consistency_kappa(
                    shap_values[sample_idx],
                    lime_importance,
                    feature_names,
                    top_l=CONSISTENCY_TOP_L_PRIMARY,
                )
                kappa_l10 = _compute_consistency_kappa(
                    shap_values[sample_idx],
                    lime_importance,
                    feature_names,
                    top_l=CONSISTENCY_TOP_L_SUPPLEMENTARY,
                )
                consistency_scores_l5.append(kappa_l5)
                consistency_scores_l10.append(kappa_l10)
                # Flagging based on primary ℓ=5
                if kappa_l5 < CONSISTENCY_KAPPA_THRESHOLD:
                    flagged_instances.append(sample_idx)
                    logger.warning(
                        f"Instance {sample_idx}: κ_l5={kappa_l5:.2f} < {CONSISTENCY_KAPPA_THRESHOLD} "
                        f"— SHAP-LIME disagreement flagged"
                    )

            instance_explanations.append({
                "instance_idx": int(sample_idx),
                "true_label": true_label,
                "predicted_label": pred_label,
                "is_correct": is_correct,
                "lime_top_features": lime_importance[:10],
                "consistency_kappa_l5": kappa_l5,
                "consistency_kappa_l10": kappa_l10,
                "html_path": html_path,
                "plot_path": plot_path,
            })

        except Exception as e:
            logger.error(f"LIME explanation failed for instance {sample_idx}: {e}")
            instance_explanations.append({
                "instance_idx": int(sample_idx),
                "true_label": true_label,
                "predicted_label": pred_label,
                "is_correct": is_correct,
                "lime_top_features": [],
                "consistency_kappa_l5": None,
                "consistency_kappa_l10": None,
                "error": str(e),
            })

    computation_time = time.time() - t0
    mean_consistency_l5 = (
        float(np.mean(consistency_scores_l5)) if consistency_scores_l5 else None
    )
    mean_consistency_l10 = (
        float(np.mean(consistency_scores_l10)) if consistency_scores_l10 else None
    )

    result = {
        "instance_explanations": instance_explanations,
        "consistency_scores_l5": consistency_scores_l5,
        "consistency_scores_l10": consistency_scores_l10,
        "mean_consistency_l5": mean_consistency_l5,
        "mean_consistency_l10": mean_consistency_l10,
        "flagged_instances": flagged_instances,
        "n_flagged": len(flagged_instances),
        "computation_time_seconds": computation_time,
    }

    logger.info(
        f"LIME for {model_name}: {computation_time:.1f}s, "
        f"{len(selected_idx)} instances explained"
    )
    if mean_consistency_l5 is not None:
        logger.info(
            f"SHAP-LIME consistency (ℓ=5): mean κ={mean_consistency_l5:.3f}, "
            f"{len(flagged_instances)} flagged (κ < {CONSISTENCY_KAPPA_THRESHOLD})"
        )
    if mean_consistency_l10 is not None:
        logger.info(
            f"SHAP-LIME consistency (ℓ=10): mean κ={mean_consistency_l10:.3f}"
        )

    return result


# ---------------------------------------------------------------------------
# SHAP-LIME cross-validation (Nugraha et al.)
# ---------------------------------------------------------------------------

def _compute_consistency_kappa(
    shap_values_instance: np.ndarray,
    lime_importance: List[Tuple[str, float]],
    feature_names: List[str],
    top_l: int = CONSISTENCY_TOP_L_PRIMARY,
) -> float:
    """Compute SHAP-LIME consistency coefficient κ for a single instance.

    κ = |S_SHAP ∩ S_LIME| / ℓ

    where S_SHAP and S_LIME are the sets of top-ℓ features from each method.

    Args:
        shap_values_instance: SHAP values for this instance (n_features,).
        lime_importance: LIME feature importances [(name, weight), ...].
        feature_names: All feature names.
        top_l: Number of top features to compare.

    Returns:
        κ in [0, 1]. 1.0 = perfect agreement, 0.0 = no overlap.
    """
    # Top-ℓ from SHAP
    abs_shap = np.abs(shap_values_instance)
    shap_top_idx = np.argsort(-abs_shap)[:top_l]
    shap_top_names = set(feature_names[j] if isinstance(feature_names, np.ndarray)
                         else feature_names[j] for j in shap_top_idx)

    # Top-ℓ from LIME
    lime_top_names = set()
    for name, _ in lime_importance[:top_l]:
        lime_top_names.add(name)

    if top_l == 0:
        return 1.0

    intersection = shap_top_names & lime_top_names
    kappa = len(intersection) / top_l

    return kappa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_lime_features(
    lime_features: List[Tuple[str, float]],
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Parse LIME feature rules back to feature names with importance weights.

    LIME returns feature rules like "Flow Duration > 1234.56" or
    "0.10 < Fwd Pkt Len Mean <= 0.45". We map these back to the original
    feature names and keep the absolute weight for ranking.

    Args:
        lime_features: Raw LIME output [(rule_string, weight), ...].
        feature_names: Original feature names.

    Returns:
        List of (feature_name, abs_weight) sorted by |weight| descending.
    """
    parsed = []
    for rule_str, weight in lime_features:
        matched_name = None
        for fname in feature_names:
            if fname in rule_str:
                matched_name = fname
                break
        if matched_name is None:
            # Fall back to rule string itself
            matched_name = rule_str
        parsed.append((matched_name, float(weight)))

    # Sort by absolute weight descending
    parsed.sort(key=lambda x: abs(x[1]), reverse=True)
    return parsed


def _save_lime_bar_plot(
    lime_importance: List[Tuple[str, float]],
    model_name: str,
    sample_idx: int,
    true_label: int,
    pred_label: int,
    is_correct: bool,
    class_names: List[str],
    output_dir: str,
    plot_idx: int,
    top_k: int = 10,
) -> Optional[str]:
    """Save horizontal bar plot of LIME feature contributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Publication-quality style (matches shap_analysis.py settings)
    _pub_rc = {
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
    }
    try:
        import seaborn as sns
        sns.set_theme(style="ticks", context="paper", font="serif", rc=_pub_rc)
    except ImportError:
        plt.rcParams.update(_pub_rc)

    _pos_color = "#1B365D"     # Navy
    _neg_color = "#C0392B"     # Crimson

    try:
        top = lime_importance[:top_k]
        names = [t[0] for t in top][::-1]
        values = [t[1] for t in top][::-1]
        colors = [_neg_color if v < 0 else _pos_color for v in values]

        fig, ax = plt.subplots(figsize=(8, max(4.5, top_k * 0.38)))
        ax.barh(names, values, color=colors, edgecolor="white",
                linewidth=0.3, height=0.65)
        ax.axvline(x=0, color="#333333", linewidth=0.8)

        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        status = "CORRECT" if is_correct else "MISCLASSIFIED"

        ax.set_title(
            f"LIME — {model_name} Instance #{sample_idx} [{status}]\n"
            f"True: {true_name}  |  Predicted: {pred_name}",
            fontweight="bold", fontsize=12, pad=10,
        )
        ax.set_xlabel("Feature contribution", fontweight="bold", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()

        plot_path = os.path.join(
            output_dir,
            f"{model_name}_lime_instance_{plot_idx}_{status.lower()}.png",
        )
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    except Exception as e:
        logger.warning(f"LIME bar plot failed: {e}")
        return None


def generate_consistency_report(
    lime_results: Dict[str, Any],
    model_name: str,
    output_dir: str,
) -> str:
    """Generate a text report summarizing SHAP-LIME consistency results.

    Args:
        lime_results: Output from run_lime_analysis().
        model_name: Model identifier.
        output_dir: Where to save the report.

    Returns:
        Path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{model_name}_consistency_report.txt")

    mean_l5 = lime_results.get('mean_consistency_l5')
    mean_l10 = lime_results.get('mean_consistency_l10')

    lines = [
        f"SHAP-LIME Consistency Report — {model_name}",
        "=" * 60,
        f"Number of instances explained: {len(lime_results['instance_explanations'])}",
        f"Mean consistency κ (ℓ=5, primary):       {mean_l5:.4f}" if mean_l5 is not None else "Mean consistency κ (ℓ=5): N/A",
        f"Mean consistency κ (ℓ=10, supplementary): {mean_l10:.4f}" if mean_l10 is not None else "Mean consistency κ (ℓ=10): N/A",
        f"Flagged instances (κ_l5 < {CONSISTENCY_KAPPA_THRESHOLD}): "
        f"{lime_results['n_flagged']}",
        "",
        "Per-Instance Details:",
        "-" * 60,
    ]

    for exp in lime_results["instance_explanations"]:
        status = "CORRECT" if exp["is_correct"] else "MISCLASS"
        k5 = exp.get("consistency_kappa_l5")
        k10 = exp.get("consistency_kappa_l10")
        k5_str = f"κ5={k5:.3f}" if k5 is not None else "κ5=N/A"
        k10_str = f"κ10={k10:.3f}" if k10 is not None else "κ10=N/A"
        flag = " *** FLAGGED" if (k5 is not None and k5 < CONSISTENCY_KAPPA_THRESHOLD) else ""

        lines.append(
            f"  Instance #{exp['instance_idx']:5d} [{status:8s}] "
            f"true={exp['true_label']} pred={exp['predicted_label']} "
            f"{k5_str}  {k10_str}{flag}"
        )

    lines.extend([
        "",
        "-" * 60,
        f"Computation time: {lime_results['computation_time_seconds']:.1f}s",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Consistency report saved to {report_path}")
    return report_path
