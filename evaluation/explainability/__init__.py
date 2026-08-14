"""Explainability analysis module for the IoMT IDS Benchmarking Framework.

Two-stage approach:
    Stage 1: Lightweight SHAP global feature importance for ALL 8 models.
    Stage 2: Full SHAP + LIME + cross-validation for top model per family.
"""

from .shap_analysis import (
    run_shap_analysis,
    run_stage1_shap,
    run_stage2_shap,
    compute_faithfulness,
    compute_stability,
    get_global_feature_importance,
)
from .lime_analysis import (
    run_lime_analysis,
    generate_consistency_report,
)

__all__ = [
    # SHAP (Stage 1 + Stage 2)
    "run_shap_analysis",
    "run_stage1_shap",
    "run_stage2_shap",
    "compute_faithfulness",
    "compute_stability",
    "get_global_feature_importance",
    # LIME (Stage 2 only)
    "run_lime_analysis",
    "generate_consistency_report",
]
