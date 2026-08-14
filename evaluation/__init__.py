"""Evaluation module for the IoMT IDS Benchmarking Framework.

Contains metrics computation, efficiency measurements, model selection,
explainability analysis, and cross-dataset generalization evaluation.
"""

from .metrics import compute_all_metrics, compute_efficiency_metrics, save_confusion_matrix
from .cross_dataset import (
    run_cross_dataset_evaluation,
    run_all_cross_dataset,
    evaluate_zero_shot,
    analyze_novel_classes,
)

__all__ = [
    # Core metrics
    'compute_all_metrics',
    'compute_efficiency_metrics',
    'save_confusion_matrix',
    # Cross-dataset generalization
    'run_cross_dataset_evaluation',
    'run_all_cross_dataset',
    'evaluate_zero_shot',
    'analyze_novel_classes',
]
