"""Training module for the IoMT IDS Benchmarking Framework.

Contains the universal training pipeline (HPO → retrain → evaluate),
early stopping callback for PyTorch models, and the model registry.
"""

from .early_stopping import EarlyStopping
from .train import (
    run_training_pipeline,
    run_hpo,
    retrain_best_model,
    load_data,
    load_config,
    get_model_instance,
    MODEL_REGISTRY,
)

__all__ = [
    "EarlyStopping",
    "run_training_pipeline",
    "run_hpo",
    "retrain_best_model",
    "load_data",
    "load_config",
    "get_model_instance",
    "MODEL_REGISTRY",
]
