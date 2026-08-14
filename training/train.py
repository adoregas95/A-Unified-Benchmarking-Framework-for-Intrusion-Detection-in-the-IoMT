"""
Universal training script for the IoMT IDS Benchmarking Framework.

Runs one (model, dataset, task) combination end-to-end:
    1. Load cached preprocessed data (NPZ from preprocessing pipeline)
    2. Optuna HPO with TPE sampler on the validation split
    3. Retrain best configuration on train + val combined
    4. Evaluate on held-out test set
    5. Save metrics (CSV), checkpoint (joblib), and confusion matrix (PNG)

This script is called by scripts/run_single.py (which is submitted via
SLURM). It does NOT handle k-fold cross-validation — our experimental
design uses a single stratified holdout split created during preprocessing.

Important design decisions:
    - Optuna maximizes weighted F1 on the validation set.
    - After HPO, we retrain on train+val and evaluate on the untouched test set.
    - Tree-based models get 50 Optuna trials (config: hpo.budgets.tree_based).
    - DL/Transformer models get more trials (separate budgets in config.yaml).
    - Transformers receive all 76 features; tree/DL receive MI-selected features.
      This is handled at preprocessing time, not here.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import optuna
import yaml
from optuna.trial import Trial

# ---------------------------------------------------------------------------
# Add project root to path for clean imports
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from preprocessing.preprocessing import load_preprocessed_data

from models.base import BaseModel
from models.tree_based.random_forest import RandomForestModel
from models.tree_based.xgboost_model import XGBoostModel
from models.tree_based.lightgbm_model import LightGBMModel
from models.tree_based.catboost_model import CatBoostModel
from models.deep_learning.cnn_1d import CNN1DModel
from models.deep_learning.bilstm import BiLSTMModel
from models.transformers.ft_transformer import FTTransformerModel
from models.transformers.saint import SAINTModel

from evaluation.metrics import (
    compute_all_metrics,
    compute_bootstrap_ci,
    compute_efficiency_metrics,
    save_confusion_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry — maps string names to classes
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    # Tree-based ensemble methods
    "RandomForest": RandomForestModel,
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
    "CatBoost": CatBoostModel,
    # Deep learning
    "CNN1D": CNN1DModel,
    "BiLSTM": BiLSTMModel,
    # Transformer architectures
    "FTTransformer": FTTransformerModel,
    "SAINT": SAINTModel,
}

# Models that require input_dim and n_classes to be set before build_model()
_PYTORCH_MODELS = {"CNN1D", "BiLSTM", "FTTransformer", "SAINT"}


def get_model_instance(
    model_name: str,
    random_state: int = 42,
    input_dim: Optional[int] = None,
    n_classes: Optional[int] = None,
) -> BaseModel:
    """Instantiate a model by name from the registry.

    For PyTorch models (CNN1D, BiLSTM, FTTransformer, SAINT), input_dim
    and n_classes must be provided so the network can be built.

    Args:
        model_name: One of the keys in MODEL_REGISTRY.
        random_state: Seed for reproducibility.
        input_dim: Number of input features (required for PyTorch models).
        n_classes: Number of output classes (required for PyTorch models).

    Returns:
        Unbuilt BaseModel instance with input_dim/n_classes set if applicable.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[model_name]
    model = cls(random_state=random_state)

    # PyTorch models need input_dim and n_classes before build_model()
    if model_name in _PYTORCH_MODELS:
        model.input_dim = input_dim
        model.n_classes = n_classes

    return model


# ---------------------------------------------------------------------------
# Data loading from preprocessed cache
# ---------------------------------------------------------------------------

def load_data(
    dataset: str,
    task: int,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load preprocessed train/val/test splits from the NPZ cache.

    Args:
        dataset: Dataset name (e.g., "CICIoMT2024").
        task: Classification task (2, 6, or 19).
        cache_dir: Root cache directory (e.g., "preprocessing/cache").

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
    """
    logger.info(f"Loading preprocessed data: {dataset}/task_{task}")
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = (
        load_preprocessed_data(cache_dir, dataset, task)
    )
    logger.info(
        f"  Shapes — train: {X_train.shape}, val: {X_val.shape}, "
        f"test: {X_test.shape}"
    )
    logger.info(
        f"  Features: {X_train.shape[1]}, "
        f"Classes: {len(np.unique(y_train))}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

class HPOObjective:
    """Optuna objective: build model → train → evaluate on val → return F1."""

    def __init__(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        random_state: int = 42,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.training_config = training_config or {}

        # Precompute for PyTorch models
        self.input_dim = X_train.shape[1]
        self.n_classes = len(np.unique(y_train))

    def __call__(self, trial: Trial) -> float:
        """Evaluate one hyperparameter configuration.

        Returns:
            Weighted F1 on the validation set (higher is better).
        """
        # 1. Get fresh model instance (with input_dim/n_classes for PyTorch)
        model = get_model_instance(
            self.model_name,
            self.random_state,
            input_dim=self.input_dim,
            n_classes=self.n_classes,
        )

        # 2. Sample hyperparameters from the model's search space
        params = model.get_optuna_search_space(trial)

        # 3. Build and train
        model.build_model(**params)

        # Pass training config (max_epochs, patience, gradient_clip) for
        # PyTorch models; tree-based models ignore these kwargs
        train_kwargs = {}
        if self.model_name in _PYTORCH_MODELS:
            model_family = _get_model_family(self.model_name)
            train_kwargs["max_epochs"] = self.training_config.get(
                "max_epochs", {}
            ).get(model_family, 50)
            train_kwargs["patience"] = self.training_config.get(
                "early_stopping", {}
            ).get("patience", 10)
            train_kwargs["gradient_clip"] = self.training_config.get(
                "gradient_clip", 1.0
            )

        train_result = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            **train_kwargs,
        )

        # 4. Evaluate on validation set — return weighted F1
        y_pred = model.predict(self.X_val)
        from sklearn.metrics import f1_score

        val_f1 = f1_score(
            self.y_val, y_pred, average="weighted", zero_division=0
        )

        # Store best_epoch as a user attribute so we can retrieve it later
        # for fixed-epoch retraining (avoids data leakage from new val split)
        if isinstance(train_result, dict) and "best_epoch" in train_result:
            trial.set_user_attr("best_epoch", train_result["best_epoch"])

        # For boosting models (LightGBM, XGBoost, CatBoost): capture the
        # early-stopped iteration count so retrain can cap n_estimators.
        # Without this, retrain runs ALL n_estimators (e.g. 900) on the
        # SMOTEENN-resampled data, causing massive overfitting.
        if hasattr(model, "model") and model.model is not None:
            best_iter = getattr(model.model, "best_iteration_", None)
            if best_iter is None:
                best_iter = getattr(model.model, "best_iteration", None)
            if best_iter is not None and best_iter > 0:
                trial.set_user_attr("best_iteration", best_iter)
                logger.debug(
                    f"Trial {trial.number}: early stopped at iteration "
                    f"{best_iter}"
                )

        # Log for Optuna's pruning / progress tracking
        logger.debug(
            f"Trial {trial.number}: val_f1={val_f1:.4f}, params={params}"
        )
        return val_f1


def run_hpo(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    random_state: int = 42,
    storage_path: Optional[str] = None,
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization.

    Args:
        model_name: Model to optimize.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_trials: Number of Optuna trials.
        random_state: Seed for the TPE sampler.
        storage_path: Optional path for SQLite Optuna study persistence.
        training_config: Training section from config.yaml (for DL/Transformer
            max_epochs, patience, gradient_clip).

    Returns:
        Dict with 'best_params', 'best_value', 'n_trials', 'study_name'.
    """
    study_name = f"hpo_{model_name}"
    storage = None
    if storage_path:
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        storage = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = HPOObjective(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
        training_config=training_config,
    )

    # Suppress Optuna's per-trial logging (we log ourselves)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Record trial count before this run so we can scope best-trial
    # selection to only the current run's trials. When load_if_exists=True,
    # the study may contain trials from prior (possibly failed) runs.
    prior_trial_count = len(study.trials)
    if prior_trial_count > 0:
        logger.info(
            f"Study already contains {prior_trial_count} trials from "
            f"prior run(s) — will select best trial from current run only"
        )

    logger.info(f"Starting HPO: {n_trials} trials with TPE sampler")
    study.optimize(objective, n_trials=n_trials)

    # Select best trial from ONLY the current run's trials to avoid
    # picking hyperparameters from a prior (possibly failed) run.
    current_run_trials = [
        t for t in study.trials
        if t.number >= prior_trial_count
        and t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not current_run_trials:
        raise RuntimeError(
            f"No completed trials in current run "
            f"(trials {prior_trial_count}–{len(study.trials) - 1})"
        )
    best = max(current_run_trials, key=lambda t: t.value)
    logger.info(
        f"HPO complete. Best trial #{best.number} (of current run): "
        f"val_f1={best.value:.4f}, params={best.params}"
    )
    if prior_trial_count > 0:
        global_best = study.best_trial
        if global_best.number != best.number:
            logger.warning(
                f"Note: global best was trial #{global_best.number} "
                f"(val_f1={global_best.value:.4f}) from a prior run — "
                f"ignored in favor of current run's best"
            )

    return {
        "best_params": best.params,
        "best_value": best.value,
        "best_epoch": best.user_attrs.get("best_epoch", None),
        "best_iteration": best.user_attrs.get("best_iteration", None),
        "n_trials": n_trials,
        "study_name": study_name,
    }


# ---------------------------------------------------------------------------
# Retrain on full training data (train + val combined)
# ---------------------------------------------------------------------------

def retrain_best_model(
    model_name: str,
    best_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    training_config: Optional[Dict[str, Any]] = None,
    best_epoch: Optional[int] = None,
    best_iteration: Optional[int] = None,
) -> BaseModel:
    """Retrain the best configuration on train+val combined.

    After HPO selects the best hyperparameters using the validation split,
    we retrain on the full available data (train + val) so the final model
    has seen more examples. The test set remains completely untouched.

    For PyTorch models, we train for a fixed number of epochs equal to the
    best epoch found during HPO. This avoids creating a new validation split
    for early stopping (which would introduce a distribution mismatch with
    the HPO validation split — a subtle data leakage risk). Since HPO already
    determined the optimal stopping point, we simply train for that many
    epochs on the combined data.

    For boosting models (LightGBM, XGBoost, CatBoost), we cap n_estimators
    to the early-stopped iteration from HPO. Without this, the model trains
    all n_estimators (e.g. 900) without early stopping on the SMOTEENN-
    resampled data, which can cause severe overfitting — especially on
    high-class-count tasks where the synthetic distribution diverges most
    from the real test distribution.

    Args:
        model_name: Model to retrain.
        best_params: Best hyperparameters from HPO.
        X_train, y_train: Original training split.
        X_val, y_val: Original validation split.
        random_state: Seed.
        training_config: Training section from config.yaml.
        best_epoch: Best epoch from HPO (for PyTorch models). If None,
            falls back to max_epochs from config.
        best_iteration: Early-stopped iteration from HPO (for boosting
            models). If provided, overrides n_estimators in best_params.

    Returns:
        Trained BaseModel instance.
    """
    training_config = training_config or {}

    # Combine train + val
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    logger.info(
        f"Retraining {model_name} on combined train+val: "
        f"{X_full.shape[0]} samples, {X_full.shape[1]} features"
    )

    input_dim = X_full.shape[1]
    n_classes = len(np.unique(y_full))

    model = get_model_instance(
        model_name, random_state,
        input_dim=input_dim, n_classes=n_classes
    )
    model.build_model(**best_params)
    model.best_params = best_params

    if model_name in _PYTORCH_MODELS:
        # For PyTorch models: train for fixed epochs (no early stopping).
        # The best_epoch from HPO tells us the optimal stopping point.
        # Training on combined data for that many epochs avoids the need
        # for a new validation split, which would introduce distribution
        # mismatch with the HPO validation split.
        model_family = _get_model_family(model_name)
        max_epochs_config = training_config.get("max_epochs", {}).get(
            model_family, 50
        )
        retrain_epochs = best_epoch if best_epoch is not None else max_epochs_config

        logger.info(
            f"  PyTorch retrain: {retrain_epochs} fixed epochs "
            f"(best_epoch from HPO={'N/A' if best_epoch is None else best_epoch})"
        )

        model.train(
            X_full, y_full,
            X_val=None, y_val=None,
            max_epochs=retrain_epochs,
            patience=0,  # disabled — fixed epoch count
            gradient_clip=training_config.get("gradient_clip", 1.0),
        )
    else:
        # For boosting models: cap n_estimators to the early-stopped
        # iteration from HPO. This prevents the retrain from running all
        # n_estimators (e.g. 900) without early stopping, which causes
        # severe overfitting on SMOTEENN-resampled data.
        if best_iteration is not None and "n_estimators" in best_params:
            original_n = best_params["n_estimators"]
            # Rebuild the model with the capped n_estimators
            capped_params = dict(best_params)
            capped_params["n_estimators"] = best_iteration
            model.build_model(**capped_params)
            model.best_params = best_params  # keep original HPO params for record
            logger.info(
                f"  Boosting retrain: capped n_estimators from {original_n} "
                f"to {best_iteration} (early-stopped iteration from HPO)"
            )
        else:
            logger.info(
                f"  Tree retrain: using n_estimators="
                f"{best_params.get('n_estimators', 'default')} "
                f"(no early stopping info from HPO)"
            )
        model.train(X_full, y_full)

    return model


# ---------------------------------------------------------------------------
# Full pipeline: load → HPO → retrain → evaluate → save
# ---------------------------------------------------------------------------

def run_training_pipeline(
    model_name: str,
    dataset: str,
    task: int,
    config: Dict[str, Any],
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Execute the full training pipeline for one (model, dataset, task).

    This is the main entry point called by scripts/run_single.py.

    Args:
        model_name: Model name (key in MODEL_REGISTRY).
        dataset: Dataset name (e.g., "CICIoMT2024").
        task: Classification task (2, 6, or 19).
        config: Parsed config.yaml dict.
        output_dir: Root directory for results.

    Returns:
        Dict with all results (metrics, efficiency, HPO info, paths).
    """
    random_state = config["preprocessing"]["random_state"]
    cache_dir = os.path.join(project_root, config["preprocessing"]["cache_dir"])

    # Determine HPO budget based on model family
    model_family = _get_model_family(model_name)
    n_trials = config["hpo"]["budgets"].get(model_family, 50)

    # Optuna study storage
    storage_dir = os.path.join(project_root, config["hpo"]["storage_dir"])
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(
        storage_dir, f"{dataset}_task{task}_{model_name}.db"
    )

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_data(
        dataset, task, cache_dir
    )

    # Validate feature dimensions match metadata expectations
    expected_n_features = metadata.get("n_features_selected", X_train.shape[1])
    if X_train.shape[1] != expected_n_features:
        raise ValueError(
            f"Feature dimension mismatch: loaded {X_train.shape[1]} features "
            f"but metadata expects {expected_n_features}. "
            f"Preprocessed cache may be stale — re-run preprocessing."
        )
    expected_n_classes = metadata.get("n_classes", len(np.unique(y_train)))
    actual_n_classes = len(np.unique(y_train))
    if actual_n_classes != expected_n_classes:
        raise ValueError(
            f"Class count mismatch: found {actual_n_classes} classes in "
            f"training data but metadata expects {expected_n_classes}."
        )

    # Extract class names from metadata for reporting
    class_names = metadata.get("label_classes", None)

    # Training config (for DL/Transformer max_epochs, patience, etc.)
    training_config = config.get("training", {})

    # ------------------------------------------------------------------
    # 2. HPO
    # ------------------------------------------------------------------
    hpo_result = run_hpo(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=n_trials,
        random_state=random_state,
        storage_path=storage_path,
        training_config=training_config,
    )

    # ------------------------------------------------------------------
    # 3. Retrain on train + val
    # ------------------------------------------------------------------
    model = retrain_best_model(
        model_name=model_name,
        best_params=hpo_result["best_params"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
        training_config=training_config,
        best_epoch=hpo_result.get("best_epoch"),
        best_iteration=hpo_result.get("best_iteration"),
    )

    # ------------------------------------------------------------------
    # 4. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    test_metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=class_names,
    )

    logger.info(
        f"Test results — F1(weighted): {test_metrics['f1_weighted']:.4f}, "
        f"F1(macro): {test_metrics['f1_macro']:.4f}, "
        f"MCC: {test_metrics['mcc']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Efficiency metrics
    # ------------------------------------------------------------------
    logger.info("Measuring efficiency metrics...")
    efficiency = compute_efficiency_metrics(model, X_test, n_runs=10)

    # ------------------------------------------------------------------
    # 6. Bootstrap confidence intervals
    # ------------------------------------------------------------------
    logger.info("Computing bootstrap confidence intervals (1000 resamples)...")
    bootstrap_ci = compute_bootstrap_ci(
        y_true=y_test,
        y_pred=y_pred,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 7. Save everything
    # ------------------------------------------------------------------
    run_dir = os.path.join(
        project_root, output_dir, dataset, f"task_{task}", model_name
    )
    os.makedirs(run_dir, exist_ok=True)

    # 7a. Checkpoint
    checkpoint_path = os.path.join(run_dir, "model.pkl")
    model.save_checkpoint(checkpoint_path)

    # 7b. Confusion matrix
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_path=cm_path,
        title=f"{model_name} — {dataset} Task {task}",
    )

    # 7c. Results JSON (human-readable, complete record)
    results = {
        "model": model_name,
        "dataset": dataset,
        "task": task,
        "n_features": int(X_train.shape[1]),
        "n_train_samples": int(X_train.shape[0] + X_val.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "hpo": {
            "n_trials": hpo_result["n_trials"],
            "best_val_f1": hpo_result["best_value"],
            "best_params": hpo_result["best_params"],
        },
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if k != "classification_report_str"
        },
        "efficiency": efficiency,
        "bootstrap_ci": bootstrap_ci,
        "checkpoint_path": checkpoint_path,
        "confusion_matrix_path": cm_path,
    }

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # 7d. Classification report (text file for quick inspection)
    report_path = os.path.join(run_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset}, Task: {task}\n")
        f.write(f"Best HPO params: {hpo_result['best_params']}\n\n")
        f.write(test_metrics["classification_report_str"])

    # 7e. Save per-job result CSV (safe for concurrent SLURM execution)
    _save_individual_result_csv(
        os.path.join(project_root, output_dir),
        results,
    )

    logger.info(f"Pipeline complete for {model_name}/{dataset}/task_{task}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_family(model_name: str) -> str:
    """Map model name to its family for HPO budget lookup."""
    tree_models = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}
    dl_models = {"CNN1D", "BiLSTM"}
    transformer_models = {"FTTransformer", "SAINT"}

    if model_name in tree_models:
        return "tree_based"
    elif model_name in dl_models:
        return "deep_learning"
    elif model_name in transformer_models:
        return "transformers"
    else:
        return "tree_based"  # safe default


def _save_individual_result_csv(
    results_dir: str, results: Dict[str, Any]
) -> str:
    """Save a per-job result CSV (avoids race conditions with concurrent SLURM jobs).

    Each job writes its own file: results/{dataset}/task_{task}/{model}_result.csv.
    After all jobs complete, use merge_result_csvs() to combine into all_results.csv.

    Args:
        results_dir: Root results directory.
        results: Result dict from run_training_pipeline.

    Returns:
        Path to the written CSV file.
    """
    import csv

    # Flatten the nested dict into a single row
    test = results["test_metrics"]
    eff = results["efficiency"]
    row = {
        "model": results["model"],
        "dataset": results["dataset"],
        "task": results["task"],
        "n_features": results["n_features"],
        "n_train_samples": results["n_train_samples"],
        "n_test_samples": results["n_test_samples"],
        "hpo_n_trials": results["hpo"]["n_trials"],
        "hpo_best_val_f1": results["hpo"]["best_val_f1"],
        # Weighted-average metrics (primary for CICIoMT2024)
        "accuracy": test["accuracy"],
        "precision_weighted": test["precision_weighted"],
        "recall_weighted": test["recall_weighted"],
        "f1_weighted": test["f1_weighted"],
        # Macro-average metrics (primary for cross-dataset generalization)
        "precision_macro": test["precision_macro"],
        "recall_macro": test["recall_macro"],
        "f1_macro": test["f1_macro"],
        # MCC (complementary balanced measure)
        "mcc": test["mcc"],
        # Efficiency metrics
        "training_time_seconds": eff.get("training_time_seconds", ""),
        "inference_latency_ms": eff.get(
            "inference_latency_ms_per_sample", ""
        ),
        "throughput_samples_per_sec": eff.get(
            "batch_throughput_samples_per_sec", ""
        ),
        "model_parameter_count": eff.get("model_parameter_count", ""),
        "peak_memory_mb_inference": eff.get("peak_memory_mb_inference", ""),
        "gpu_power_watts": eff.get("gpu_power_watts_inference", ""),
        "energy_joules_per_sample": eff.get("energy_joules_per_sample", ""),
        # Bootstrap 95% CI bounds for key metrics
        "f1_weighted_ci_lower": results.get("bootstrap_ci", {}).get("f1_weighted", {}).get("ci_lower", ""),
        "f1_weighted_ci_upper": results.get("bootstrap_ci", {}).get("f1_weighted", {}).get("ci_upper", ""),
        "f1_macro_ci_lower": results.get("bootstrap_ci", {}).get("f1_macro", {}).get("ci_lower", ""),
        "f1_macro_ci_upper": results.get("bootstrap_ci", {}).get("f1_macro", {}).get("ci_upper", ""),
        "mcc_ci_lower": results.get("bootstrap_ci", {}).get("mcc", {}).get("ci_lower", ""),
        "mcc_ci_upper": results.get("bootstrap_ci", {}).get("mcc", {}).get("ci_upper", ""),
    }

    # Write to per-job file (no concurrent access risk)
    job_csv_path = os.path.join(
        results_dir,
        results["dataset"],
        f"task_{results['task']}",
        results["model"],
        "result.csv",
    )
    os.makedirs(os.path.dirname(job_csv_path), exist_ok=True)

    with open(job_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"Saved result CSV to {job_csv_path}")
    return job_csv_path


def merge_result_csvs(results_dir: str, output_path: Optional[str] = None) -> str:
    """Merge all per-job result CSVs into a single all_results.csv.

    Call this AFTER all training jobs have completed (not during concurrent
    execution). Scans for result.csv files under the results directory tree.

    Args:
        results_dir: Root results directory to scan.
        output_path: Path for the merged CSV. Defaults to
            {results_dir}/all_results.csv.

    Returns:
        Path to the merged CSV.
    """
    import csv
    import glob

    if output_path is None:
        output_path = os.path.join(results_dir, "all_results.csv")

    pattern = os.path.join(results_dir, "**", "result.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))

    if not csv_files:
        logger.warning(f"No result.csv files found under {results_dir}")
        return output_path

    # Read all individual CSVs and merge
    all_rows = []
    fieldnames = None
    for csv_file in csv_files:
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    # Write merged CSV atomically (write to temp, then rename)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp_path, output_path)

    logger.info(
        f"Merged {len(csv_files)} result files → {output_path} "
        f"({len(all_rows)} rows)"
    )
    return output_path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml from the default or specified path."""
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
