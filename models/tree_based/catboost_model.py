"""
CatBoost model for the IoMT IDS Benchmarking Framework.

Wraps CatBoost's CatBoostClassifier with Optuna HPO, early stopping,
and native model serialization.

Search space rationale:
    - iterations 300-1200: CatBoost's ordered boosting is slower per
      iteration but converges more stably; 1200 is an upper bound
      guarded by early stopping.
    - depth 4-10: CatBoost grows symmetric trees, so depth is the
      primary complexity knob.
    - learning_rate 0.02-0.2 (log): same gradient boosting trade-off.
    - l2_leaf_reg 1.0-10.0: L2 regularization per leaf value.
    - bagging_temperature 0.0-1.0: controls Bayesian bootstrap intensity.
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from optuna.trial import Trial

from models.base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost classifier with Optuna HPO integration and GPU support."""

    def __init__(self, random_state: int = 42):
        super().__init__(model_name="CatBoost", random_state=random_state)
        self.model = None
        self.task_type = self._detect_task_type()

    @staticmethod
    def _detect_task_type() -> str:
        """Return 'GPU' if CatBoost can use CUDA, else 'CPU'."""
        try:
            from catboost import CatBoostClassifier

            # Quick smoke test: fit a tiny model on GPU
            clf = CatBoostClassifier(
                iterations=1, depth=1, task_type="GPU",
                verbose=0, allow_writing_files=False,
            )
            import numpy as np
            clf.fit(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
            )
            return "GPU"
        except Exception:
            return "CPU"

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            ),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> Any:
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(
            random_seed=self.random_state,
            verbose=0,
            task_type=self.task_type,
            allow_writing_files=False,  # avoid catboost_info/ clutter on HPC
            **params,
        )
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 20,
        **kwargs,
    ) -> None:
        """Fit CatBoost with optional early stopping on validation loss."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        t0 = time.perf_counter()
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.training_time_seconds = time.perf_counter() - t0
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        return self.model.predict(X).flatten().astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        return self.model.predict_proba(X)

    # ---- Parameter count -------------------------------------------------

    def get_n_params(self) -> int:
        """Total nodes across all CatBoost trees."""
        if not self.is_fitted:
            return 0
        # CatBoost's tree_count_ gives the number of trees
        # Each symmetric tree of depth d has 2^d - 1 internal + 2^d leaves
        try:
            n_trees = self.model.tree_count_
            depth = self.model.get_param("depth") or 6
            nodes_per_tree = (2 ** (depth + 1)) - 1
            return n_trees * nodes_per_tree
        except Exception:
            return 0

    # ---- Serialization ---------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save using CatBoost's native format + metadata via joblib."""
        import os
        import joblib

        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_name": self.model_name,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted,
                "best_params": self.best_params,
                "training_time_seconds": self.training_time_seconds,
            },
            path,
            compress=3,
        )

    def load_checkpoint(self, path: str) -> None:
        import joblib

        data = joblib.load(path)
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
        self.best_params = data.get("best_params")
        self.training_time_seconds = data.get("training_time_seconds")
