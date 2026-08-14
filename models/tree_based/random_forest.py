"""
RandomForest model for the IoMT IDS Benchmarking Framework.

Wraps sklearn's RandomForestClassifier with Optuna HPO, efficiency
measurement, and joblib serialization.

Search space rationale:
    - n_estimators 200-1000: fewer trees under-fit the 19-class task;
      more than 1000 gives diminishing returns on CICFlowMeter features.
    - max_depth 8-40: shallow trees miss interaction effects; very deep
      trees don't hurt much thanks to bagging, but waste memory.
    - max_features: sqrt/log2/None — standard RF wisdom.
    - min_samples_split/leaf: regularization knobs to avoid memorizing
      SMOTE-generated minority samples.
"""

import time
from typing import Any, Dict, Optional

import numpy as np
import joblib
from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier

from models.base import BaseModel


class RandomForestModel(BaseModel):
    """RandomForest classifier with Optuna HPO integration."""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        super().__init__(model_name="RandomForest", random_state=random_state)
        self.n_jobs = n_jobs
        self.model: Optional[RandomForestClassifier] = None

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 8, 40),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> RandomForestClassifier:
        self.model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **params,
        )
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Fit the RandomForest.

        RF doesn't support native early stopping, so X_val/y_val are
        ignored here but kept in the signature for interface consistency.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.training_time_seconds = time.perf_counter() - t0
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        return self.model.predict_proba(X)

    # ---- Parameter count -------------------------------------------------

    def get_n_params(self) -> int:
        """Total number of nodes across all trees in the forest."""
        if not self.is_fitted:
            return 0
        return sum(
            tree.tree_.node_count for tree in self.model.estimators_
        )
