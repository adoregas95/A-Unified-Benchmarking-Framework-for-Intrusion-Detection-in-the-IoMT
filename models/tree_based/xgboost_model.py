"""
XGBoost model for the IoMT IDS Benchmarking Framework.

Wraps XGBoost's XGBClassifier with Optuna HPO, early stopping on the
validation set, GPU auto-detection, and native model serialization.

Search space rationale:
    - n_estimators 200-800: early stopping means we can be generous;
      the actual tree count is determined by val performance.
    - learning_rate 0.02-0.2 (log): lower LR + more trees is classic
      bias-variance trade-off for gradient boosting.
    - max_depth 4-10: XGB trees are grown level-wise; 10 is already deep.
    - subsample / colsample_bytree 0.6-1.0: regularization via stochastic
      gradient boosting.
    - reg_lambda 1e-3 to 10 (log): L2 regularization on leaf weights.
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from optuna.trial import Trial

from models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier with Optuna HPO integration and GPU support."""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        super().__init__(model_name="XGBoost", random_state=random_state)
        self.n_jobs = n_jobs
        self.model = None
        self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        """Return 'cuda' if a CUDA GPU is usable by XGBoost, else 'cpu'."""
        try:
            import xgboost as xgb

            # XGBoost ≥ 2.0 supports device='cuda'
            if hasattr(xgb, "build_info"):
                info = xgb.build_info()
                if info.get("USE_CUDA", False):
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> Any:
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            device=self.device,
            eval_metric="logloss",  # works for both binary and multi-class
            verbosity=0,
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
        """Fit XGBoost with optional early stopping on validation loss."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            # XGBoost ≥ 1.6 uses callbacks; set_params approach for compat
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)

        t0 = time.perf_counter()
        self.model.fit(X_train, y_train, **fit_kwargs)
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
        """Total nodes across all boosting rounds."""
        if not self.is_fitted:
            return 0
        booster = self.model.get_booster()
        # Each tree's dump has one line per node
        trees = booster.get_dump()
        return sum(len(t.strip().split("\n")) for t in trees)

    # ---- Serialization (use native XGBoost format) -----------------------

    def save_checkpoint(self, path: str) -> None:
        """Save using XGBoost's native binary format + metadata via joblib."""
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
