"""
LightGBM model for the IoMT IDS Benchmarking Framework.

Wraps LightGBM's LGBMClassifier with Optuna HPO, early stopping via
callbacks, GPU fallback, and joblib serialization.

Search space rationale:
    - n_estimators 200-1000: combined with early stopping, this sets an
      upper bound; the effective count is usually lower.
    - learning_rate 0.02-0.2 (log): same logic as XGBoost.
    - num_leaves 31-255: LightGBM grows leaf-wise, so num_leaves controls
      complexity more than max_depth. 255 is already very expressive.
    - max_depth 3-12: secondary guard against overfitting.
    - feature_fraction 0.6-0.9: column subsampling per tree (like colsample).
    - min_data_in_leaf 20-200: prevents tiny leaves on SMOTE artifacts.
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from optuna.trial import Trial

from models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM classifier with Optuna HPO integration and GPU fallback."""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        super().__init__(model_name="LightGBM", random_state=random_state)
        self.n_jobs = n_jobs
        self.model = None
        self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        """Return 'gpu' if LightGBM was built with GPU support, else 'cpu'."""
        try:
            import lightgbm as lgb

            # Quick smoke test: create a tiny booster with device_type=gpu
            # If it throws, fall back to cpu
            params = {"device_type": "gpu", "num_leaves": 2, "verbose": -1}
            train_data = lgb.Dataset(
                np.array([[0.0], [1.0]]), label=np.array([0, 1]), free_raw_data=False
            )
            lgb.train(params, train_data, num_boost_round=1, verbose_eval=False)
            return "gpu"
        except Exception:
            return "cpu"

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> Any:
        from lightgbm import LGBMClassifier

        self.model = LGBMClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            device_type=self.device,
            verbose=-1,
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
        """Fit LightGBM with optional early stopping via callbacks."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        import lightgbm as lgb

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),  # suppress per-iter logging
            ]

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
        """Total nodes across all boosted trees."""
        if not self.is_fitted:
            return 0
        booster = self.model.booster_
        # model_to_string contains tree dumps; count 'split' and 'leaf' lines
        dump = booster.model_to_string()
        # Each node line starts with whitespace and has 'split_feature' or 'leaf'
        return dump.count("split_feature=") + dump.count("leaf_value=")
