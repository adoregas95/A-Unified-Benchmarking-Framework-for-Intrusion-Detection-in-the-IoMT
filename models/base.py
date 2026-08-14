"""
Base model class for all models in the IoMT IDS Benchmarking Framework.

Defines the abstract interface that all model implementations must follow,
including training, prediction, hyperparameter optimization, serialization,
and efficiency measurement.

Design Notes:
    - Tree-based models use joblib for serialization (sklearn-compatible).
    - DL/Transformer models override save/load with torch.save/torch.load.
    - Timing is measured here (train + inference) so every model gets it
      automatically without duplicating timing code.
    - get_n_params() is abstract because counting differs between sklearn
      estimators and PyTorch modules.
"""

import time
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import joblib
from optuna.trial import Trial

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models in the framework.

    Provides the common interface for training, prediction, HPO, and
    serialization, plus built-in timing instrumentation for efficiency
    benchmarking.
    """

    def __init__(self, model_name: str, random_state: int = 42):
        """Initialize the base model.

        Args:
            model_name: Human-readable name (e.g., "RandomForest", "CNN1D").
            random_state: Seed for reproducibility.
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

        # Timing bookkeeping — filled automatically by train() / predict()
        self.training_time_seconds: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Abstract interface — every concrete model MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        """Define the Optuna hyperparameter search space.

        Args:
            trial: Optuna trial object for parameter sampling.

        Returns:
            Dict mapping parameter names to sampled values.
        """
        ...

    @abstractmethod
    def build_model(self, **params) -> Any:
        """Instantiate the underlying estimator with given hyperparameters.

        Must set self.model and return it.

        Args:
            **params: Hyperparameters (typically from get_optuna_search_space).

        Returns:
            The initialized estimator/module.
        """
        ...

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Train the model.

        Implementations MUST:
          1. Call self.model.fit (or equivalent training loop).
          2. Set self.is_fitted = True on success.
          3. Store self.training_time_seconds (use _time_training helper).

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).
            X_val: Optional validation features (for early stopping).
            y_val: Optional validation labels.
            **kwargs: Model-specific training arguments.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for each class.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Class probabilities (n_samples, n_classes).
        """
        ...

    @abstractmethod
    def get_n_params(self) -> int:
        """Return the number of trainable parameters / estimator complexity.

        For tree models: total number of tree nodes across all estimators.
        For DL models: sum of parameter tensor elements.

        Returns:
            Integer parameter count.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete methods — shared across all models
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Get current hyperparameters from the underlying estimator.

        Works for any sklearn-compatible model that implements get_params().
        PyTorch models should override to return their config dict.
        """
        if self.model is None:
            return {}
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def set_params(self, **params) -> "BaseModel":
        """Set hyperparameters on the underlying estimator.

        Args:
            **params: Parameters to set.

        Returns:
            Self for method chaining.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self

    def save_checkpoint(self, path: str) -> None:
        """Save model state to disk using joblib.

        This default implementation works for sklearn-compatible estimators.
        PyTorch models should override with torch.save().

        Args:
            path: Destination file path (e.g., "checkpoints/rf_task2.pkl").
        """
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
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Source file path.
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
        self.best_params = data.get("best_params")
        self.training_time_seconds = data.get("training_time_seconds")
        logger.info(f"Checkpoint loaded from {path}")

    # ------------------------------------------------------------------
    # Efficiency measurement helpers
    # ------------------------------------------------------------------

    def measure_inference_latency(
        self, X: np.ndarray, n_runs: int = 10
    ) -> Dict[str, float]:
        """Measure per-sample inference latency over multiple runs.

        Args:
            X: Input features (n_samples, n_features).
            n_runs: Number of timed prediction passes.

        Returns:
            Dict with 'mean_latency_ms', 'std_latency_ms',
            'throughput_samples_per_sec'.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(X)
            times.append(time.perf_counter() - t0)

        times_arr = np.array(times)
        n_samples = X.shape[0]
        mean_total = float(times_arr.mean())
        std_total = float(times_arr.std())

        return {
            "mean_latency_ms_per_sample": (mean_total / n_samples) * 1000,
            "std_latency_ms_per_sample": (std_total / n_samples) * 1000,
            "throughput_samples_per_sec": n_samples / mean_total,
            "total_inference_time_seconds": mean_total,
        }

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(model_name='{self.model_name}', {fitted})"
