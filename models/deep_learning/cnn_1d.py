"""
CNN1D model for the IoMT IDS Benchmarking Framework.

PyTorch-based 1D Convolutional Neural Network that treats the feature vector
as a 1D signal. Each feature becomes one position in a length-N sequence with
a single channel, then Conv1d layers extract local feature patterns.

Architecture:
    Input (batch, n_features)
    → Reshape to (batch, 1, n_features)  — 1 input channel
    → [Conv1d → BatchNorm → ReLU → Dropout] × n_conv_layers
      (filters double after the first layer, kernel_size stays constant)
    → AdaptiveAvgPool1d → Flatten
    → Linear(n_filters_last * pool_out, 128) → ReLU → Dropout
    → Linear(128, n_classes)

Search space rationale:
    - n_filters 32-128: start moderate, double each layer
    - kernel_size 3/5/7: odd sizes for symmetric receptive fields
    - n_conv_layers 2-4: deeper captures more feature interactions
    - dropout 0.1-0.5: regularization against SMOTEENN-expanded data
    - learning_rate 1e-4 to 3e-3: standard for Adam on tabular data
    - batch_size 256/512/1024: searched during HPO per config.yaml
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from optuna.trial import Trial

from models.base import BaseModel


class CNN1DNet(nn.Module):
    """1D CNN neural network architecture."""

    def __init__(
        self,
        input_dim: int,
        n_filters: int,
        kernel_size: int,
        n_conv_layers: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()

        layers = []
        in_channels = 1

        for i in range(n_conv_layers):
            out_channels = n_filters * (2 ** min(i, 2))  # cap doubling at 4x
            # Padding to preserve spatial dim: (kernel_size - 1) // 2
            padding = (kernel_size - 1) // 2
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        # Adaptive pool to fixed size regardless of input_dim
        self.pool = nn.AdaptiveAvgPool1d(4)

        # Classification head
        pool_flat = in_channels * 4
        self.classifier = nn.Sequential(
            nn.Linear(pool_flat, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, input_dim) — flat feature vector.

        Returns:
            (batch_size, n_classes) — logits.
        """
        # Reshape: (batch, features) → (batch, 1_channel, features)
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class CNN1DModel(BaseModel):
    """1D CNN model with full PyTorch training loop integration."""

    def __init__(self, random_state: int = 42):
        super().__init__(model_name="CNN1D", random_state=random_state)
        self.device = None  # set in build_model or train
        self.net: Optional[CNN1DNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.n_classes: Optional[int] = None
        self._hparams: Dict[str, Any] = {}

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        return {
            "n_filters": trial.suggest_int("n_filters", 32, 128, step=16),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "n_conv_layers": trial.suggest_int("n_conv_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [256, 512, 1024]
            ),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> CNN1DNet:
        from training.pytorch_utils import get_device

        self._hparams = params
        if self.device is None:
            self.device = get_device()

        # input_dim and n_classes are set by the training pipeline
        # before build_model is called
        if self.input_dim is None or self.n_classes is None:
            raise ValueError(
                "input_dim and n_classes must be set before building. "
                "The training pipeline sets these from the data shape."
            )

        self.net = CNN1DNet(
            input_dim=self.input_dim,
            n_filters=params["n_filters"],
            kernel_size=params["kernel_size"],
            n_conv_layers=params["n_conv_layers"],
            dropout=params["dropout"],
            n_classes=self.n_classes,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=params["learning_rate"],
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        self.model = self.net  # for BaseModel compatibility
        return self.net

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 50,
        patience: int = 10,
        gradient_clip: float = 1.0,
        **kwargs,
    ) -> dict:
        from training.pytorch_utils import (
            create_dataloaders,
            train_pytorch_model,
        )

        if self.net is None:
            raise ValueError("Model not built. Call build_model() first.")

        batch_size = self._hparams.get("batch_size", 512)
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=batch_size
        )

        result = train_pytorch_model(
            net=self.net,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            max_epochs=max_epochs,
            patience=patience,
            gradient_clip=gradient_clip,
            scheduler=self.scheduler,
            checkpoint_path=kwargs.get("checkpoint_path"),
        )

        self.training_time_seconds = result["training_time_seconds"]
        self.is_fitted = True
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        from training.pytorch_utils import predict_with_model
        return predict_with_model(self.net, X, self.device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        from training.pytorch_utils import predict_proba_with_model
        return predict_proba_with_model(self.net, X, self.device)

    # ---- Parameter count -------------------------------------------------

    def get_n_params(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    # ---- Serialization ---------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        if self.net is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "model_name": self.model_name,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted,
                "best_params": self.best_params,
                "training_time_seconds": self.training_time_seconds,
                "input_dim": self.input_dim,
                "n_classes": self.n_classes,
                "hparams": self._hparams,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.input_dim = data["input_dim"]
        self.n_classes = data["n_classes"]
        self._hparams = data["hparams"]
        self.best_params = data.get("best_params")
        self.training_time_seconds = data.get("training_time_seconds")

        # Rebuild network and load weights
        self.build_model(**self._hparams)
        self.net.load_state_dict(data["model_state_dict"])
        self.is_fitted = data["is_fitted"]
