"""
BiLSTM model for the IoMT IDS Benchmarking Framework.

PyTorch-based Bidirectional LSTM for tabular data classification. Each
feature vector is treated as a sequence of length 1 with input_dim
dimensions (i.e., the LSTM processes the entire feature vector as a
single timestep, using the hidden state to capture feature dependencies).

Alternatively, features can be chunked into groups to create a multi-step
sequence. We use the single-step approach (seq_len=1) because:
  1. CICFlowMeter features have no natural temporal ordering within a flow
  2. The BiLSTM hidden state still captures inter-feature dependencies
  3. It matches the approach used in related IoT-IDS literature

Architecture:
    Input (batch, n_features)
    → Reshape to (batch, 1, n_features)  — seq_len=1, input_size=n_features
    → BiLSTM(n_layers, hidden_dim) with dropout between layers
    → Concatenate final forward and backward hidden states
    → Linear(hidden_dim * 2, 128) → ReLU → Dropout
    → Linear(128, n_classes)

Search space rationale:
    - hidden_dim 64-256: captures feature interactions in hidden state
    - n_layers 1-3: deeper LSTM for more abstract representations
    - dropout 0.1-0.5: regularization
    - learning_rate 1e-4 to 3e-3: standard Adam range
    - batch_size 256/512/1024: per config.yaml
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from optuna.trial import Trial

from models.base import BaseModel


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM neural network architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM: input_size = n_features, processes as single timestep
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Classification head: BiLSTM output is 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
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
        # Reshape: (batch, features) → (batch, seq_len=1, features)
        x = x.unsqueeze(1)

        # LSTM forward pass
        # output: (batch, seq_len, 2*hidden_dim)
        # h_n: (2*n_layers, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(x)

        # Concatenate final forward and backward hidden states
        # h_n[-2] = last forward, h_n[-1] = last backward
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)

        return self.classifier(hidden)


class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model with full PyTorch training loop integration."""

    def __init__(self, random_state: int = 42):
        super().__init__(model_name="BiLSTM", random_state=random_state)
        self.device = None
        self.net: Optional[BiLSTMNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.n_classes: Optional[int] = None
        self._hparams: Dict[str, Any] = {}

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [256, 512, 1024]
            ),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> BiLSTMNet:
        from training.pytorch_utils import get_device

        self._hparams = params
        if self.device is None:
            self.device = get_device()

        if self.input_dim is None or self.n_classes is None:
            raise ValueError(
                "input_dim and n_classes must be set before building."
            )

        self.net = BiLSTMNet(
            input_dim=self.input_dim,
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
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

        self.model = self.net
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

        self.build_model(**self._hparams)
        self.net.load_state_dict(data["model_state_dict"])
        self.is_fitted = data["is_fitted"]
