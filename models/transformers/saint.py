"""
SAINT (Self-Attention and Intersample Attention Transformer) for the IoMT
IDS Benchmarking Framework.

Custom PyTorch implementation based on Somepalli et al. (2021):
"SAINT: Improved Neural Networks for Tabular Data via Row Attention
and Contrastive Pre-Training."

Architecture:
    Input (batch, n_features)
    → Feature Tokenizer: per-feature linear projection → (batch, n_features, embed_dim)
    → Prepend learnable [CLS] token → (batch, n_features+1, embed_dim)
    → Stack of SAINTBlocks, each containing:
        1. Self-Attention (feature-wise): standard transformer attention over
           the feature sequence. Each token attends to all other tokens.
           Shape: (batch, n_features+1, embed_dim).
        2. Intersample Attention (row-wise): attention ACROSS samples in the
           batch. The sequence dimension becomes the batch dimension and vice
           versa. Each sample attends to all other samples in the batch for
           each feature position independently.
           This is SAINT's unique contribution — it captures relationships
           between data points, not just between features.
    → Extract [CLS] token → Linear(embed_dim, n_classes)

Intersample Attention detail:
    Given tokens of shape (batch, seq, embed_dim), we transpose to
    (seq, batch, embed_dim) and run multi-head attention. This means for
    each feature token position, samples attend to each other. The transpose
    is reversed afterward. During inference on single samples, intersample
    attention degenerates to identity (only one sample to attend to), which
    is fine — the self-attention layers carry the classification signal.

Search space: Same as FTTransformer (embed_dim, num_heads, depth, dropouts,
learning_rate, batch_size). The architecture difference is in the block
structure, not the hyperparameters.
"""

import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.trial import Trial

from models.base import BaseModel


class SAINTBlock(nn.Module):
    """SAINT block: Self-Attention → Intersample Attention → FFN.

    Each block applies:
    1. Pre-norm self-attention over feature tokens (column attention, standard)
    2. Pre-norm intersample attention (row attention, SAINT's key contribution)
    3. Pre-norm FFN with residual connections after each attention stage

    Intersample attention follows the reference implementation (Somepalli et al.):
    all feature embeddings for each sample are flattened into a single vector
    (n_tokens * embed_dim), then attention is computed across samples in the
    batch. This allows cross-sample, cross-feature interactions — a sample can
    attend to different features of other samples simultaneously.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ff_dropout: float,
        seq_len: int = 0,
    ):
        super().__init__()

        self.seq_len = seq_len  # n_features + 1 (for CLS token)

        # --- Self-Attention (column/feature-wise) ---
        self.norm_sa = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # --- FFN after self-attention ---
        self.norm_ff1 = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(ff_dropout),
        )

        # --- Intersample Attention (row-wise) ---
        # Operates on flattened representations: each sample becomes a single
        # vector of size (seq_len * embed_dim), attention across batch samples.
        ia_dim = seq_len * embed_dim if seq_len > 0 else embed_dim
        self.norm_ia = nn.LayerNorm(ia_dim)
        self.intersample_attn = nn.MultiheadAttention(
            embed_dim=ia_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # --- FFN after intersample attention ---
        self.norm_ff2 = nn.LayerNorm(ia_dim)
        self.ff2 = nn.Sequential(
            nn.Linear(ia_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ffn_dim, ia_dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention + intersample attention.

        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # 1. Self-attention (column/feature-wise) with pre-norm + FFN
        normed = self.norm_sa(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        normed = self.norm_ff1(x)
        x = x + self.ff1(normed)

        # 2. Intersample attention (row-wise) following reference impl:
        #    Flatten: (batch, seq, embed) → (1, batch, seq*embed)
        #    Each sample becomes one token, attention across all samples
        x_flat = x.reshape(batch_size, seq_len * embed_dim)  # (batch, seq*embed)
        x_flat = x_flat.unsqueeze(0)  # (1, batch, seq*embed)

        normed = self.norm_ia(x_flat)
        ia_out, _ = self.intersample_attn(normed, normed, normed)
        x_flat = x_flat + ia_out

        normed = self.norm_ff2(x_flat)
        x_flat = x_flat + self.ff2(normed)

        # Reshape back: (1, batch, seq*embed) → (batch, seq, embed)
        x = x_flat.squeeze(0).reshape(batch_size, seq_len, embed_dim)

        return x


class SAINTNet(nn.Module):
    """SAINT neural network architecture.

    Uses MLP embedding for continuous features (following the reference
    implementation) and a multi-layer classification head.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        attn_dropout: float,
        ff_dropout: float,
        n_classes: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        # Feature embedding: per-feature MLP (scalar → embed_dim)
        # Following reference: simple_MLP([1, 100, dim]) per feature
        self.feature_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 100),
                nn.ReLU(),
                nn.Linear(100, embed_dim),
            )
            for _ in range(input_dim)
        ])

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Sequence length = n_features + 1 (CLS token)
        seq_len = input_dim + 1

        # SAINT encoder blocks
        ffn_dim = embed_dim * 4
        # num_heads for intersample attention must divide (seq_len * embed_dim)
        # We use the same num_heads; the ia_dim must be divisible
        ia_dim = seq_len * embed_dim
        # Adjust num_heads for intersample attention if needed
        ia_num_heads = num_heads
        while ia_dim % ia_num_heads != 0 and ia_num_heads > 1:
            ia_num_heads -= 1

        self.blocks = nn.ModuleList([
            SAINTBlock(
                embed_dim, num_heads, ffn_dim, attn_dropout, ff_dropout,
                seq_len=seq_len,
            )
            for _ in range(depth)
        ])

        # Fix intersample attention num_heads if needed
        for block in self.blocks:
            if ia_dim % num_heads != 0:
                block.intersample_attn = nn.MultiheadAttention(
                    embed_dim=ia_dim,
                    num_heads=ia_num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Classification head: deeper MLP following reference (4x → 2x → out)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, input_dim) — flat feature vector.

        Returns:
            (batch_size, n_classes) — logits.
        """
        batch_size = x.shape[0]

        # Feature embedding via per-feature MLPs
        # Each feature_i (scalar) → embed_dim vector via MLP
        tokens = torch.stack([
            self.feature_mlps[i](x[:, i:i+1])  # (batch, 1) → (batch, embed_dim)
            for i in range(self.input_dim)
        ], dim=1)  # (batch, input_dim, embed_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # SAINT encoder blocks (self-attention + intersample attention)
        for block in self.blocks:
            tokens = block(tokens)

        # Extract [CLS] representation
        cls_repr = self.final_norm(tokens[:, 0, :])

        return self.classifier(cls_repr)


class SAINTModel(BaseModel):
    """SAINT model with full training loop integration."""

    def __init__(self, random_state: int = 42):
        super().__init__(model_name="SAINT", random_state=random_state)
        self.device = None
        self.net: Optional[SAINTNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.n_classes: Optional[int] = None
        self._hparams: Dict[str, Any] = {}

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        embed_dim = trial.suggest_int("embed_dim", 32, 128, step=16)
        possible_heads = [h for h in [2, 4, 8] if embed_dim % h == 0]
        if not possible_heads:
            possible_heads = [2]

        return {
            "embed_dim": embed_dim,
            "num_heads": trial.suggest_categorical("num_heads", possible_heads),
            "depth": trial.suggest_int("depth", 2, 6),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
            "ff_dropout": trial.suggest_float("ff_dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [256, 512, 1024]
            ),
        }

    # ---- Build / Train / Predict -----------------------------------------

    def build_model(self, **params) -> SAINTNet:
        from training.pytorch_utils import get_device

        self._hparams = params
        if self.device is None:
            self.device = get_device()

        if self.input_dim is None or self.n_classes is None:
            raise ValueError(
                "input_dim and n_classes must be set before building."
            )

        embed_dim = params["embed_dim"]
        num_heads = params["num_heads"]
        if embed_dim % num_heads != 0:
            for h in [4, 2, 1]:
                if embed_dim % h == 0:
                    num_heads = h
                    break

        self.net = SAINTNet(
            input_dim=self.input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=params["depth"],
            attn_dropout=params["attn_dropout"],
            ff_dropout=params["ff_dropout"],
            n_classes=self.n_classes,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=params["learning_rate"],
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
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
