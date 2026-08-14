"""
FT-Transformer (Feature Tokenizer Transformer) for the IoMT IDS Framework.

Custom PyTorch implementation — NOT using tab_transformer_pytorch library.
This gives us full control over the architecture, training loop, and
compatibility with our Optuna HPO and efficiency measurement infrastructure.

Architecture (Gorishniy et al., 2021):
    Input (batch, n_features)
    → Feature Tokenizer: each scalar feature is projected to embed_dim
      via a per-feature linear layer, producing (batch, n_features, embed_dim)
    → Prepend learnable [CLS] token → (batch, n_features+1, embed_dim)
    → Stack of TransformerBlocks (self-attention + FFN with pre-norm)
    → Extract [CLS] token representation
    → Linear(embed_dim, n_classes)

Key design choices:
    - Pre-layer normalization (LayerNorm before attention and FFN) following
      the original FT-Transformer paper. More stable than post-norm for
      deeper stacks.
    - No feature selection: config.yaml sets transformers_use_all=true.
      Conference paper showed 64-point F1 gap when MI feature selection
      was applied to transformers on Task 19.
    - embed_dim must be divisible by num_heads (enforced in search space).
    - CosineAnnealingWarmRestarts scheduler for transformers — handles
      the learning rate better than ReduceLROnPlateau for attention models.

Search space rationale:
    - embed_dim 32-128 (step 16): memory scales with n_features × embed_dim
    - num_heads 2/4/8: must divide embed_dim (validated in build_model)
    - depth 2-6: deeper captures more feature interactions via attention
    - attn_dropout 0.0-0.3: regularization on attention weights
    - ff_dropout 0.0-0.3: regularization on FFN activations
    - learning_rate 1e-4 to 3e-3: transformers are sensitive to LR
    - batch_size 256/512/1024: per config.yaml
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


class ReGLU(nn.Module):
    """ReGLU activation: ReLU-gated linear unit.

    Splits input in half along last dimension, applies ReLU to one half,
    and multiplies element-wise. Used in the FT-Transformer FFN following
    Gorishniy et al. (2021).

    Input dim must be 2 * desired output dim (the split halves it).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.relu(gate)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-layer normalization.

    Pre-norm: LayerNorm → Attention → Residual, LayerNorm → FFN → Residual.
    More stable for deep stacks than post-norm.

    Uses ReGLU activation in FFN following the original FT-Transformer paper.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(ff_dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        # ReGLU FFN: first linear outputs 2 * ffn_dim (split by ReGLU),
        # then project back to embed_dim
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim * 2),  # 2x for ReGLU split
            ReGLU(),                              # halves back to ffn_dim
            nn.Dropout(ff_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        # Pre-norm self-attention with residual + dropout on attention output
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.attn_dropout(attn_out)

        # Pre-norm ReGLU FFN with residual
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x


class FTTransformerNet(nn.Module):
    """Feature Tokenizer Transformer neural network.

    Each scalar feature is independently projected to an embedding space,
    a [CLS] token is prepended, the sequence passes through transformer
    blocks, and the [CLS] representation is used for classification.
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

        # Feature tokenizer: per-feature weight multiplication + bias.
        # Each scalar feature_i is projected to embed_dim via:
        #   token_i = x_i * weight_i + bias_i
        # This matches the Gorishniy et al. (2021) implementation:
        #   weight[None] * x_num[:, :, None]
        self.feature_weights = nn.Parameter(
            torch.empty(1, input_dim, embed_dim)
        )
        self.feature_biases = nn.Parameter(torch.empty(1, input_dim, embed_dim))
        nn.init.kaiming_uniform_(self.feature_weights, a=math.sqrt(5))
        nn.init.zeros_(self.feature_biases)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoder blocks
        ffn_dim = embed_dim * 4  # standard 4x expansion
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])

        # Final layer norm (after all blocks, before classification head)
        self.final_norm = nn.LayerNorm(embed_dim)

        # Classification head from [CLS] token
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, input_dim) — flat feature vector.

        Returns:
            (batch_size, n_classes) — logits.
        """
        batch_size = x.shape[0]

        # Feature tokenization: each feature_i → embed_dim vector
        # x: (batch, input_dim) → (batch, input_dim, 1)
        x = x.unsqueeze(-1)
        # Per-feature linear: (batch, input_dim, 1) * (1, input_dim, embed_dim)
        # → (batch, input_dim, embed_dim) via broadcasting
        tokens = x * self.feature_weights + self.feature_biases

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, embed_dim)

        # Transformer encoder blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Extract [CLS] token (position 0)
        cls_repr = self.final_norm(tokens[:, 0, :])

        return self.classifier(cls_repr)


class FTTransformerModel(BaseModel):
    """Feature Tokenizer Transformer with full training loop integration."""

    def __init__(self, random_state: int = 42):
        super().__init__(model_name="FTTransformer", random_state=random_state)
        self.device = None
        self.net: Optional[FTTransformerNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.n_classes: Optional[int] = None
        self._hparams: Dict[str, Any] = {}

    # ---- HPO search space ------------------------------------------------

    def get_optuna_search_space(self, trial: Trial) -> Dict[str, Any]:
        embed_dim = trial.suggest_int("embed_dim", 32, 128, step=16)
        # num_heads must divide embed_dim
        possible_heads = [h for h in [2, 4, 8] if embed_dim % h == 0]
        if not possible_heads:
            possible_heads = [2]  # fallback

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

    def build_model(self, **params) -> FTTransformerNet:
        from training.pytorch_utils import get_device

        self._hparams = params
        if self.device is None:
            self.device = get_device()

        if self.input_dim is None or self.n_classes is None:
            raise ValueError(
                "input_dim and n_classes must be set before building."
            )

        # Validate num_heads divides embed_dim
        embed_dim = params["embed_dim"]
        num_heads = params["num_heads"]
        if embed_dim % num_heads != 0:
            # Fallback: reduce to largest valid divisor
            for h in [4, 2, 1]:
                if embed_dim % h == 0:
                    num_heads = h
                    break

        self.net = FTTransformerNet(
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
        # Cosine annealing with warm restarts — better for transformers
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
