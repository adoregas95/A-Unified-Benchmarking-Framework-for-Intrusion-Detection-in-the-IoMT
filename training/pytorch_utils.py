"""
Shared PyTorch training utilities for DL and Transformer models.

Provides a reusable training loop, DataLoader creation, and inference
helpers that all PyTorch-based models (CNN1D, BiLSTM, FTTransformer, SAINT)
use through composition rather than inheritance.

Design decisions:
    - Training loop is a standalone function, NOT a base class method.
      This keeps BaseModel clean (it serves tree-based models too) and
      avoids deep inheritance chains.
    - Gradient clipping (max_norm=1.0) is always applied for DL/Transformer
      models to prevent exploding gradients on imbalanced batches.
    - Early stopping monitors weighted F1 on the validation set with
      patience=10 (from config.yaml).
    - Mixed precision (AMP) is used when a CUDA device is available to
      reduce memory footprint and speed up transformer training.
    - LR scheduling uses CosineAnnealingWarmRestarts for transformers
      and ReduceLROnPlateau for DL models.
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 512,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create PyTorch DataLoaders from numpy arrays.

    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training labels (n_samples,).
        X_val: Validation features. If None, no val loader is returned.
        y_val: Validation labels.
        batch_size: Batch size for both train and val.
        num_workers: Number of data loading workers. None = auto-detect
            (min of 4 and available CPUs). 0 = main process only.

    Returns:
        (train_loader, val_loader) where val_loader may be None.
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
        logger.debug(f"DataLoader num_workers auto-set to {num_workers}")
    # Clamp extreme values to prevent NaN loss under mixed precision (AMP).
    # RobustScaler output is unbounded — CICFlowMeter outliers (e.g., DDoS
    # packet floods) can produce values of ±100+, which overflow float16
    # in AMP's autocast. Clamping to ±10 preserves the signal (outliers
    # remain clearly extreme) while keeping values in float16-safe range.
    X_train_safe = np.clip(X_train, -10, 10)
    if np.any(np.isnan(X_train_safe)) or np.any(np.isinf(X_train_safe)):
        logger.warning("NaN/Inf detected in training data after clipping — replacing with 0")
        X_train_safe = np.nan_to_num(X_train_safe, nan=0.0, posinf=10.0, neginf=-10.0)

    X_train_t = torch.FloatTensor(X_train_safe)
    y_train_t = torch.LongTensor(y_train.astype(np.int64))

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_safe = np.clip(X_val, -10, 10)
        X_val_safe = np.nan_to_num(X_val_safe, nan=0.0, posinf=10.0, neginf=-10.0)
        X_val_t = torch.FloatTensor(X_val_safe)
        y_val_t = torch.LongTensor(y_val.astype(np.int64))
        val_ds = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,  # larger batch for eval (no grads)
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader


def train_pytorch_model(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
    gradient_clip: float = 1.0,
    scheduler: Optional[object] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Universal PyTorch training loop with early stopping.

    Used by CNN1D, BiLSTM, FTTransformer, and SAINT.

    Args:
        net: PyTorch model (already on device).
        optimizer: Optimizer instance.
        criterion: Loss function (e.g., CrossEntropyLoss).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (required for early stopping).
        device: torch.device to train on.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        gradient_clip: Max gradient norm for clipping.
        scheduler: Optional LR scheduler. If it has a `step(metric)` method
            (like ReduceLROnPlateau), it is stepped with val_f1. Otherwise
            stepped per epoch.
        checkpoint_path: Path to save best model state_dict.
        verbose: Whether to log epoch-level progress.

    Returns:
        Dict with 'best_val_f1', 'best_epoch', 'total_epochs',
        'training_time_seconds'.
    """
    if patience > 0 and val_loader is None:
        logger.warning(
            "Early stopping requested (patience=%d) but no val_loader "
            "provided. Disabling early stopping — training will run for "
            "all %d epochs.",
            patience, max_epochs,
        )
        patience = 0

    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # ---- Training phase ----
        net.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = net(X_batch)
                    loss = criterion(logits, y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN/Inf loss at epoch {epoch}, batch {n_batches} — "
                        f"skipping gradient update"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = net(X_batch)
                loss = criterion(logits, y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN/Inf loss at epoch {epoch}, batch {n_batches} — "
                        f"skipping gradient update"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
                optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- Validation phase ----
        val_f1 = 0.0
        if val_loader is not None:
            val_f1 = _evaluate(net, val_loader, device, use_amp)

        # ---- LR scheduling ----
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # ReduceLROnPlateau — step with metric
                scheduler.step(val_f1)
            else:
                scheduler.step()

        if verbose and epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{max_epochs} — "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_f1={val_f1:.4f}, lr={current_lr:.2e}"
            )

        # ---- Early stopping ----
        if val_loader is not None:
            should_stop = early_stopper(val_f1, net, epoch)
            if should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_f1={early_stopper.best_score:.4f} "
                    f"at epoch {early_stopper.best_epoch}"
                )
                break

    training_time = time.perf_counter() - t0

    # Restore best weights if we have a checkpoint
    if checkpoint_path and early_stopper.best_score is not None:
        try:
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Restored best model from epoch {early_stopper.best_epoch}")
        except Exception as e:
            logger.warning(f"Could not restore checkpoint: {e}")

    return {
        "best_val_f1": early_stopper.best_score or val_f1,
        "best_epoch": early_stopper.best_epoch or epoch,
        "total_epochs": epoch,
        "training_time_seconds": training_time,
    }


def _evaluate(
    net: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Evaluate model on validation set, return weighted F1.

    Args:
        net: PyTorch model in eval mode.
        val_loader: Validation DataLoader.
        device: Device.
        use_amp: Whether to use mixed precision.

    Returns:
        Weighted F1 score.
    """
    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = net(X_batch)
            else:
                logits = net(X_batch)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def predict_with_model(
    net: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Run inference and return class predictions.

    Args:
        net: Trained PyTorch model.
        X: Features (n_samples, n_features).
        device: Device.
        batch_size: Inference batch size.

    Returns:
        Predicted labels (n_samples,).
    """
    net.eval()
    X_safe = np.clip(X, -10, 10)
    X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=10.0, neginf=-10.0)
    X_t = torch.FloatTensor(X_safe)
    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            logits = net(batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(all_preds)


def predict_proba_with_model(
    net: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Run inference and return class probabilities.

    Args:
        net: Trained PyTorch model.
        X: Features (n_samples, n_features).
        device: Device.
        batch_size: Inference batch size.

    Returns:
        Class probabilities (n_samples, n_classes).
    """
    net.eval()
    X_safe = np.clip(X, -10, 10)
    X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=10.0, neginf=-10.0)
    X_t = torch.FloatTensor(X_safe)
    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False,
    )

    all_probs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            logits = net(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available.

    Returns:
        torch.device for training/inference.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device
