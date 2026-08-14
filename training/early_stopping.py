"""EarlyStopping utility class for PyTorch models.

Monitors validation metrics, saves best model state, and stops training
if no improvement is observed for a patience period.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback for PyTorch training.
    
    Monitors a validation metric and stops training if no improvement
    is observed for a specified number of epochs (patience).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        checkpoint_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs with no improvement after which
                      training will be stopped.
            min_delta: Minimum change in monitored metric to qualify as improvement.
            checkpoint_path: Path to save best model checkpoint.
            verbose: Whether to print status messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.early_stop = False

    def __call__(
        self,
        val_metric: float,
        model: torch.nn.Module,
        epoch: int,
    ) -> bool:
        """Check if training should stop.
        
        Args:
            val_metric: Current validation metric value (higher is better).
            model: PyTorch model to save if new best is achieved.
            epoch: Current epoch number.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_metric
            self.best_epoch = epoch
            self._save_checkpoint(model, epoch)
        elif val_metric > self.best_score + self.min_delta:
            self.best_score = val_metric
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, epoch)
            if self.verbose:
                logger.info(f"Epoch {epoch}: validation metric improved to {val_metric:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Epoch {epoch}: validation metric did not improve. "
                           f"Counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}. "
                               f"Best score was {self.best_score:.4f} at epoch {self.best_epoch}")
                self.early_stop = True
                return True
        
        return False

    def _save_checkpoint(self, model: torch.nn.Module, epoch: int) -> None:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model to save.
            epoch: Current epoch number.
        """
        if self.checkpoint_path is None:
            return

        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        if self.verbose:
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    @property
    def should_stop(self) -> bool:
        """Check if early stopping condition is met.
        
        Returns:
            True if early stopping should be triggered.
        """
        return self.early_stop
