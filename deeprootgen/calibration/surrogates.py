"""Contains methods for implementing surrogate models for the root generator.

This module defines various utlities and methods for working with surrogate models.
"""


class EarlyStopper:
    """Early stopping for training."""

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """EarlyStopper constructor.

        Args:
            patience (int, optional):
                The number of iterations before performing early stopping. Defaults to 1.
            min_delta (float, optional):
                The minimum difference for the validation loss. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
