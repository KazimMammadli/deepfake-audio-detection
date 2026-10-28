"""
src.training — Training procedures and preprocessing adapters.
"""

from src.training.preprocessing import preprocess_for_model
from src.training.trainer import (
    train_baseline_model,
    train_transfer_model,
    get_preprocessed_datasets,
)

__all__ = [
    "preprocess_for_model",
    "train_baseline_model",
    "train_transfer_model",
    "get_preprocessed_datasets",
]
