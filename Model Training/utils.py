"""Shared training utilities are defined here."""

import random
from pathlib import Path

import numpy as np
import torch

from config import SEED, SEQ_DIR, TGT_LEN


def set_seed() -> None:
    """Random generators are initialised with the configured seed."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_device() -> torch.device:
    """The available CUDA, MPS, or CPU device is returned."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def coerce_numeric(array: np.ndarray) -> np.ndarray:
    """An array is converted to 32-bit floating-point values."""
    return array.astype(np.float32, copy=False)


def pad_or_trim(sequence: np.ndarray, length: int = TGT_LEN) -> np.ndarray:
    """The event-aligned tail is kept and right padding is added when needed."""
    sequence = coerce_numeric(sequence)
    if sequence.shape[0] >= length:
        return sequence[-length:, :]
    padding = np.full(
        (length - sequence.shape[0], sequence.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    return np.vstack([sequence, padding])


def make_abs_path(path: str | Path) -> str:
    """A stored sequence path is resolved from the sequence directory."""
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else SEQ_DIR / path)
