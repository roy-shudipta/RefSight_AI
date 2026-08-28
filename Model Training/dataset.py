"""Matched query lists are converted to model-ready tensors here."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, TGT_LEN, TOP_K_CORRECT
from utils import coerce_numeric, pad_or_trim


class QueryListDataset(Dataset):
    """One item contains one incorrect clip and seven correct clips."""

    def __init__(
        self,
        items: List[Tuple[object, List[str], List[int], List[int]]],
        scaler: Dict[str, np.ndarray],
    ) -> None:
        self.items = items
        self.keep_cols = scaler["keep_cols"].astype(int)
        self.mean = scaler["mean"].astype(np.float32)
        self.std = scaler["std"].astype(np.float32)
        print(
            f"[dataset] queries={len(self.items)}, features={len(self.keep_cols)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.items)

    def _prepare_sequence(self, file_path: str, shared_length: int) -> np.ndarray:
        """One sequence is trimmed, padded, and standardised."""
        sequence = coerce_numeric(np.load(file_path, allow_pickle=False))
        sequence = sequence[:, self.keep_cols]
        sequence = sequence[-shared_length:, :]
        sequence = pad_or_trim(sequence, TGT_LEN)
        sequence = (sequence - self.mean) / self.std
        return np.nan_to_num(
            sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def __getitem__(self, index: int):
        query_id, file_paths, lengths, ranks = self.items[index]
        expected_candidates = 1 + TOP_K_CORRECT
        if not (len(file_paths) == len(lengths) == len(ranks) == expected_candidates):
            raise ValueError(f"Query {query_id} does not contain eight candidates.")

        shared_length = min(min(lengths), TGT_LEN)
        if shared_length < 1:
            raise ValueError(f"Query {query_id} has no valid frames.")

        sequences = [
            self._prepare_sequence(file_path, shared_length) for file_path in file_paths
        ]
        return (
            torch.from_numpy(np.stack(sequences)).float(),
            torch.full((expected_candidates,), shared_length, dtype=torch.long),
            torch.tensor(ranks, dtype=torch.long),
            query_id,
        )


def make_listwise_loaders(
    train_items,
    validation_items,
    test_items,
    scaler,
    device,
    batch_size: int = BATCH_SIZE,
):
    """Training, validation, and test data loaders are created."""
    use_mps = device.type == "mps"
    loader_options = {
        "batch_size": batch_size,
        "num_workers": 0 if use_mps else 2,
        "pin_memory": not use_mps,
    }
    train_loader = DataLoader(
        QueryListDataset(train_items, scaler),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        QueryListDataset(validation_items, scaler),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        QueryListDataset(test_items, scaler),
        shuffle=False,
        **loader_options,
    )
    return train_loader, validation_loader, test_loader
