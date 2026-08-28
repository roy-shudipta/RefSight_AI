"""The listwise training and evaluation loops are defined here."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import GRADIENT_CLIP_NORM, LISTWISE_TAU
from losses import listwise_softmax_loss


def _candidate_scores(
    model: nn.Module,
    sequences: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """All candidate clips are scored and restored to query form."""
    batch_size, candidate_count, time_steps, feature_count = sequences.shape
    flat_sequences = sequences.reshape(
        batch_size * candidate_count,
        time_steps,
        feature_count,
    )
    flat_lengths = lengths.reshape(batch_size * candidate_count)
    return model(flat_sequences, flat_lengths).reshape(batch_size, candidate_count)


def train_epoch_listwise(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loader: DataLoader,
) -> float:
    """One listwise training epoch is completed."""
    model.train()
    total_loss = 0.0
    query_count = 0

    for sequences, lengths, _, _ in loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        scores = _candidate_scores(model, sequences, lengths)
        loss = listwise_softmax_loss(scores, tau=LISTWISE_TAU)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        batch_size = sequences.shape[0]
        total_loss += loss.item() * batch_size
        query_count += batch_size

    return total_loss / max(query_count, 1)


@torch.no_grad()
def eval_listwise(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
) -> tuple[float, float, float, float]:
    """Loss, Top 1, Top 2, and mean reciprocal rank are calculated."""
    model.eval()
    total_loss = 0.0
    query_count = 0
    top1_count = 0
    top2_count = 0
    reciprocal_ranks: list[float] = []

    for sequences, lengths, _, _ in loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        scores = _candidate_scores(model, sequences, lengths)
        loss = listwise_softmax_loss(scores, tau=LISTWISE_TAU)

        batch_size = sequences.shape[0]
        top1 = torch.argmax(scores, dim=1)
        top2 = torch.topk(scores, k=2, dim=1).indices
        top1_count += int(top1.eq(0).sum().item())
        top2_count += int(top2.eq(0).any(dim=1).sum().item())

        ordered = torch.argsort(scores, dim=1, descending=True)
        positions = ordered.eq(0).nonzero(as_tuple=False)[:, 1] + 1
        reciprocal_ranks.extend((1.0 / positions.float()).cpu().tolist())

        total_loss += loss.item() * batch_size
        query_count += batch_size

    return (
        total_loss / max(query_count, 1),
        top1_count / max(query_count, 1),
        top2_count / max(query_count, 1),
        float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
    )
