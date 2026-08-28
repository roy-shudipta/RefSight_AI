"""The listwise training objective is defined here."""

import torch
import torch.nn.functional as functional


def listwise_softmax_loss(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """Cross-entropy is calculated with the incorrect clip at index zero."""
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("A two-dimensional candidate score tensor is required.")
    targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return functional.cross_entropy(scores / float(tau), targets)
