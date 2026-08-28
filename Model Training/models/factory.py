"""The selected model architecture is constructed here."""

from torch import nn

from config import MODEL_NAME
from .lstm import LSTMScorer
from .tcn import TCNScorer
from .transformer import TemporalTransformerScorer


MODEL_CLASSES = {
    "lstm": LSTMScorer,
    "tcn": TCNScorer,
    "transformer": TemporalTransformerScorer,
}


def build_model(in_dim: int) -> nn.Module:
    """One configured model is returned."""
    return MODEL_CLASSES[MODEL_NAME](in_dim)
