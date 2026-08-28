"""The temporal Transformer clip scorer is defined here."""

import math

import torch
from torch import nn

from config import (
    TRANSFORMER_DIM_FF,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions are added to frame embeddings."""

    def __init__(self, dimension: int, maximum_length: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        rates = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(maximum_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * rates)
        encoding[:, 1::2] = torch.cos(positions * rates)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.encoding[:, : sequence.shape[1]]


class TemporalTransformerScorer(nn.Module):
    """A masked attention score is produced from a Transformer encoder."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FF,
        dropout: float = TRANSFORMER_DROPOUT,
    ) -> None:
        super().__init__()
        self.hparams = {
            "in_dim": int(in_dim),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
        }
        self.input_projection = nn.Linear(in_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            enable_nested_tensor=False,
        )
        self.attention = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Matrix parameters are initialised with Xavier uniform values."""
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """One scalar score is returned for each clip."""
        time_steps = sequence.shape[1]
        lengths = lengths.detach().long().clamp(1, time_steps).to(sequence.device)
        valid = torch.arange(time_steps, device=sequence.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        embedded = self.input_dropout(self.input_projection(sequence))
        encoded = self.encoder(
            self.position(embedded),
            src_key_padding_mask=~valid,
        )
        logits = self.attention(encoded).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)
