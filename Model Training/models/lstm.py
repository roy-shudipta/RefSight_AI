"""The bidirectional LSTM clip scorer is defined here."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import BIDIR, DROPOUT, HIDDEN, LAYERS


class LSTMScorer(nn.Module):
    """A masked attention score is produced from bidirectional LSTM outputs."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = HIDDEN,
        layers: int = LAYERS,
        bidir: bool = BIDIR,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.hparams = {
            "in_dim": int(in_dim),
            "hidden": int(hidden),
            "layers": int(layers),
            "bidir": bool(bidir),
            "dropout": float(dropout),
        }
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout if layers > 1 else 0.0,
        )
        output_dim = hidden * (2 if bidir else 1)
        self.attention = nn.Linear(output_dim, 1)
        self.head = nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, 1))

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """One scalar score is returned for each clip."""
        time_steps = sequence.shape[1]
        lengths_cpu = lengths.detach().cpu().long().clamp(1, time_steps)
        packed = pack_padded_sequence(
            sequence,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=time_steps,
        )

        valid = torch.arange(time_steps, device=output.device).unsqueeze(
            0
        ) < lengths_cpu.to(output.device).unsqueeze(1)
        logits = self.attention(output).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = (output * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)
