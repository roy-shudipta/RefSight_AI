"""The temporal convolutional clip scorer is defined here."""

import torch
from torch import nn

from config import TCN_CHANNELS, TCN_DROPOUT, TCN_KERNEL


class _Chomp1d(nn.Module):
    """Right-side convolution padding is removed to preserve causality."""

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = int(amount)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence[:, :, : -self.amount].contiguous()


class _TCNBlock(nn.Module):
    """Two causal convolutions and one residual connection are applied."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.layers = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=kernel,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.layers(sequence) + self.residual(sequence)


class TCNScorer(nn.Module):
    """A masked attention score is produced from four causal TCN blocks."""

    def __init__(
        self,
        in_dim: int,
        channels=TCN_CHANNELS,
        kernel: int = TCN_KERNEL,
        dropout: float = TCN_DROPOUT,
    ) -> None:
        super().__init__()
        self.hparams = {
            "in_dim": int(in_dim),
            "channels": [int(channel) for channel in channels],
            "kernel": int(kernel),
            "dilations": [2**index for index in range(len(channels))],
            "dropout": float(dropout),
        }
        blocks = []
        previous_channels = in_dim
        for index, output_channels in enumerate(channels):
            blocks.append(
                _TCNBlock(
                    previous_channels,
                    output_channels,
                    kernel,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            previous_channels = output_channels

        self.tcn = nn.Sequential(*blocks)
        self.attention = nn.Linear(previous_channels, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(previous_channels),
            nn.Linear(previous_channels, 1),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """One scalar score is returned for each clip."""
        time_steps = sequence.shape[1]
        lengths = lengths.detach().long().clamp(1, time_steps).to(sequence.device)
        encoded = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
        valid = torch.arange(time_steps, device=encoded.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        logits = self.attention(encoded).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)
