"""1D temporal aggregator over per-frame CLS tokens."""

from __future__ import annotations

import torch
from torch import nn


class TemporalAggregator(nn.Module):
    """Residual 1D conv stack over [B, T, D] → [B, T, D]."""

    def __init__(
        self,
        dim: int = 384,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = hidden_dim or dim
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, hidden, kernel_size, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, dim, kernel_size, padding=padding),
                    nn.Dropout(dropout),
                )
            )
            in_dim = dim
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        h = x
        for block in self.layers:
            # Conv1d wants [B, D, T]
            residual = h
            y = block(h.transpose(1, 2)).transpose(1, 2)
            h = residual + y
        return self.norm(h)
