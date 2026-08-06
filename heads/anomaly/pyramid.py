"""1D temporal feature pyramid (ActionFormer-style strides)."""

from __future__ import annotations

import torch
from torch import nn


class FeaturePyramid1D(nn.Module):
    """Build multi-scale temporal features with strides 1, 2, 4, 8."""

    def __init__(
        self,
        dim: int = 384,
        strides: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        if strides[0] != 1:
            raise ValueError("First pyramid stride must be 1")
        self.strides = tuple(strides)
        # Downsample between consecutive levels (stride ratio).
        downs: list[nn.Module] = []
        for i in range(1, len(strides)):
            ratio = strides[i] // strides[i - 1]
            if ratio < 2:
                raise ValueError(f"Non-increasing strides: {strides}")
            downs.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=3, stride=ratio, padding=1),
                    nn.GELU(),
                )
            )
        self.downs = nn.ModuleList(downs)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: [B, T, D] stride-1 features

        Returns:
            list of [B, T_s, D] for each stride in self.strides
        """
        levels = [x]
        h = x.transpose(1, 2)  # [B, D, T]
        for down in self.downs:
            h = down(h)
            levels.append(h.transpose(1, 2))
        return levels
