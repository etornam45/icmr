"""Shared classification and (inactive) boundary heads for WTAL."""

from __future__ import annotations

from torch import nn


class ClassificationHead(nn.Module):
    """MLP → C-way class activation sequence (CAS) logits."""

    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """x: [..., D] → [..., C]"""
        return self.net(x)


class BoundaryHead(nn.Module):
    """Stub MLP → (dist_to_start, dist_to_end). Inactive in WTAL v1."""

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.enabled = False
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        if not self.enabled:
            raise RuntimeError(
                "BoundaryHead is inactive in temporal_wtal_v1; "
                "enable only when span-labeled train data is available"
            )
        return self.net(x)
