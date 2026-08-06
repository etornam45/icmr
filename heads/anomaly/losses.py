"""Weakly-supervised losses: top-k MIL, binary MIL ranking, SVDD."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def topk_mil_loss(
    cas_logits: torch.Tensor,
    labels: torch.Tensor,
    k_ratio: float = 1.0 / 8.0,
) -> torch.Tensor:
    """STPN / W-TALC style top-k pooling CE on video-level labels.

    Args:
        cas_logits: [B, T, C] stride-1 class activation logits
        labels: [B] class indices
        k_ratio: fraction of timesteps to pool (default ceil(T/8))
    """
    if cas_logits.dim() != 3:
        raise ValueError(f"Expected cas_logits [B,T,C], got {tuple(cas_logits.shape)}")
    batch, num_steps, num_classes = cas_logits.shape
    k = max(1, int(math.ceil(num_steps * k_ratio)))
    # top-k over time for every class → [B, C, k]
    topk_vals, _ = torch.topk(cas_logits.transpose(1, 2), k=k, dim=-1)
    video_logits = topk_vals.mean(dim=-1)  # [B, C]
    return F.cross_entropy(video_logits, labels)


def mil_ranking_loss(
    actionness: torch.Tensor,
    is_anomaly: torch.Tensor,
    lambda_smooth: float = 8e-5,
    lambda_sparse: float = 8e-5,
) -> torch.Tensor:
    """Sultani-style MIL ranking on binary anomaly scores.

    Args:
        actionness: [B, T] = 1 - p(normal)
        is_anomaly: [B] bool / 0-1 — True for anomalous bags
    """
    if actionness.dim() != 2:
        raise ValueError(f"Expected actionness [B,T], got {tuple(actionness.shape)}")
    anom_mask = is_anomaly.bool()
    norm_mask = ~anom_mask
    if not anom_mask.any() or not norm_mask.any():
        return actionness.new_zeros(())

    max_a = actionness[anom_mask].amax(dim=-1)  # [Na]
    max_n = actionness[norm_mask].amax(dim=-1)  # [Nn]

    # Pair each anomalous bag with a normal bag (cycle if uneven).
    n_pairs = max(max_a.numel(), max_n.numel())
    a_idx = torch.arange(n_pairs, device=actionness.device) % max_a.numel()
    n_idx = torch.arange(n_pairs, device=actionness.device) % max_n.numel()
    rank = F.relu(1.0 - max_a[a_idx] + max_n[n_idx]).mean()

    a_scores = actionness[anom_mask]
    smooth = ((a_scores[:, 1:] - a_scores[:, :-1]) ** 2).sum(dim=-1).mean()
    sparse = a_scores.sum(dim=-1).mean()
    return rank + lambda_smooth * smooth + lambda_sparse * sparse


class SVDDRegularizer:
    """Deep SVDD-style EMA center over confirmed-normal embeddings."""

    def __init__(self, dim: int, momentum: float = 0.9):
        self.momentum = momentum
        self.register_buffer_owner: torch.nn.Module | None = None
        self._center: torch.Tensor | None = None
        self.dim = dim
        self._initialized = False

    def attach(self, module: torch.nn.Module) -> None:
        """Register center as a non-trainable buffer on ``module``."""
        self.register_buffer_owner = module
        if not hasattr(module, "svdd_center"):
            module.register_buffer(
                "svdd_center",
                torch.zeros(self.dim),
                persistent=True,
            )
            module.register_buffer(
                "svdd_initialized",
                torch.tensor(0, dtype=torch.long),
                persistent=True,
            )

    @property
    def center(self) -> torch.Tensor:
        if self.register_buffer_owner is None:
            raise RuntimeError("SVDDRegularizer.attach(model) was not called")
        return self.register_buffer_owner.svdd_center  # type: ignore[attr-defined]

    def update_center(self, embeddings: torch.Tensor) -> None:
        """EMA-update center from [N, D] or [B, T, D] normal embeddings."""
        if embeddings.numel() == 0:
            return
        flat = embeddings.reshape(-1, embeddings.shape[-1]).detach()
        batch_mean = flat.mean(dim=0)
        owner = self.register_buffer_owner
        assert owner is not None
        if int(owner.svdd_initialized.item()) == 0:  # type: ignore[attr-defined]
            owner.svdd_center.copy_(batch_mean)  # type: ignore[attr-defined]
            owner.svdd_initialized.fill_(1)  # type: ignore[attr-defined]
        else:
            owner.svdd_center.mul_(self.momentum).add_(  # type: ignore[attr-defined]
                batch_mean, alpha=1.0 - self.momentum
            )

    def loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Mean squared distance to center over [..., D]."""
        owner = self.register_buffer_owner
        assert owner is not None
        if int(owner.svdd_initialized.item()) == 0:  # type: ignore[attr-defined]
            return embeddings.new_zeros(())
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        return ((flat - self.center.detach()) ** 2).sum(dim=-1).mean()

    def distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Per-timestep L2 distance to center. embeddings [B, T, D] → [B, T]."""
        owner = self.register_buffer_owner
        assert owner is not None
        if int(owner.svdd_initialized.item()) == 0:  # type: ignore[attr-defined]
            return embeddings.new_zeros(embeddings.shape[:2])
        return torch.linalg.norm(embeddings - self.center.detach(), dim=-1)
