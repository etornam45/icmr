"""Perceiver-style resampler: compress spatiotemporal DINOv3 tokens to K latents."""

from __future__ import annotations

import torch
from torch import nn


def build_2d_sincos_pos_embed(height: int, width: int, dim: int) -> torch.Tensor:
    """Sinusoidal 2D spatial position encoding: (1, H*W, dim)."""
    assert dim % 4 == 0, "pos embed dim must be divisible by 4 for 2D"
    half = dim // 2
    grid_y = torch.arange(height, dtype=torch.float32)
    grid_x = torch.arange(width, dtype=torch.float32)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000**omega)

    def _encode(coord: torch.Tensor) -> torch.Tensor:
        flat = coord.reshape(-1, 1)
        angles = flat * omega.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    pos = torch.cat([_encode(gy), _encode(gx)], dim=1)
    return pos.unsqueeze(0)


class _PerceiverLayer(nn.Module):
    """One resampler layer: cross-attn (queries→media) → self-attn → FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.cross_ln_q = nn.LayerNorm(dim)
        self.cross_ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.0
        )
        self.self_ln = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.0
        )
        self.ffn_ln = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, latents: torch.Tensor, media: torch.Tensor) -> torch.Tensor:
        q = self.cross_ln_q(latents)
        kv = self.cross_ln_kv(media)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latents = latents + cross_out

        s = self.self_ln(latents)
        self_out, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + self_out

        latents = latents + self.ffn(self.ffn_ln(latents))
        return latents


class PerceiverResampler(nn.Module):
    """Compress variable-length visual tokens to a fixed set of latents."""

    def __init__(
        self,
        vision_dim: int = 384,
        llm_dim: int = 1536,
        num_latents: int = 64,
        depth: int = 3,
        num_heads: int = 8,
        max_frames: int = 32,
        grid_size: int = 14,
        include_cls: bool = True,
    ):
        super().__init__()
        if llm_dim % num_heads != 0:
            raise ValueError(f"llm_dim={llm_dim} must be divisible by num_heads={num_heads}")
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.num_latents = num_latents
        self.max_frames = max_frames
        self.grid_size = grid_size
        self.include_cls = include_cls

        self.input_proj = nn.Linear(vision_dim, llm_dim)
        self.temporal_embed = nn.Embedding(max_frames, llm_dim)
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

        spatial = build_2d_sincos_pos_embed(grid_size, grid_size, llm_dim)
        self.register_buffer("_spatial_pos", spatial, persistent=False)

        # CLS slot gets a zero spatial PE (index handled separately).
        self.cls_spatial = nn.Parameter(torch.zeros(1, 1, llm_dim))

        scale = llm_dim**-0.5
        self.latents = nn.Parameter(scale * torch.randn(num_latents, llm_dim))
        self.layers = nn.ModuleList(
            [_PerceiverLayer(llm_dim, num_heads) for _ in range(depth)]
        )
        self.out_ln = nn.LayerNorm(llm_dim)

    def _build_media(
        self,
        cls_tokens: torch.Tensor | None,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_tokens: [B, T, vision_dim] or None
            patch_tokens: [B, T, N, vision_dim] with N = grid_size^2

        Returns:
            media: [B, T*(1+N) or T*N, llm_dim]
        """
        batch, num_frames, num_patches, dim = patch_tokens.shape
        if num_patches != self.grid_size * self.grid_size:
            raise ValueError(
                f"Expected {self.grid_size * self.grid_size} patches, got {num_patches}"
            )
        if num_frames > self.max_frames:
            raise ValueError(
                f"num_frames={num_frames} exceeds max_frames={self.max_frames}"
            )

        patches = self.input_proj(patch_tokens)
        # patches: [B, T, N, D]
        spatial = self._spatial_pos.to(device=patches.device, dtype=patches.dtype)
        patches = patches + spatial.unsqueeze(1)

        frame_ids = torch.arange(num_frames, device=patches.device)
        temporal = self.temporal_embed(frame_ids).to(dtype=patches.dtype)
        # [T, D] → [1, T, 1, D]
        patches = patches + temporal.view(1, num_frames, 1, -1)

        if self.include_cls and cls_tokens is not None:
            cls = self.input_proj(cls_tokens)  # [B, T, D]
            cls = cls + temporal.unsqueeze(0)
            cls = cls + self.cls_spatial.to(dtype=cls.dtype)
            cls = cls.unsqueeze(2)  # [B, T, 1, D]
            media = torch.cat([cls, patches], dim=2)
        else:
            media = patches

        return media.reshape(batch, -1, self.llm_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, T, N, vision_dim]
            cls_tokens: optional [B, T, vision_dim]

        Returns:
            latents: [B, num_latents, llm_dim]
        """
        media = self._build_media(cls_tokens, patch_tokens)
        batch = media.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents = latents.to(dtype=media.dtype)

        for layer in self.layers:
            latents = layer(latents, media)
        return self.out_ln(latents)
