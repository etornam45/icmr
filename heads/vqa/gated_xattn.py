"""Flamingo-style tanh-gated cross-attention blocks for splicing into an LM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from heads.vqa.llm_loader import get_decoder_layers


class VisualFeatureHolder:
    """Shared visual token buffer read by wrapped decoder layers each forward."""

    def __init__(self) -> None:
        self.features: torch.Tensor | None = None

    def set(self, features: torch.Tensor | None) -> None:
        self.features = features

    def clear(self) -> None:
        self.features = None


class GatedCrossAttentionBlock(nn.Module):
    """
    x = x + tanh(g_attn) * XAttn(LN(x), visual)
    x = x + tanh(g_ffn)  * FFN(LN(x))

    Gates init at 0 so the LM forward is unchanged at step 0.
    """

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_visual = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.0
        )
        self.gate_attn = nn.Parameter(torch.tensor(0.0))

        self.ln_ffn = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gate_ffn = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # Align hidden states and visual tokens with block weights (bf16 LM vs fp32 init).
        compute_dtype = self.ln_attn.weight.dtype
        x = x.to(dtype=compute_dtype)
        visual = visual.to(dtype=compute_dtype)
        q = self.ln_attn(x)
        kv = self.ln_visual(visual)
        # Broadcast visual batch if needed (shouldn't happen in practice).
        if kv.shape[0] == 1 and q.shape[0] > 1:
            kv = kv.expand(q.shape[0], -1, -1)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + torch.tanh(self.gate_attn).to(dtype=compute_dtype) * attn_out
        x = x + torch.tanh(self.gate_ffn).to(dtype=compute_dtype) * self.ffn(
            self.ln_ffn(x)
        )
        return x


class XAttnWrappedDecoderLayer(nn.Module):
    """Apply gated cross-attn then the original HF decoder layer."""

    def __init__(
        self,
        original_layer: nn.Module,
        xattn: GatedCrossAttentionBlock,
        holder: VisualFeatureHolder,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.xattn = xattn
        self.holder = holder

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        visual = self.holder.features
        if visual is not None:
            hidden_states = self.xattn(hidden_states, visual)
        return self.original_layer(hidden_states, *args, **kwargs)


def build_gated_blocks(
    num_blocks: int,
    dim: int,
    num_heads: int,
    ffn_mult: int = 4,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            GatedCrossAttentionBlock(dim, num_heads, ffn_mult=ffn_mult)
            for _ in range(num_blocks)
        ]
    )


def wrap_decoder_layers(
    llm: nn.Module,
    blocks: nn.ModuleList,
    holder: VisualFeatureHolder,
    every_n: int = 4,
) -> list[int]:
    """
    Replace every N-th decoder layer with an XAttn-wrapped version.

    Returns the list of layer indices that were wrapped.
    """
    layers = get_decoder_layers(llm)
    num_layers = len(layers)
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")

    # Place blocks on layers every_n-1, 2*every_n-1, ... (same as Flamingo every-nth)
    indices = list(range(every_n - 1, num_layers, every_n))
    if len(indices) != len(blocks):
        raise ValueError(
            f"Expected {len(blocks)} xattn blocks for every_n={every_n} "
            f"across {num_layers} layers, got indices {indices}"
        )

    for block, idx in zip(blocks, indices):
        original = layers[idx]
        # Avoid double-wrapping if called twice.
        if isinstance(original, XAttnWrappedDecoderLayer):
            original.xattn = block
            original.holder = holder
        else:
            layers[idx] = XAttnWrappedDecoderLayer(original, block, holder)
    return indices


def count_xattn_slots(num_layers: int, every_n: int) -> int:
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")
    return len(range(every_n - 1, num_layers, every_n))
