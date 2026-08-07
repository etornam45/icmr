"""Video caption head over shared DINOv3 patch tokens.

The backbone in ``heads.backbone`` encodes frames; this module consumes
``patch_tokens`` and runs a Transformer decoder for caption logits.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from heads.backbone import IMG_SIZE, PATCH_SIZE, VISION_DIM

DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/caption"
DEFAULT_NUM_FRAMES = 16
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
PATCHES_PER_FRAME = (IMG_SIZE // PATCH_SIZE) ** 2
ARCHITECTURE = "caption_decoder_v1"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first sequences (B, L, D)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class CaptionHead(nn.Module):
    """Decoder-only captioner conditioned on spatiotemporal patch tokens.

    Visual path: project patches → spatial + temporal PE → flatten to memory.
    Text path: embed token ids → causal TransformerDecoder → LM logits.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        vision_dim: int = VISION_DIM,
        num_frames: int = DEFAULT_NUM_FRAMES,
        patches_per_frame: int = PATCHES_PER_FRAME,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_heads: int = DEFAULT_NUM_HEADS,
        vocab_size: int = 32000,
        max_text_len: int = 128,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.architecture = ARCHITECTURE
        self.d_model = d_model
        self.vision_dim = vision_dim
        self.num_frames = num_frames
        self.patches_per_frame = patches_per_frame
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.pad_token_id = pad_token_id

        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.spatial_embed = PositionalEncoding(d_model, max_len=patches_per_frame)
        self.temporal_embed = PositionalEncoding(d_model, max_len=num_frames)

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.text_pos = PositionalEncoding(d_model, max_len=max_text_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=mlp_ratio * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embed.weight[self.pad_token_id].zero_()
        nn.init.xavier_uniform_(self.vision_proj.weight)
        nn.init.zeros_(self.vision_proj.bias)
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def encode_memory(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, T, P, vision_dim)
        Returns:
            memory: (B, T*P, d_model)
        """
        b, t, p, _ = patch_tokens.shape
        if t != self.num_frames:
            raise ValueError(f"expected {self.num_frames} frames, got {t}")
        if p != self.patches_per_frame:
            raise ValueError(f"expected {self.patches_per_frame} patches/frame, got {p}")

        x = self.vision_proj(patch_tokens)  # (B, T, P, D)
        x = self.spatial_embed(x.reshape(b * t, p, self.d_model)).view(b, t, p, self.d_model)
        x = x + self.temporal_embed.encoding[:, :t].unsqueeze(2)
        return x.view(b, t * p, self.d_model)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, T, P, vision_dim) from ``encode_frames``
            input_ids: (B, L) caption token ids (teacher-forced)
            attention_mask: optional (B, L) — 1 = keep, 0 = pad

        Returns:
            logits: (B, L, vocab_size)
        """
        memory = self.encode_memory(patch_tokens)

        tgt = self.token_embed(input_ids) * math.sqrt(self.d_model)
        tgt = self.text_pos(tgt)

        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask == 0

        hidden = self.transformer(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        patch_tokens: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedy decode. Returns token ids (B, 1 + generated)."""
        memory = self.encode_memory(patch_tokens)
        batch = patch_tokens.size(0)
        device = patch_tokens.device
        tokens = torch.full(
            (batch, 1), bos_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_new_tokens):
            tgt = self.token_embed(tokens) * math.sqrt(self.d_model)
            tgt = self.text_pos(tgt)
            causal_mask = torch.triu(
                torch.ones(
                    tokens.size(1),
                    tokens.size(1),
                    device=device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            hidden = self.transformer(tgt, memory, tgt_mask=causal_mask)
            next_id = self.lm_head(hidden[:, -1:]).argmax(dim=-1)
            tokens = torch.cat([tokens, next_id], dim=1)
            if (next_id == eos_token_id).all():
                break
        return tokens


def build_caption_model(
    device: torch.device,
    vocab_size: int,
    num_frames: int = DEFAULT_NUM_FRAMES,
    d_model: int = DEFAULT_D_MODEL,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    pad_token_id: int = 0,
) -> CaptionHead:
    """Build the caption head only (no backbone)."""
    return CaptionHead(
        d_model=d_model,
        vision_dim=VISION_DIM,
        num_frames=num_frames,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
    ).to(device)


if __name__ == "__main__":
    model = CaptionHead(
        d_model=DEFAULT_D_MODEL,
        num_frames=DEFAULT_NUM_FRAMES,
        vocab_size=1000,
        num_layers=8,
        num_heads=8,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Caption Head parameters: {n_params / 1e6:.1f}M")

    patches = torch.randn(
        2, DEFAULT_NUM_FRAMES, PATCHES_PER_FRAME, VISION_DIM
    )
    input_ids = torch.randint(1, 1000, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    logits = model(patches, input_ids, attention_mask=attn)
    assert logits.shape == (2, 16, 1000), logits.shape

    gen = model.generate(patches, bos_token_id=1, eos_token_id=2, max_new_tokens=8)
    assert gen.shape[0] == 2 and gen.shape[1] >= 2
    print(f"ok: logits={tuple(logits.shape)} gen={tuple(gen.shape)}")
