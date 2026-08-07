"""DINOv3 patch-token video anomaly classifier (head only).

The shared backbone in ``heads.backbone`` encodes frames; this module
consumes ``patch_tokens`` and predicts a video-level class.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from heads.backbone import IMG_SIZE, PATCH_SIZE, VISION_DIM
from heads.vqa.vau_dataset import save_label_maps

ARCHITECTURE = "patch_transformer_v1"
DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/anomaly_vau"
DEFAULT_NUM_FRAMES = 16
PATCHES_PER_FRAME = (IMG_SIZE // PATCH_SIZE) ** 2


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first sequences (B, L, D)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class AnomalyClassifier(nn.Module):
    """Transformer over DINOv3 patch tokens → video-level class logits."""

    def __init__(
        self,
        vision_dim: int = VISION_DIM,
        num_classes: int = 12,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_layers: int = 2,
        n_heads: int = 8,
        patches_per_frame: int = PATCHES_PER_FRAME,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.architecture = ARCHITECTURE
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.patches_per_frame = patches_per_frame
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        self.spatial_embed = PositionalEncoding(vision_dim, max_len=patches_per_frame)
        self.temporal_embed = PositionalEncoding(vision_dim, max_len=num_frames)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=n_heads,
            dim_feedforward=vision_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, T, P, D) from ``heads.backbone.encode_frames``
        Returns:
            logits: (B, C)
        """
        b, t, p, d = patch_tokens.shape
        if t != self.num_frames:
            raise ValueError(f"expected {self.num_frames} frames, got {t}")
        if p != self.patches_per_frame:
            raise ValueError(f"expected {self.patches_per_frame} patches/frame, got {p}")

        x = patch_tokens.reshape(b * t, p, d)
        x = self.spatial_embed(x).view(b, t, p, d)
        x = x + self.temporal_embed.encoding[:, :t].unsqueeze(2)
        x = self.transformer(x.view(b, t * p, d))
        return self.cls_head(x.mean(dim=1))


# Back-compat aliases.
DINOv3AnomalyClassifier = AnomalyClassifier
DINOv3AnomalyLocalizer = AnomalyClassifier


def classifier_trainable_params(model: AnomalyClassifier):
    return list(model.parameters())


localizer_trainable_params = classifier_trainable_params


def build_anomaly_model(
    device: torch.device,
    num_classes: int,
    num_frames: int = DEFAULT_NUM_FRAMES,
    hidden_dim: int = 256,
    num_layers: int = 2,
    n_heads: int = 8,
    **_legacy_kwargs,
) -> AnomalyClassifier:
    """Build the anomaly head only (no backbone)."""
    del _legacy_kwargs
    return AnomalyClassifier(
        vision_dim=VISION_DIM,
        num_classes=num_classes,
        num_frames=num_frames,
        num_layers=num_layers,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
    ).to(device)


def save_anomaly_checkpoint(
    model: AnomalyClassifier,
    checkpoint_dir: str | Path,
    label2id: dict[str, int],
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "classifier": model.state_dict(),
            "num_classes": model.num_classes,
            "num_frames": model.num_frames,
            "hidden_dim": model.hidden_dim,
            "label2id": label2id,
        },
        path / "classifier.pt",
    )
    save_label_maps(label2id, path / "label2id.json")


def load_anomaly_checkpoint(
    model: AnomalyClassifier,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> dict[str, int]:
    path = Path(checkpoint_dir)
    ckpt_path = path / "classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No classifier checkpoint at {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = state.get("architecture")
    if arch is not None and arch != ARCHITECTURE:
        raise RuntimeError(
            f"Checkpoint architecture={arch!r} is incompatible with "
            f"{ARCHITECTURE!r}. Retrain with heads.anomaly.train."
        )
    if "classifier" not in state:
        raise RuntimeError(
            "Legacy WTAL / mean-pool anomaly checkpoint detected. "
            f"Retrain for {ARCHITECTURE}."
        )

    num_classes = int(state["num_classes"])
    if num_classes != model.num_classes:
        raise RuntimeError(
            f"Checkpoint num_classes={num_classes} != model "
            f"num_classes={model.num_classes}"
        )
    ckpt_frames = state.get("num_frames")
    if ckpt_frames is not None and int(ckpt_frames) != model.num_frames:
        raise RuntimeError(
            f"Checkpoint num_frames={ckpt_frames} != model "
            f"num_frames={model.num_frames}"
        )

    model.load_state_dict(state["classifier"])
    label2id = {str(k): int(v) for k, v in state["label2id"].items()}
    return label2id


if __name__ == "__main__":
    model = AnomalyClassifier(num_classes=12, num_frames=DEFAULT_NUM_FRAMES)
    patches = torch.randn(2, DEFAULT_NUM_FRAMES, PATCHES_PER_FRAME, VISION_DIM)
    out = model(patches)
    assert out.shape == (2, 12), out.shape
    print(f"ok: {tuple(out.shape)}")
