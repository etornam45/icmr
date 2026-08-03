"""DINOv3 video anomaly classifier for VAU-Bench Anomaly Class."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dinov3.checkpoints.load import (
    ensure_backbone_checkpoint,
    load_checkpoint,
    validate_checkpoint_file,
)
from dinov3.models import vit_small

BACKBONE_WEIGHTS = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_CHECKPOINT_DIR = "dinov3/checkpoints/model/anomaly_vau"
VISION_DIM = 384
IMG_SIZE = 224
DEFAULT_NUM_FRAMES = 16


class DINOv3AnomalyClassifier(nn.Module):
    """Frozen DINOv3 CLS tokens → temporal mean-pool → linear classifier."""

    def __init__(
        self,
        vision_model: nn.Module,
        num_classes: int,
        vision_dim: int = VISION_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(vision_dim)
        self.head = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            videos: [B, T, 3, H, W] or [B, 3, H, W]

        Returns:
            pooled: [B, vision_dim]
        """
        if videos.dim() == 4:
            videos = videos.unsqueeze(1)
        if videos.dim() != 5:
            raise ValueError(
                f"Expected video tensor [B, T, 3, H, W], got shape {tuple(videos.shape)}"
            )

        batch_size, num_frames, channels, height, width = videos.shape
        flat = videos.reshape(batch_size * num_frames, channels, height, width)

        with torch.no_grad():
            features = self.vision_model(flat, masks=None, is_training=True)
            cls_tokens = features["x_norm_clstoken"]

        cls_tokens = cls_tokens.view(batch_size, num_frames, -1)
        return cls_tokens.mean(dim=1)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(videos)
        return self.head(self.norm(pooled))


def classifier_trainable_params(model: DINOv3AnomalyClassifier):
    return list(model.norm.parameters()) + list(model.head.parameters())


def build_anomaly_model(
    device: torch.device,
    num_classes: int,
    backbone_weights: str = BACKBONE_WEIGHTS,
    hidden_dim: int = 256,
) -> DINOv3AnomalyClassifier:
    vision_model = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    if Path(backbone_weights).exists() and not validate_checkpoint_file(
        backbone_weights, expected_sha256=None
    ):
        print(f"Warning: checkpoint at {backbone_weights} looks corrupt, re-downloading")
        Path(backbone_weights).unlink(missing_ok=True)

    backbone_weights = ensure_backbone_checkpoint(backbone_weights)
    load_checkpoint(vision_model, backbone_weights)
    vision_model.to(device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False

    model = DINOv3AnomalyClassifier(
        vision_model=vision_model,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    ).to(device)
    return model


def save_anomaly_checkpoint(
    model: DINOv3AnomalyClassifier,
    checkpoint_dir: str | Path,
    label2id: dict[str, int],
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "norm": model.norm.state_dict(),
            "head": model.head.state_dict(),
            "num_classes": model.num_classes,
            "label2id": label2id,
        },
        path / "classifier.pt",
    )
    # Also write a JSON map for easy inspection / inference.
    from heads.vqa.vau_dataset import save_label_maps

    save_label_maps(label2id, path / "label2id.json")


def load_anomaly_checkpoint(
    model: DINOv3AnomalyClassifier,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> dict[str, int]:
    path = Path(checkpoint_dir)
    ckpt_path = path / "classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No classifier checkpoint at {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(state["num_classes"])
    if num_classes != model.num_classes:
        raise RuntimeError(
            f"Checkpoint num_classes={num_classes} != model "
            f"num_classes={model.num_classes}"
        )
    model.norm.load_state_dict(state["norm"])
    model.head.load_state_dict(state["head"])
    label2id = {str(k): int(v) for k, v in state["label2id"].items()}
    return label2id
