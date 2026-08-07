"""Shared frozen DINOv3 backbone: build once, encode frames, hand tokens to heads."""

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
VISION_DIM = 384
IMG_SIZE = 224
PATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(videos: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std to video/image tensors in [0, 1]."""
    if videos.dim() == 5:
        shape = (1, 1, 3, 1, 1)
    elif videos.dim() == 4:
        shape = (1, 3, 1, 1)
    elif videos.dim() == 3:
        shape = (3, 1, 1)
    else:
        raise ValueError(f"Expected 3D/4D/5D tensor, got shape {tuple(videos.shape)}")
    mean = videos.new_tensor(IMAGENET_MEAN).view(*shape)
    std = videos.new_tensor(IMAGENET_STD).view(*shape)
    return (videos - mean) / std


def build_backbone(
    device: torch.device,
    weights: str = BACKBONE_WEIGHTS,
    auto_download: bool = True,
) -> nn.Module:
    """Load frozen DINOv3 ViT-S/16 and move it to ``device``."""
    vision = vit_small(
        patch_size=PATCH_SIZE,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    if Path(weights).exists() and not validate_checkpoint_file(
        weights, expected_sha256=None
    ):
        print(f"Warning: checkpoint at {weights} looks corrupt, re-downloading")
        Path(weights).unlink(missing_ok=True)

    weights = ensure_backbone_checkpoint(weights, auto_download=auto_download)
    load_checkpoint(vision, weights)
    vision.to(device)
    vision.eval()
    for param in vision.parameters():
        param.requires_grad = False
    return vision


@torch.no_grad()
def encode_frames(
    backbone: nn.Module,
    videos: torch.Tensor,
    *,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Run the shared backbone over sampled frames.

    Args:
        backbone: frozen DINOv3 module
        videos: [B, T, 3, H, W] or [B, 3, H, W] or [T, 3, H, W] in [0, 1]
        normalize: apply ImageNet mean/std (default True)

    Returns:
        cls_tokens: [B, T, D]
        patch_tokens: [B, T, N, D]
        last_patches: [B, N, D]
    """
    if videos.dim() == 4:
        videos = videos.unsqueeze(1)  # [B, 3, H, W] → single-frame video
    if videos.dim() != 5:
        raise ValueError(
            f"Expected video tensor [B, T, 3, H, W], got shape {tuple(videos.shape)}"
        )

    if normalize:
        videos = imagenet_normalize(videos)

    batch, num_frames, channels, height, width = videos.shape
    if height != width:
        raise ValueError(f"Expected square frames, got HxW={height}x{width}")
    if height % PATCH_SIZE != 0:
        raise ValueError(
            f"Frame size {height} must be divisible by patch size {PATCH_SIZE}"
        )

    flat = videos.reshape(batch * num_frames, channels, height, width)
    features = backbone(flat, masks=None, is_training=True)
    cls = features["x_norm_clstoken"].view(batch, num_frames, -1)
    patches = features["x_norm_patchtokens"]
    n_patches = patches.shape[1]
    expected = (height // PATCH_SIZE) ** 2
    if n_patches != expected:
        raise ValueError(f"Expected {expected} patch tokens, got {n_patches}")
    patches = patches.view(batch, num_frames, n_patches, -1)
    return {
        "cls_tokens": cls,
        "patch_tokens": patches,
        "last_patches": patches[:, -1],
    }
