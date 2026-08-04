"""Shared DINOv3 runtime with DETR, anomaly, and optional VQA heads."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dinov3.checkpoints.load import ensure_backbone_checkpoint, load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.anomaly.inference import (
    load_anomaly_label_maps,
    resolve_anomaly_checkpoint,
)
from heads.anomaly.model import (
    DINOv3AnomalyClassifier,
    build_anomaly_model,
    load_anomaly_checkpoint,
)
from heads.detr.transformer import build_detr
from heads.vqa.inference import resolve_vqa_checkpoint
from heads.vqa.model import (
    DINOv3MiniCPMHybrid,
    build_hybrid_model,
    load_hybrid_checkpoint,
)
from server.config import ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelRuntime:
    device: torch.device
    backbone: nn.Module
    detr: nn.Module
    anomaly: DINOv3AnomalyClassifier | None = None
    anomaly_id2label: dict[int, str] = field(default_factory=dict)
    vqa: DINOv3MiniCPMHybrid | None = None
    vqa_tokenizer: Any = None
    vqa_num_frames: int = 16
    config: ServerConfig | None = None

    @property
    def has_anomaly(self) -> bool:
        return self.anomaly is not None and bool(self.anomaly_id2label)

    @property
    def has_vqa(self) -> bool:
        return self.vqa is not None and self.vqa_tokenizer is not None


def _build_backbone(weights: str, device: torch.device) -> nn.Module:
    vision = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    weights = ensure_backbone_checkpoint(weights)
    load_checkpoint(vision, weights)
    vision.to(device)
    vision.eval()
    for param in vision.parameters():
        param.requires_grad = False
    return vision


def _load_detr(checkpoint: str, device: torch.device) -> nn.Module:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"DETR checkpoint not found: {checkpoint}")
    detr = build_detr(
        d_model=384,
        num_layers=4,
        n_classes=92,
        n_points=5,
    ).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    detr.load_state_dict(state)
    detr.eval()
    return detr


def _resolve_vqa_dir(cfg: ServerConfig) -> Path | None:
    primary = resolve_vqa_checkpoint(cfg.vqa_checkpoint)
    if primary.exists():
        return primary
    fallback = resolve_vqa_checkpoint(cfg.vqa_fallback_checkpoint)
    if fallback.exists():
        logger.warning(
            "VQA checkpoint %s missing; using fallback %s",
            cfg.vqa_checkpoint,
            fallback,
        )
        return fallback
    return None


def _peek_vqa_num_frames(checkpoint_dir: Path, default: int) -> int:
    vision_path = checkpoint_dir / "vision_adapter.pt"
    if not vision_path.exists():
        return default
    try:
        state = torch.load(vision_path, map_location="cpu", weights_only=True)
        # Prefer num_frames only: num_visual_tokens is T * spatial_pool^2.
        frames = state.get("num_frames")
        if frames is not None:
            return int(frames)
    except Exception:
        logger.exception("Could not read num_frames from %s", vision_path)
    return default


def load_runtime(cfg: ServerConfig) -> ModelRuntime:
    device = get_device()
    logger.info("Loading shared DINOv3 backbone on %s", device)
    backbone = _build_backbone(cfg.backbone_weights, device)

    logger.info("Loading DETR decoder from %s", cfg.detr_checkpoint)
    detr = _load_detr(cfg.detr_checkpoint, device)

    anomaly: DINOv3AnomalyClassifier | None = None
    id2label: dict[int, str] = {}
    anomaly_path = resolve_anomaly_checkpoint(cfg.anomaly_checkpoint)
    if anomaly_path.exists():
        try:
            label2id, id2label = load_anomaly_label_maps(anomaly_path)
            anomaly = build_anomaly_model(
                device,
                num_classes=len(label2id),
                vision_model=backbone,
            )
            load_anomaly_checkpoint(anomaly, anomaly_path, device)
            anomaly.eval()
            logger.info(
                "Anomaly head loaded (%d classes) from %s",
                len(label2id),
                anomaly_path,
            )
        except Exception:
            logger.exception("Failed to load anomaly head; continuing without it")
            anomaly = None
            id2label = {}
    else:
        logger.warning(
            "Anomaly checkpoint not found at %s — live anomaly scores and "
            "auto VQA captions are disabled until it exists",
            cfg.anomaly_checkpoint,
        )

    vqa: DINOv3MiniCPMHybrid | None = None
    tokenizer = None
    vqa_num_frames = cfg.num_frames
    if cfg.load_vqa:
        vqa_dir = _resolve_vqa_dir(cfg)
        if vqa_dir is not None:
            try:
                vqa_num_frames = _peek_vqa_num_frames(vqa_dir, cfg.num_frames)
                logger.info(
                    "Loading VQA head from %s (num_frames=%d; this may take a while)",
                    vqa_dir,
                    vqa_num_frames,
                )
                vqa, tokenizer = build_hybrid_model(
                    device,
                    num_frames=vqa_num_frames,
                    vision_model=backbone,
                )
                load_hybrid_checkpoint(
                    vqa, str(vqa_dir), device, trainable_adapter=False
                )
                vqa.eval()
                logger.info("VQA head ready")
            except Exception:
                logger.exception("Failed to load VQA head; continuing without captions")
                vqa = None
                tokenizer = None
                vqa_num_frames = cfg.num_frames
        else:
            logger.warning("No VQA checkpoint found; captions disabled")

    return ModelRuntime(
        device=device,
        backbone=backbone,
        detr=detr,
        anomaly=anomaly,
        anomaly_id2label=id2label,
        vqa=vqa,
        vqa_tokenizer=tokenizer,
        vqa_num_frames=vqa_num_frames,
        config=cfg,
    )


@torch.no_grad()
def encode_video_window(
    runtime: ModelRuntime,
    videos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Single shared backbone forward over a video window.

    Args:
        videos: [B, T, 3, H, W] on runtime.device

    Returns:
        cls_tokens: [B, T, D]
        patch_tokens: [B, T, N, D]
        last_patches: [B, N, D]
    """
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    batch, num_frames, channels, height, width = videos.shape
    flat = videos.reshape(batch * num_frames, channels, height, width)
    features = runtime.backbone(flat, masks=None, is_training=True)
    cls = features["x_norm_clstoken"].view(batch, num_frames, -1)
    patches = features["x_norm_patchtokens"]
    n_patches = patches.shape[1]
    patches = patches.view(batch, num_frames, n_patches, -1)
    return {
        "cls_tokens": cls,
        "patch_tokens": patches,
        "last_patches": patches[:, -1],
    }
