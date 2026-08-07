"""Shared DINOv3 runtime with DETR, anomaly, and optional caption heads."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dinov3.utils.device import get_device
from heads.anomaly.inference import (
    load_anomaly_label_maps,
    resolve_anomaly_checkpoint,
)
from heads.anomaly.model import (
    AnomalyClassifier,
    build_anomaly_model,
    load_anomaly_checkpoint,
)
from heads.backbone import build_backbone, encode_frames
from heads.caption.inference import resolve_caption_checkpoint
from heads.caption.model import (
    CaptionHead,
    build_caption_model,
    load_caption_checkpoint,
)
from heads.caption.tokenizer import DEFAULT_TOKENIZER_DIR, ensure_tokenizer
from heads.detr.transformer import build_detr
from server.config import ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelRuntime:
    device: torch.device
    backbone: nn.Module
    detr: nn.Module
    anomaly: AnomalyClassifier | None = None
    anomaly_id2label: dict[int, str] = field(default_factory=dict)
    caption: CaptionHead | None = None
    caption_tokenizer: Any = None
    caption_num_frames: int = 16
    config: ServerConfig | None = None

    @property
    def has_anomaly(self) -> bool:
        return self.anomaly is not None and bool(self.anomaly_id2label)

    @property
    def has_caption(self) -> bool:
        return self.caption is not None and self.caption_tokenizer is not None

    # Back-compat aliases used by older call sites.
    @property
    def vqa(self):
        return self.caption

    @property
    def vqa_tokenizer(self):
        return self.caption_tokenizer

    @property
    def vqa_num_frames(self) -> int:
        return self.caption_num_frames

    @property
    def has_vqa(self) -> bool:
        return self.has_caption


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


def _resolve_caption_dir(cfg: ServerConfig) -> Path | None:
    primary = resolve_caption_checkpoint(cfg.caption_checkpoint)
    if primary.exists():
        return primary
    fallback = resolve_caption_checkpoint(cfg.caption_fallback_checkpoint)
    if fallback.exists():
        logger.warning(
            "Caption checkpoint %s missing; using fallback %s",
            cfg.caption_checkpoint,
            fallback,
        )
        return fallback
    return None


def _peek_caption_num_frames(checkpoint_dir: Path, default: int) -> int:
    ckpt_path = checkpoint_dir / "caption_head.pt"
    if not ckpt_path.exists():
        return default
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        frames = state.get("num_frames")
        if frames is not None:
            return int(frames)
    except Exception:
        logger.exception("Could not read num_frames from %s", ckpt_path)
    return default


def _peek_tokenizer_dir(checkpoint_dir: Path, cfg: ServerConfig) -> str:
    ckpt_path = checkpoint_dir / "caption_head.pt"
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            tok_dir = state.get("tokenizer_dir")
            if tok_dir:
                return str(tok_dir)
        except Exception:
            logger.exception("Could not read tokenizer_dir from %s", ckpt_path)
    return cfg.caption_tokenizer_dir or DEFAULT_TOKENIZER_DIR


def load_runtime(cfg: ServerConfig) -> ModelRuntime:
    device = get_device()
    logger.info("Loading shared DINOv3 backbone on %s", device)
    backbone = build_backbone(device, weights=cfg.backbone_weights)

    logger.info("Loading DETR decoder from %s", cfg.detr_checkpoint)
    detr = _load_detr(cfg.detr_checkpoint, device)

    anomaly: AnomalyClassifier | None = None
    id2label: dict[int, str] = {}
    anomaly_path = resolve_anomaly_checkpoint(cfg.anomaly_checkpoint)
    if anomaly_path.exists():
        try:
            label2id, id2label = load_anomaly_label_maps(anomaly_path)
            anomaly = build_anomaly_model(
                device,
                num_classes=len(label2id),
                num_frames=cfg.num_frames,
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
            "auto captions are disabled until it exists",
            cfg.anomaly_checkpoint,
        )

    caption: CaptionHead | None = None
    tokenizer = None
    caption_num_frames = cfg.num_frames
    if cfg.load_caption:
        caption_dir = _resolve_caption_dir(cfg)
        if caption_dir is not None:
            try:
                caption_num_frames = _peek_caption_num_frames(
                    caption_dir, cfg.num_frames
                )
                tok_dir = _peek_tokenizer_dir(caption_dir, cfg)
                logger.info(
                    "Loading caption head from %s (num_frames=%d)",
                    caption_dir,
                    caption_num_frames,
                )
                tokenizer = ensure_tokenizer(tok_dir)
                caption = build_caption_model(
                    device,
                    vocab_size=tokenizer.vocab_size,
                    num_frames=caption_num_frames,
                    pad_token_id=tokenizer.pad_token_id,
                )
                load_caption_checkpoint(caption, caption_dir, device)
                caption.eval()
                logger.info("Caption head ready")
            except Exception:
                logger.exception(
                    "Failed to load caption head; continuing without captions"
                )
                caption = None
                tokenizer = None
                caption_num_frames = cfg.num_frames
        else:
            logger.warning("No caption checkpoint found; captions disabled")

    return ModelRuntime(
        device=device,
        backbone=backbone,
        detr=detr,
        anomaly=anomaly,
        anomaly_id2label=id2label,
        caption=caption,
        caption_tokenizer=tokenizer,
        caption_num_frames=caption_num_frames,
        config=cfg,
    )


@torch.no_grad()
def encode_video_window(
    runtime: ModelRuntime,
    videos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Single shared backbone forward over a video window."""
    return encode_frames(runtime.backbone, videos)
