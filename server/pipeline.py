"""Inference pipeline: preprocess frames → shared DINO → heads → overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from heads.anomaly.inference import predict_from_cls
from heads.detr.dataset import letterbox
from heads.detr.predict import detect_from_patches
from server.config import ServerConfig
from server.overlays import render_all_previews
from server.runtime import ModelRuntime, encode_video_window


@dataclass
class PipelineResult:
    jpegs: dict[str, bytes]
    anomaly_class: str | None
    anomaly_score: float | None
    top_k: list[dict[str, Any]]
    detections: list[dict[str, Any]]
    is_anomaly: bool
    videos_tensor: torch.Tensor | None
    preview_bgr: np.ndarray


def bgr_frames_to_tensor(
    frames_bgr: list[np.ndarray],
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Letterbox + normalize BGR frames → [1, T, 3, H, W] float tensor."""
    tensors: list[torch.Tensor] = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image, _, _, _ = letterbox(Image.fromarray(rgb), img_size)
        array = np.array(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    stacked = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    return stacked


def is_anomaly_event(
    prediction: str | None,
    score: float | None,
    cfg: ServerConfig,
) -> bool:
    if prediction is None or score is None:
        return False
    if prediction in cfg.normal_labels:
        return False
    # Case-insensitive normal check
    if prediction.lower() in {n.lower() for n in cfg.normal_labels}:
        return False
    return score >= cfg.anomaly_threshold


@torch.no_grad()
def run_pipeline(
    runtime: ModelRuntime,
    frames_bgr: list[np.ndarray],
    preview_bgr: np.ndarray,
    cfg: ServerConfig,
) -> PipelineResult:
    if not frames_bgr:
        raise ValueError("No frames for inference")

    videos = bgr_frames_to_tensor(frames_bgr, cfg.img_size, runtime.device)
    features = encode_video_window(runtime, videos)

    detections = detect_from_patches(
        runtime.detr,
        features["last_patches"],
        threshold=cfg.detection_threshold,
    )

    anomaly_class: str | None = None
    anomaly_score: float | None = None
    top_k: list[dict[str, Any]] = []
    if runtime.has_anomaly:
        result = predict_from_cls(
            runtime.anomaly,
            features["cls_tokens"],
            runtime.anomaly_id2label,
            top_k=5,
        )
        anomaly_class = result["prediction"]
        anomaly_score = float(result["score"])
        top_k = result["top_k"]

    jpegs = render_all_previews(
        preview_bgr,
        detections=detections,
        patch_tokens=features["last_patches"][0],
        anomaly_label=anomaly_class,
        anomaly_score=anomaly_score,
        jpeg_quality=cfg.preview_jpeg_quality,
    )

    return PipelineResult(
        jpegs=jpegs,
        anomaly_class=anomaly_class,
        anomaly_score=anomaly_score,
        top_k=top_k,
        detections=detections,
        is_anomaly=is_anomaly_event(anomaly_class, anomaly_score, cfg),
        videos_tensor=videos.detach().cpu(),
        preview_bgr=preview_bgr,
    )
