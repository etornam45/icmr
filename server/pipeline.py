"""Inference pipeline: preprocess frames → shared DINO → heads → overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from heads.anomaly.inference import predict_from_patches
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
    segments: list[dict[str, Any]] = field(default_factory=list)
    segment: dict[str, Any] | None = None
    svdd_score: float | None = None
    window_start_ts: float | None = None
    window_end_ts: float | None = None


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
    segments: list[dict[str, Any]] | None = None,
) -> bool:
    if segments:
        return any(
            s.get("class") not in cfg.normal_labels
            and str(s.get("class", "")).lower()
            not in {n.lower() for n in cfg.normal_labels}
            and float(s.get("confidence", 0.0)) >= cfg.deploy_threshold
            for s in segments
        )
    if prediction is None or score is None:
        return False
    if prediction in cfg.normal_labels:
        return False
    if prediction.lower() in {n.lower() for n in cfg.normal_labels}:
        return False
    return score >= cfg.anomaly_threshold


@torch.no_grad()
def run_pipeline(
    runtime: ModelRuntime,
    frames_bgr: list[np.ndarray],
    preview_bgr: np.ndarray,
    cfg: ServerConfig,
    window_start_ts: float | None = None,
    window_end_ts: float | None = None,
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
    segments: list[dict[str, Any]] = []
    segment: dict[str, Any] | None = None
    svdd_score: float | None = None

    if runtime.has_anomaly:
        result = predict_from_patches(
            runtime.anomaly,
            features["patch_tokens"],
            runtime.anomaly_id2label,
            top_k=5,
        )
        anomaly_class = result["prediction"]
        anomaly_score = float(result["score"])
        top_k = result["top_k"]
        segments = list(result.get("segments") or [])
        segment = result.get("segment")

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
        is_anomaly=is_anomaly_event(
            anomaly_class, anomaly_score, cfg, segments=segments
        ),
        videos_tensor=videos.detach().cpu(),
        preview_bgr=preview_bgr,
        segments=segments,
        segment=segment,
        svdd_score=svdd_score,
        window_start_ts=window_start_ts,
        window_end_ts=window_end_ts,
    )
