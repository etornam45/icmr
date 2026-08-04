"""ICMR server configuration (env-overridable)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class ServerConfig:
    backbone_weights: str = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    detr_checkpoint: str = "dinov3/checkpoints/model/detr_decoder.pt"
    anomaly_checkpoint: str = "dinov3/checkpoints/model/anomaly_vau"
    vqa_checkpoint: str = "dinov3/checkpoints/model/vqa_vau_minicpm_best"
    vqa_fallback_checkpoint: str = "dinov3/checkpoints/model/vqa_minicpm"

    img_size: int = 224
    num_frames: int = 16
    buffer_seconds: float = 8.0
    inference_interval_sec: float = 0.75
    preview_jpeg_quality: int = 75

    detection_threshold: float = 0.7
    anomaly_threshold: float = 0.35
    anomaly_cooldown_sec: float = 30.0
    normal_labels: set[str] = field(
        default_factory=lambda: {
            "Normal",
            "normal",
            "Normal_Videos",
            "Normal Videos",
            "none",
            "None",
            "N/A",
        }
    )

    events_db_path: str = "logs/icmr_events.db"
    uploads_dir: str = "logs/uploads"
    max_upload_mb: int = 512
    max_events: int = 200
    load_vqa: bool = True
    default_overlay: str = "none"  # none | detection | pca

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )


def load_config() -> ServerConfig:
    cfg = ServerConfig()
    cfg.backbone_weights = _env_str("ICMR_BACKBONE", cfg.backbone_weights)
    cfg.detr_checkpoint = _env_str("ICMR_DETR", cfg.detr_checkpoint)
    cfg.anomaly_checkpoint = _env_str("ICMR_ANOMALY", cfg.anomaly_checkpoint)
    cfg.vqa_checkpoint = _env_str("ICMR_VQA", cfg.vqa_checkpoint)
    cfg.vqa_fallback_checkpoint = _env_str(
        "ICMR_VQA_FALLBACK", cfg.vqa_fallback_checkpoint
    )
    cfg.num_frames = _env_int("ICMR_NUM_FRAMES", cfg.num_frames)
    cfg.inference_interval_sec = _env_float(
        "ICMR_INFERENCE_INTERVAL", cfg.inference_interval_sec
    )
    cfg.detection_threshold = _env_float(
        "ICMR_DET_THRESHOLD", cfg.detection_threshold
    )
    cfg.anomaly_threshold = _env_float(
        "ICMR_ANOMALY_THRESHOLD", cfg.anomaly_threshold
    )
    cfg.anomaly_cooldown_sec = _env_float(
        "ICMR_ANOMALY_COOLDOWN", cfg.anomaly_cooldown_sec
    )
    cfg.events_db_path = _env_str("ICMR_EVENTS_DB", cfg.events_db_path)
    cfg.uploads_dir = _env_str("ICMR_UPLOADS_DIR", cfg.uploads_dir)
    cfg.max_upload_mb = _env_int("ICMR_MAX_UPLOAD_MB", cfg.max_upload_mb)
    cfg.default_overlay = _env_str("ICMR_OVERLAY", cfg.default_overlay)
    cfg.load_vqa = os.getenv("ICMR_LOAD_VQA", "1") not in {"0", "false", "False"}
    cfg.host = _env_str("ICMR_HOST", cfg.host)
    cfg.port = _env_int("ICMR_PORT", cfg.port)
    return cfg
