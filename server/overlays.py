"""Draw detection boxes / prepare preview JPEG."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from heads.detr.predict import boxes_xyxy_pixels
from server.pca import blend_pca_overlay


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1, label, score in boxes_xyxy_pixels(detections, w, h):
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 40, 220), 2)
        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(0, y0 - 4)
        cv2.rectangle(
            out,
            (x0, y_text - th - baseline - 2),
            (x0 + tw + 4, y_text + 2),
            (40, 40, 220),
            -1,
        )
        cv2.putText(
            out,
            text,
            (x0 + 2, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_anomaly_badge(
    frame_bgr: np.ndarray,
    label: str | None,
    score: float | None,
) -> np.ndarray:
    if not label:
        return frame_bgr
    out = frame_bgr.copy()
    text = f"{label}" if score is None else f"{label}  {score:.2f}"
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 2)
    x0, y0 = 12, 12
    cv2.rectangle(
        out,
        (x0, y0),
        (x0 + tw + pad * 2, y0 + th + baseline + pad * 2),
        (20, 20, 20),
        -1,
    )
    # Amber accent bar
    cv2.rectangle(
        out,
        (x0, y0),
        (x0 + 4, y0 + th + baseline + pad * 2),
        (0, 165, 255),
        -1,
    )
    cv2.putText(
        out,
        text,
        (x0 + pad + 2, y0 + pad + th),
        font,
        scale,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return out


def encode_jpeg(frame_bgr: np.ndarray, jpeg_quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def render_all_previews(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    patch_tokens,
    anomaly_label: str | None,
    anomaly_score: float | None,
    jpeg_quality: int = 75,
) -> dict[str, bytes]:
    """Encode detection, PCA, and anomaly preview JPEGs."""
    detection = draw_detections(frame_bgr, detections)
    if patch_tokens is not None:
        pca = blend_pca_overlay(frame_bgr, patch_tokens)
    else:
        pca = frame_bgr.copy()
    anomaly = draw_anomaly_badge(frame_bgr.copy(), anomaly_label, anomaly_score)
    return {
        "detection": encode_jpeg(detection, jpeg_quality),
        "pca": encode_jpeg(pca, jpeg_quality),
        "anomaly": encode_jpeg(anomaly, jpeg_quality),
    }


def render_preview(
    frame_bgr: np.ndarray,
    overlay_mode: str,
    detections: list[dict[str, Any]],
    patch_tokens,
    anomaly_label: str | None,
    anomaly_score: float | None,
    jpeg_quality: int = 75,
) -> bytes:
    """Compose a single overlay + badge and encode JPEG bytes."""
    if overlay_mode == "detection":
        framed = draw_detections(frame_bgr, detections)
    elif overlay_mode == "pca" and patch_tokens is not None:
        framed = blend_pca_overlay(frame_bgr, patch_tokens)
    else:
        framed = frame_bgr.copy()

    framed = draw_anomaly_badge(framed, anomaly_label, anomaly_score)
    return encode_jpeg(framed, jpeg_quality)
