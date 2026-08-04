"""Reusable DETR detection helpers (no CLI / no model reload)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def postprocess_detections(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    threshold: float = 0.62,
) -> list[dict[str, Any]]:
    """Convert DETR logits/boxes [Q, C] / [Q, 4] into a list of detections."""
    probs = F.softmax(logits, dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)
    keep = scores > threshold
    scores_np = scores[keep].detach().cpu().numpy()
    labels_np = labels[keep].detach().cpu().numpy()
    boxes_np = boxes[keep].detach().cpu().numpy()

    detections: list[dict[str, Any]] = []
    for score, label, (cx, cy, w, h) in zip(scores_np, labels_np, boxes_np):
        label_i = int(label)
        class_name = (
            COCO_CLASSES[label_i] if label_i < len(COCO_CLASSES) else f"Class {label_i}"
        )
        detections.append(
            {
                "class": class_name,
                "score": float(score),
                "box": {
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                },
            }
        )
    return detections


@torch.no_grad()
def detect_from_patches(
    detr_decoder: torch.nn.Module,
    patch_tokens: torch.Tensor,
    threshold: float = 0.62,
) -> list[dict[str, Any]]:
    """
    Run DETR decoder on patch tokens.

    Args:
        patch_tokens: [B, N, D] or [N, D]
    """
    if patch_tokens.dim() == 2:
        patch_tokens = patch_tokens.unsqueeze(0)
    output = detr_decoder(patch_tokens)
    return postprocess_detections(output["logits"][0], output["boxes"][0], threshold)


def boxes_xyxy_pixels(
    detections: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int, str, float]]:
    """Convert normalized cxcywh detections to pixel xyxy + label + score."""
    results: list[tuple[int, int, int, int, str, float]] = []
    for det in detections:
        box = det["box"]
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        x0 = int((cx - w / 2) * width)
        y0 = int((cy - h / 2) * height)
        x1 = int((cx + w / 2) * width)
        y1 = int((cy + h / 2) * height)
        results.append((x0, y0, x1, y1, det["class"], det["score"]))
    return results
