"""Shared video loading utilities for VAU-Bench / UCF-Crime caption training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from heads.detr.dataset import letterbox

IMG_SIZE = 224
DEFAULT_NUM_FRAMES = 16
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def build_video_index(video_root: str | Path) -> dict[str, Path]:
    """Index videos by exact filename and reject duplicate filenames."""
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Video root not found: {root}. Pass a directory containing "
            "extracted videos, or download them first."
        )

    index: dict[str, Path] = {}
    collisions: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        key = path.name
        if key in index:
            collisions.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path

    if collisions:
        examples = [
            f"{name}: {[str(path) for path in paths]}"
            for name, paths in list(collisions.items())[:5]
        ]
        raise RuntimeError(
            "Duplicate video filenames found. Each video name must map "
            "to exactly one file.\n" + "\n".join(examples)
        )
    if not index:
        raise RuntimeError(f"No supported video files found under {root}")
    return index


def resolve_samples_to_videos(
    samples: list[dict[str, str]],
    video_index: dict[str, Path],
    skip_missing: bool = False,
) -> list[dict[str, str]]:
    """Resolve each sample ``video_name`` to its extracted video path."""
    resolved: list[dict[str, str]] = []
    missing: list[str] = []
    for sample in samples:
        video_name = sample["video_name"]
        path = video_index.get(video_name)
        if path is None:
            missing.append(video_name)
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"No extracted video found for video_name={video_name!r}"
            )
        item = dict(sample)
        item["video_path"] = str(path)
        resolved.append(item)

    if missing:
        unique_missing = sorted(set(missing))
        print(
            f"Warning: {len(unique_missing)} videos are missing "
            f"({len(missing)} rows skipped). Examples: {unique_missing[:5]}"
        )
    if not resolved:
        raise RuntimeError("No annotation rows could be matched to videos")
    return resolved


def sample_frame_indices(num_frames_total: int, num_frames: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames_total <= 1:
        return [0] * num_frames
    if num_frames == 1:
        return [num_frames_total // 2]
    return [
        round(index * (num_frames_total - 1) / (num_frames - 1))
        for index in range(num_frames)
    ]


def sample_frame_indices_in_range(
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> list[int]:
    """Uniformly sample ``num_frames`` indices in the inclusive range [start, end]."""
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame ({end_frame}) must be >= start_frame ({start_frame})"
        )
    span = end_frame - start_frame + 1
    relative = sample_frame_indices(span, num_frames)
    return [start_frame + offset for offset in relative]


def _resolve_trim_frame_range(
    total_frames: int,
    fps: float,
    start_sec: float | None,
    end_sec: float | None,
    video_path: str | Path,
) -> tuple[int, int]:
    """Map optional start/end seconds to an inclusive frame range.

    ``-1`` or ``None`` for either bound means use the full video. Invalid
    ranges fall back to the full video with a warning.
    """
    use_full = (
        start_sec is None
        or end_sec is None
        or float(start_sec) < 0
        or float(end_sec) < 0
    )
    if use_full or total_frames <= 0:
        return 0, max(total_frames - 1, 0)

    if fps <= 0:
        print(
            f"Warning: invalid FPS ({fps}) for {video_path}; using full video"
        )
        return 0, max(total_frames - 1, 0)

    start_f = round(float(start_sec) * fps)
    end_f = round(float(end_sec) * fps)
    start_f = max(0, min(start_f, total_frames - 1))
    end_f = max(0, min(end_f, total_frames - 1))
    if start_f >= end_f:
        print(
            f"Warning: empty trim range [{start_sec}, {end_sec}] for "
            f"{video_path}; using full video"
        )
        return 0, max(total_frames - 1, 0)
    return start_f, end_f


def load_video_frames(
    video_path: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    img_size: int = IMG_SIZE,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> torch.Tensor:
    """Uniformly sample a video into a [T, 3, H, W] float tensor.

    When both ``start_sec`` and ``end_sec`` are provided and non-negative,
    frames are sampled only inside that temporal window.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        total = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        start_f, end_f = _resolve_trim_frame_range(
            max(total, 0), fps, start_sec, end_sec, video_path
        )
        indices = sample_frame_indices_in_range(start_f, end_f, num_frames)
        frames: list[torch.Tensor] = []
        last_frame: np.ndarray | None = None
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                last_frame = frame_bgr
            elif last_frame is not None:
                frame_bgr = last_frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"Could not decode any frames from {video_path}")
                last_frame = frame_bgr

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image, _, _, _ = letterbox(Image.fromarray(frame_rgb), img_size)
            array = np.array(image, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames)
    finally:
        cap.release()


def build_caption_batch(
    batch: list[dict],
    tokenizer: Any,
    max_length: int = 128,
) -> dict:
    """Collate video + description using CaptionTokenizer (or any batch_encode API)."""
    videos = torch.stack([item["video"] for item in batch])
    texts = [item.get("answer") or item.get("description") or "" for item in batch]
    encoded = tokenizer.batch_encode(texts, max_length=max_length)
    return {
        "video": videos,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "question": [item.get("question", "") for item in batch],
        "answer": texts,
        "clip_id": [item.get("clip_id", "") for item in batch],
        "sample_id": [item.get("sample_id", item.get("id", "")) for item in batch],
        "video_path": [item.get("video_path", "") for item in batch],
    }
