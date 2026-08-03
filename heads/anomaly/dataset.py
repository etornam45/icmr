"""VAU-Bench Anomaly Class dataset wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heads.vqa.vau_dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_NUM_FRAMES,
    IMG_SIZE,
    build_label_maps,
    load_vau_samples,
    make_vau_class_dataloader,
    resolve_vau_samples,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_NUM_FRAMES",
    "IMG_SIZE",
    "build_label_maps",
    "load_vau_samples",
    "make_vau_class_dataloader",
    "resolve_vau_samples",
    "build_train_label_maps",
]


def build_train_label_maps(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    sources: Sequence[str] | str | None = "ucf",
) -> tuple[dict[str, int], dict[int, str]]:
    """Build class vocabulary from the VAU-Bench train split."""
    samples = load_vau_samples(
        "train",
        cache_dir=cache_dir,
        dedup_by_video=True,
        sources=sources,
    )
    return build_label_maps(samples)
