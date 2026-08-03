"""VAU-Bench dataset helpers for Description captioning and anomaly classification.

Annotations come from Hugging Face ``7xiang/VAU-Bench``. UCF-Crime videos are
downloaded from ``etornam/ufc-crime-videos`` and staged under
``data/VAU-Bench/videos/`` as ``ucf_<original_filename>``.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from heads.vqa.dataset import (
    DEFAULT_NUM_FRAMES,
    IMG_SIZE,
    VIDEO_EXTENSIONS,
    _build_minicpm_batch,
    build_video_index,
    load_video_frames,
    resolve_samples_to_videos,
)

HF_REPO_ID = "7xiang/VAU-Bench"
UCF_HF_REPO_ID = "etornam/ufc-crime-videos"
DEFAULT_CACHE_DIR = "data/VAU-Bench"
DEFAULT_UCF_DOWNLOAD_DIR = "data/UCF-Crime/hf"
VALID_SPLITS = {"train", "validation", "val", "test"}
VALID_SOURCES = {"ucf", "msad", "ecva"}
CAPTION_PROMPT = "Describe the anomalous event in the video."

UCF_HF_FOLDERS = (
    "Anomaly-Videos-Part-1",
    "Anomaly-Videos-Part-2",
    "Anomaly-Videos-Part-3",
    "Anomaly-Videos-Part-4",
    "Normal_Videos_for_Event_Recognition",
    "Testing_Normal_Videos",
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
)

_UCF_CLASS_RE = re.compile(
    r"^(?:ucf_)?(?:"
    r"Abuse|Arrest|Arson|Assault|Burglary|Explosion|Fighting|"
    r"RoadAccidents|Robbery|Shooting|Shoplifting|Stealing|Vandalism"
    r")\d+_x264\.mp4$",
    re.IGNORECASE,
)
_UCF_NORMAL_RE = re.compile(
    r"^(?:ucf_)?Normal_Videos_\d+_x264\.mp4$",
    re.IGNORECASE,
)

# HF column names
COL_VIDEO = "Video Name"
COL_DESCRIPTION = "Description"
COL_ANOMALY_CLASS = "Anomaly Class"
COL_START = "Start Time"
COL_END = "End Time"


def _normalize_split(split: str) -> str:
    if split == "val":
        return "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            f"Unknown split {split!r}; expected one of "
            f"{sorted(VALID_SPLITS)}"
        )
    return split


def _parse_sources(sources: Sequence[str] | str | None) -> set[str] | None:
    if sources is None:
        return None
    if isinstance(sources, str):
        values = [part.strip().casefold() for part in sources.split(",") if part.strip()]
    else:
        values = [str(part).strip().casefold() for part in sources if str(part).strip()]
    if not values:
        return None
    invalid = [value for value in values if value not in VALID_SOURCES]
    if invalid:
        raise ValueError(
            f"Unknown sources {invalid!r}; expected subset of {sorted(VALID_SOURCES)}"
        )
    return set(values)


def is_ucf_video_name(video_name: str) -> bool:
    name = Path(video_name).name
    if name.startswith("ucf_"):
        return True
    return bool(_UCF_CLASS_RE.match(name) or _UCF_NORMAL_RE.match(name))


def normalize_ucf_video_name(video_name: str) -> str:
    """Ensure UCF filenames use the ``ucf_`` prefix expected after staging."""
    name = Path(video_name).name
    if name.startswith("ucf_"):
        return name
    if is_ucf_video_name(name):
        return f"ucf_{name}"
    return name


def _video_source(video_name: str) -> str | None:
    name = Path(video_name).name
    if name.startswith("ucf_") or is_ucf_video_name(name):
        return "ucf"
    if name.startswith("msad_"):
        return "msad"
    if name.startswith("ecva_"):
        return "ecva"
    return None


def _parse_time(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _has_valid_trim(start_sec: float, end_sec: float) -> bool:
    return start_sec >= 0 and end_sec >= 0 and end_sec > start_sec


def ensure_vau_annotations(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Download VAU-Bench annotations into the local cache and return the root."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    # Trigger a download/cache of all splits.
    load_dataset(HF_REPO_ID, cache_dir=str(cache_root / "hf_cache"))
    return cache_root


def ensure_ucf_videos(
    download_dir: str | Path = DEFAULT_UCF_DOWNLOAD_DIR,
    video_root: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    stage: bool = True,
    link: bool = True,
    folders: Sequence[str] | None = None,
) -> dict[str, object]:
    """Download UCF-Crime videos from Hugging Face and optionally stage them.

    Videos are pulled from ``etornam/ufc-crime-videos`` into ``download_dir``,
    then symlinked/copied into ``video_root`` as ``ucf_<filename>``.
    """
    download_root = Path(download_dir)
    download_root.mkdir(parents=True, exist_ok=True)
    selected = list(folders) if folders is not None else list(UCF_HF_FOLDERS)
    allow_patterns: list[str] = [f"{folder}/**" for folder in selected]
    allow_patterns.extend(
        [
            "Anomaly_Train.txt",
            "ReadMe-Anomaly-Detection.txt",
            "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
            "UCF_Crimes-Train-Test-Split/**",
        ]
    )

    print(
        f"Downloading UCF-Crime videos from {UCF_HF_REPO_ID} "
        f"into {download_root} (~105 GB) ..."
    )
    local_path = snapshot_download(
        repo_id=UCF_HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(download_root),
        allow_patterns=allow_patterns,
    )

    report: dict[str, object] = {
        "download_dir": str(local_path),
        "repo_id": UCF_HF_REPO_ID,
        "folders": selected,
    }
    if stage:
        if video_root is None:
            video_root = Path(cache_dir) / "videos"
        stage_report = stage_prefixed_videos(
            local_path,
            video_root,
            prefix="ucf_",
            link=link,
        )
        report["stage"] = stage_report
    return report


def load_vau_raw_split(
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
):
    """Load a raw Hugging Face VAU-Bench split."""
    split = _normalize_split(split)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        HF_REPO_ID,
        split=split,
        cache_dir=str(cache_root / "hf_cache"),
    )


def _row_to_sample(row, index: int) -> dict | None:
    video_name = Path(str(row.get(COL_VIDEO, ""))).name.strip()
    description = str(row.get(COL_DESCRIPTION, "") or "").strip()
    anomaly_class = str(row.get(COL_ANOMALY_CLASS, "") or "").strip()
    if not video_name or not description or not anomaly_class:
        return None
    if anomaly_class in {"-1", "None", "N/A", "n/a"}:
        return None

    start_sec = _parse_time(row.get(COL_START, -1))
    end_sec = _parse_time(row.get(COL_END, -1))
    if not _has_valid_trim(start_sec, end_sec):
        start_sec, end_sec = -1.0, -1.0

    return {
        "id": f"{Path(video_name).stem}_{index}",
        "video_name": video_name,
        "clip_id": Path(video_name).stem,
        "question": CAPTION_PROMPT,
        "answer": description,
        "description": description,
        "anomaly_class": anomaly_class,
        "qa_type": "Description",
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def _prefer_trim(existing: dict, candidate: dict) -> dict:
    """Keep the row with a valid temporal span when deduplicating."""
    existing_ok = _has_valid_trim(existing["start_sec"], existing["end_sec"])
    candidate_ok = _has_valid_trim(candidate["start_sec"], candidate["end_sec"])
    if candidate_ok and not existing_ok:
        merged = dict(candidate)
        merged["id"] = existing["id"]
        return merged
    return existing


def load_vau_samples(
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    dedup_by_video: bool = True,
    sources: Sequence[str] | str | None = None,
) -> list[dict]:
    """Load and normalize VAU-Bench rows for Description / Anomaly Class.

    When ``dedup_by_video`` is True (default), keep one sample per
    ``Video Name``, preferring the first row that has a valid Start/End span.

    ``sources`` filters by dataset origin (``ucf``, ``msad``, ``ecva``).
    Bare UCF test filenames are normalized to the ``ucf_`` prefix.
    """
    allowed = _parse_sources(sources)
    dataset = load_vau_raw_split(split, cache_dir=cache_dir)
    samples: list[dict] = []
    by_video: dict[str, dict] = {}
    skipped = 0
    filtered = 0

    for index, row in enumerate(dataset):
        sample = _row_to_sample(row, index)
        if sample is None:
            skipped += 1
            continue

        source = _video_source(sample["video_name"])
        if allowed is not None and source not in allowed:
            filtered += 1
            continue

        if source == "ucf":
            sample["video_name"] = normalize_ucf_video_name(sample["video_name"])
            sample["clip_id"] = Path(sample["video_name"]).stem

        if not dedup_by_video:
            samples.append(sample)
            continue
        name = sample["video_name"]
        if name not in by_video:
            by_video[name] = sample
        else:
            by_video[name] = _prefer_trim(by_video[name], sample)

    if dedup_by_video:
        samples = list(by_video.values())

    if skipped:
        print(f"Warning: skipped {skipped} VAU-Bench rows with missing fields")
    if filtered:
        print(
            f"Filtered out {filtered} rows not in sources="
            f"{sorted(allowed) if allowed else []}"
        )
    if not samples:
        raise RuntimeError(
            f"No VAU-Bench samples found for split={split!r}"
            + (f", sources={sorted(allowed)}" if allowed else "")
        )
    return samples


def build_label_maps(
    samples: Sequence[dict],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build sorted anomaly-class label maps from samples."""
    classes = sorted({sample["anomaly_class"] for sample in samples})
    if not classes:
        raise RuntimeError("No anomaly classes found in samples")
    label2id = {name: index for index, name in enumerate(classes)}
    id2label = {index: name for name, index in label2id.items()}
    return label2id, id2label


def save_label_maps(
    label2id: dict[str, int],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label2id, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_label_maps(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    label2id = {str(name): int(index) for name, index in raw.items()}
    id2label = {index: name for name, index in label2id.items()}
    return label2id, id2label


def resolve_vau_samples(
    samples: list[dict],
    video_root: str | Path,
    skip_missing: bool = False,
) -> list[dict]:
    """Resolve VAU-Bench video filenames under ``video_root``."""
    # Reuse CUVA resolver; it only needs video_name keys.
    return resolve_samples_to_videos(
        samples,
        build_video_index(video_root),
        skip_missing=skip_missing,
    )


def verify_vau_videos(
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    video_root: str | Path | None = None,
    sources: Sequence[str] | str | None = None,
) -> dict[str, object]:
    """Report how many annotation videos are present under ``video_root``."""
    if video_root is None:
        video_root = Path(cache_dir) / "videos"
    samples = load_vau_samples(
        split, cache_dir=cache_dir, dedup_by_video=True, sources=sources
    )
    root = Path(video_root)
    if not root.exists():
        return {
            "split": _normalize_split(split),
            "total": len(samples),
            "found": 0,
            "missing": len(samples),
            "missing_examples": [s["video_name"] for s in samples[:10]],
            "video_root": str(root),
        }

    index = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            index[path.name] = path

    missing = [s["video_name"] for s in samples if s["video_name"] not in index]
    found = len(samples) - len(missing)
    return {
        "split": _normalize_split(split),
        "total": len(samples),
        "found": found,
        "missing": len(missing),
        "missing_examples": missing[:10],
        "video_root": str(root),
    }


class VAUCaptionDataset(Dataset):
    """Video → Description captioning samples (fixed prompt, no MCQ question)."""

    def __init__(
        self,
        samples: list[dict],
        num_frames: int = DEFAULT_NUM_FRAMES,
        img_size: int = IMG_SIZE,
    ):
        self.samples = samples
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        start = sample.get("start_sec", -1.0)
        end = sample.get("end_sec", -1.0)
        start_sec = float(start) if float(start) >= 0 else None
        end_sec = float(end) if float(end) >= 0 else None
        return {
            "video": load_video_frames(
                sample["video_path"],
                num_frames=self.num_frames,
                img_size=self.img_size,
                start_sec=start_sec,
                end_sec=end_sec,
            ),
            "question": sample["question"],
            "answer": sample["answer"],
            "qa_type": sample["qa_type"],
            "clip_id": sample["clip_id"],
            "sample_id": sample["id"],
            "video_path": sample["video_path"],
            "anomaly_class": sample["anomaly_class"],
            "start_sec": sample["start_sec"],
            "end_sec": sample["end_sec"],
        }


class VAUClassDataset(Dataset):
    """Video → Anomaly Class classification samples."""

    def __init__(
        self,
        samples: list[dict],
        label2id: dict[str, int],
        num_frames: int = DEFAULT_NUM_FRAMES,
        img_size: int = IMG_SIZE,
    ):
        self.samples = samples
        self.label2id = label2id
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        start = sample.get("start_sec", -1.0)
        end = sample.get("end_sec", -1.0)
        start_sec = float(start) if float(start) >= 0 else None
        end_sec = float(end) if float(end) >= 0 else None
        label = self.label2id[sample["anomaly_class"]]
        return {
            "video": load_video_frames(
                sample["video_path"],
                num_frames=self.num_frames,
                img_size=self.img_size,
                start_sec=start_sec,
                end_sec=end_sec,
            ),
            "label": label,
            "anomaly_class": sample["anomaly_class"],
            "clip_id": sample["clip_id"],
            "sample_id": sample["id"],
            "video_path": sample["video_path"],
            "start_sec": sample["start_sec"],
            "end_sec": sample["end_sec"],
        }


def make_vau_caption_dataloader(
    tokenizer: PreTrainedTokenizer,
    video_root: str | Path | None = None,
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    img_size: int = IMG_SIZE,
    max_length: int = 256,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    skip_missing: bool = False,
    sources: Sequence[str] | str | None = None,
) -> tuple[DataLoader, int]:
    if video_root is None:
        video_root = Path(cache_dir) / "videos"

    samples = load_vau_samples(
        split, cache_dir=cache_dir, dedup_by_video=True, sources=sources
    )
    samples = resolve_vau_samples(samples, video_root, skip_missing=skip_missing)
    dataset = VAUCaptionDataset(samples, num_frames=num_frames, img_size=img_size)

    def collate_fn(batch):
        return _build_minicpm_batch(batch, tokenizer, max_length)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    drop_last = _normalize_split(split) == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader, len(loader)


def _collate_class_batch(batch: list[dict]) -> dict:
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "anomaly_class": [item["anomaly_class"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "video_path": [item["video_path"] for item in batch],
    }


def make_vau_class_dataloader(
    label2id: dict[str, int],
    video_root: str | Path | None = None,
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    img_size: int = IMG_SIZE,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    skip_missing: bool = False,
    sources: Sequence[str] | str | None = None,
) -> tuple[DataLoader, int]:
    if video_root is None:
        video_root = Path(cache_dir) / "videos"

    samples = load_vau_samples(
        split, cache_dir=cache_dir, dedup_by_video=True, sources=sources
    )
    # Drop classes not in the training vocabulary.
    unknown = [
        s for s in samples if s["anomaly_class"] not in label2id
    ]
    if unknown:
        print(
            f"Warning: dropping {len(unknown)} samples with unseen classes "
            f"(examples: {[u['anomaly_class'] for u in unknown[:5]]})"
        )
        samples = [s for s in samples if s["anomaly_class"] in label2id]
    samples = resolve_vau_samples(samples, video_root, skip_missing=skip_missing)
    dataset = VAUClassDataset(
        samples, label2id=label2id, num_frames=num_frames, img_size=img_size
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    drop_last = _normalize_split(split) == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_class_batch,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader, len(loader)


def stage_prefixed_videos(
    source_dir: str | Path,
    dest_dir: str | Path,
    prefix: str = "ucf_",
    link: bool = True,
) -> dict[str, int]:
    """Copy or symlink videos into ``dest_dir`` with a VAU-Bench filename prefix.

    UCF-Crime archives ship files like ``Abuse001_x264.mp4``. VAU-Bench expects
    ``ucf_Abuse001_x264.mp4``. This walks ``source_dir`` recursively and stages
    every supported video as ``{prefix}{original_filename}``.
    """
    source = Path(source_dir)
    dest = Path(dest_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source video directory not found: {source}")
    dest.mkdir(parents=True, exist_ok=True)

    staged = 0
    skipped = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        target_name = f"{prefix}{path.name}"
        target = dest / target_name
        if target.exists() or target.is_symlink():
            skipped += 1
            continue
        if link:
            target.symlink_to(path.resolve())
        else:
            shutil.copy2(path, target)
        staged += 1

    return {"staged": staged, "skipped_existing": skipped, "dest": str(dest)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download or verify VAU-Bench annotations / videos"
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--split",
        choices=sorted(VALID_SPLITS),
        default="train",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Directory of source videos (default: <cache-dir>/videos)",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source filter, e.g. ucf or ucf,msad (default: all)",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download/inspect annotations without requiring videos",
    )
    parser.add_argument(
        "--verify-videos",
        action="store_true",
        help="Report how many annotation videos exist under video-root",
    )
    parser.add_argument(
        "--download-ucf",
        action="store_true",
        help=(
            f"Download UCF-Crime videos from {UCF_HF_REPO_ID} (~105 GB) "
            f"into {DEFAULT_UCF_DOWNLOAD_DIR} and stage as ucf_* under videos/"
        ),
    )
    parser.add_argument(
        "--ucf-download-dir",
        default=DEFAULT_UCF_DOWNLOAD_DIR,
        help=f"Local dir for HF UCF snapshot (default: {DEFAULT_UCF_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="With --download-ucf, skip download and only stage from --ucf-download-dir",
    )
    parser.add_argument(
        "--stage-ucf",
        metavar="SOURCE_DIR",
        default=None,
        help=(
            "Recursively find UCF-Crime .mp4 files under SOURCE_DIR and "
            "symlink them into <cache-dir>/videos as ucf_<filename>"
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="With --download-ucf / --stage-ucf, copy files instead of symlinks",
    )
    args = parser.parse_args()
    sources = _parse_sources(args.sources)

    if args.download_ucf:
        dest = Path(args.video_root) if args.video_root else Path(args.cache_dir) / "videos"
        if args.stage_only:
            report = {
                "download_dir": str(args.ucf_download_dir),
                "stage": stage_prefixed_videos(
                    args.ucf_download_dir,
                    dest,
                    prefix="ucf_",
                    link=not args.copy,
                ),
            }
        else:
            report = ensure_ucf_videos(
                download_dir=args.ucf_download_dir,
                video_root=dest,
                cache_dir=args.cache_dir,
                stage=True,
                link=not args.copy,
            )
        print(json.dumps(report, indent=2, default=str))
    elif args.stage_ucf:
        dest = Path(args.video_root) if args.video_root else Path(args.cache_dir) / "videos"
        report = stage_prefixed_videos(
            args.stage_ucf,
            dest,
            prefix="ucf_",
            link=not args.copy,
        )
        print(json.dumps(report, indent=2))
    elif args.verify_videos:
        report = verify_vau_videos(
            split=args.split,
            cache_dir=args.cache_dir,
            video_root=args.video_root,
            sources=sources,
        )
        print(json.dumps(report, indent=2))
    else:
        ensure_vau_annotations(args.cache_dir)
        samples = load_vau_samples(
            args.split, cache_dir=args.cache_dir, sources=sources
        )
        print(f"VAU-Bench {args.split} unique videos: {len(samples)}")
        example = samples[0]
        print(
            "Example:",
            {
                key: example[key]
                for key in (
                    "id",
                    "video_name",
                    "anomaly_class",
                    "start_sec",
                    "end_sec",
                    "answer",
                )
            },
        )
        classes = sorted({s["anomaly_class"] for s in samples})
        print(f"Anomaly classes ({len(classes)}): {classes}")
        if not args.annotations_only:
            print(
                "Annotations are ready. Download UCF videos with "
                "`python -m heads.vqa.vau_dataset --download-ucf`, then "
                "run with --verify-videos --sources ucf."
            )
