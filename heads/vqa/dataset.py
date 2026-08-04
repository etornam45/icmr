"""CUVA video QA dataset for DINOv3 spatial-pool + MiniCPM."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from heads.detr.dataset import letterbox
from heads.vqa.minicpm_loader import apply_chat, tokenize_chat_pair

HF_REPO_ID = "fesvhtr/CUVA"
DEFAULT_CACHE_DIR = "data/CUVA"
IMG_SIZE = 224
DEFAULT_NUM_FRAMES = 16
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
VIDEO_ARCHIVES = tuple(f"raw/group_{index}.zip" for index in range(4))
PARQUET_FILES = {
    "full": "data/all.parquet",
    "test": "data/test.parquet",
}
VALID_SPLITS = {"train", "full", "test"}
TASK_ALIASES = {
    "T1": "Classification",
    "T2": "Cause",
    "T3": "Result",
    "T4": "Timestamp",
    "T5": "Description",
}


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract an archive while rejecting absolute paths and path traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = []
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            if "__MACOSX" in member_path.parts or member_path.name.startswith("._"):
                continue
            target = (dest_dir / member_path).resolve()
            if not target.is_relative_to(dest_root):
                raise RuntimeError(f"Zip slip detected for member: {member.filename}")
            members.append(member)
        archive.extractall(dest_dir, members=members)


def ensure_cuva_annotations(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict[str, Path]:
    """Download CUVA parquet annotations without downloading video archives."""
    cache_root = Path(cache_dir)
    paths: dict[str, Path] = {}
    for split, repo_path in PARQUET_FILES.items():
        paths[split] = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=repo_path,
                local_dir=str(cache_root),
            )
        )
    return paths


def ensure_cuva_videos(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    groups: Sequence[int] | None = None,
    force_extract: bool = False,
) -> Path:
    """Download and extract CUVA video groups (about 25.6 GB in total)."""
    cache_root = Path(cache_dir)
    video_root = cache_root / "videos"
    selected = list(range(4)) if groups is None else list(groups)
    invalid = [group for group in selected if group not in range(4)]
    if invalid:
        raise ValueError(f"CUVA video groups must be 0-3, got {invalid}")

    for group in selected:
        repo_path = VIDEO_ARCHIVES[group]
        marker = video_root / f".group_{group}.extracted"
        if marker.exists() and not force_extract:
            continue
        zip_path = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=repo_path,
                local_dir=str(cache_root),
            )
        )
        print(f"Extracting {zip_path.name} to {video_root} ...")
        _safe_extract_zip(zip_path, video_root)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")
    return video_root


def load_cuva_samples(
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    tasks: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Load CUVA rows and normalize them to question/answer video samples.

    CUVA's ``full`` split contains both the original records and the records
    from ``test``. Test videos also retain older rows under different IDs, so
    the ``train`` alias removes every row whose ``visual_input`` occurs in
    ``test`` to avoid video-level evaluation leakage.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(VALID_SPLITS)}")

    parquet = ensure_cuva_annotations(cache_dir)
    source_split = "full" if split in {"train", "full"} else "test"
    dataset = load_dataset(
        "parquet",
        data_files={source_split: str(parquet[source_split])},
        split=source_split,
    )

    test_video_names: set[str] = set()
    if split == "train":
        test_dataset = load_dataset(
            "parquet",
            data_files={"test": str(parquet["test"])},
            split="test",
        )
        test_video_names = {
            Path(str(video_name)).name
            for video_name in test_dataset["visual_input"]
        }

    allowed_tasks = (
        {task.casefold() for task in tasks} if tasks is not None else None
    )
    samples: list[dict[str, str]] = []
    for row in dataset:
        row_id = str(row["ID"])
        video_name = Path(str(row["visual_input"])).name
        if video_name in test_video_names:
            continue
        raw_task = str(row["task"]).strip()
        task = TASK_ALIASES.get(raw_task, raw_task)
        if allowed_tasks is not None and task.casefold() not in allowed_tasks:
            continue
        question = str(row["instruction"]).strip()
        answer = str(row["output"]).strip()
        if not video_name or not question or not answer:
            continue
        samples.append(
            {
                "id": row_id,
                "video_name": video_name,
                "clip_id": Path(video_name).stem,
                "question": question,
                "answer": answer,
                "qa_type": task,
            }
        )

    if not samples:
        raise RuntimeError(f"No CUVA samples found for split={split!r}")
    return samples


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


class CUVADataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, str]],
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
        return {
            "video": load_video_frames(
                sample["video_path"],
                num_frames=self.num_frames,
                img_size=self.img_size,
            ),
            "question": sample["question"],
            "answer": sample["answer"],
            "qa_type": sample["qa_type"],
            "clip_id": sample["clip_id"],
            "sample_id": sample["id"],
            "video_path": sample["video_path"],
        }


def _pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
) -> torch.Tensor:
    max_length = max(len(sequence) for sequence in sequences)
    batch = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        batch[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch


def _build_minicpm_batch(
    batch,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict:
    input_ids_list = []
    labels_list = []
    for item in batch:
        input_ids, labels = tokenize_chat_pair(
            tokenizer,
            item["question"],
            item["answer"],
            max_length=max_length,
        )
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    input_ids = _pad_sequences(input_ids_list, tokenizer.pad_token_id)
    labels = _pad_sequences(labels_list, -100)
    return {
        "video": torch.stack([item["video"] for item in batch]),
        "input_ids": input_ids,
        "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
        "labels": labels,
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "qa_type": [item["qa_type"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }


def make_dataloader(
    tokenizer: PreTrainedTokenizer,
    video_root: str | Path | None = None,
    split: str = "train",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_videos: bool = False,
    video_groups: Sequence[int] | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    img_size: int = IMG_SIZE,
    max_length: int = 256,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    tasks: Sequence[str] | None = None,
    skip_missing: bool = False,
) -> tuple[DataLoader, int]:
    if download_videos:
        downloaded_root = ensure_cuva_videos(cache_dir, groups=video_groups)
        if video_root is None:
            video_root = downloaded_root
    if video_root is None:
        video_root = Path(cache_dir) / "videos"

    samples = load_cuva_samples(split, cache_dir=cache_dir, tasks=tasks)
    samples = resolve_samples_to_videos(
        samples,
        build_video_index(video_root),
        skip_missing=skip_missing,
    )
    dataset = CUVADataset(samples, num_frames=num_frames, img_size=img_size)

    def collate_fn(batch):
        return _build_minicpm_batch(batch, tokenizer, max_length)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=split == "train",
    )
    return loader, len(loader)


def encode_user_prompt(
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids_list = []
    for question in questions:
        input_ids_list.append(
            apply_chat(
                tokenizer,
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
        )
    input_ids = _pad_sequences(input_ids_list, tokenizer.pad_token_id).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
    }


def _parse_groups(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(group.strip()) for group in value.split(",") if group.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download or inspect CUVA")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="train")
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download and extract the selected 7.5 GB CUVA video groups",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated archive groups (0-3); default downloads all",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download/inspect annotations without requiring videos",
    )
    args = parser.parse_args()

    samples = load_cuva_samples(args.split, cache_dir=args.cache_dir)
    print(f"CUVA {args.split} rows: {len(samples)}")
    print(
        "Example:",
        {
            key: samples[0][key]
            for key in ("id", "video_name", "qa_type", "question", "answer")
        },
    )
    if args.download_videos:
        root = ensure_cuva_videos(
            args.cache_dir,
            groups=_parse_groups(args.groups),
        )
        print(f"Videos extracted under: {root}")
    elif not args.annotations_only:
        print(
            "Annotations are ready. Add --download-videos to download the "
            "~25.6 GB video archives, optionally with --groups 0."
        )
