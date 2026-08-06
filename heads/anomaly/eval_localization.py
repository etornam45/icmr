"""Offline localization eval against native UCF temporal annotations (eval-only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dinov3.utils.device import get_device
from heads.anomaly.inference import (
    load_anomaly_label_maps,
    predict_from_videos,
    resolve_anomaly_checkpoint,
)
from heads.anomaly.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    IMG_SIZE,
    build_anomaly_model,
    load_anomaly_checkpoint,
)
from heads.anomaly.ucf_temporal import (
    find_temporal_annotation_file,
    frame_auc,
    frame_level_labels,
    mean_ap_at_tiou_range,
    parse_temporal_annotation_file,
    split_hyperparam_fold,
)
from heads.vqa.dataset import load_video_frames
from heads.vqa.vau_dataset import DEFAULT_CACHE_DIR, DEFAULT_UCF_DOWNLOAD_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WTAL localization on native UCF temporal annotations"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="Directory with ucf_* videos (default: data/VAU-Bench/videos)",
    )
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", choices=("holdout", "report", "all"), default="report")
    parser.add_argument("--nms-sigma", type=float, default=0.5)
    parser.add_argument("--deploy-threshold", type=float, default=0.0)
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all")
    return parser.parse_args()


def _resolve_video(video_root: Path, name: str) -> Path | None:
    candidates = [
        video_root / f"ucf_{name}",
        video_root / name,
        video_root / Path(name).name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def main():
    args = parse_args()
    device = get_device()

    ann_path = (
        Path(args.annotation)
        if args.annotation
        else find_temporal_annotation_file(
            [DEFAULT_UCF_DOWNLOAD_DIR, "data/UCF-Crime", DEFAULT_CACHE_DIR]
        )
    )
    if ann_path is None or not ann_path.exists():
        raise FileNotFoundError(
            "Temporal annotation file not found. Download UCF metadata via "
            "`python -m heads.vqa.vau_dataset --download-ucf` "
            "(includes Temporal_Anomaly_Annotation_for_Testing_Videos.txt)."
        )

    annotations = parse_temporal_annotation_file(ann_path)
    holdout, report = split_hyperparam_fold(
        annotations, holdout_frac=args.holdout_frac, seed=args.seed
    )
    if args.fold == "holdout":
        selected = holdout
    elif args.fold == "report":
        selected = report
    else:
        selected = annotations
    if args.max_videos > 0:
        selected = selected[: args.max_videos]

    video_root = Path(args.video_root or Path(DEFAULT_CACHE_DIR) / "videos")
    checkpoint_path = resolve_anomaly_checkpoint(args.checkpoint)
    label2id, id2label = load_anomaly_label_maps(checkpoint_path)
    normal_index = label2id.get("Normal", label2id.get("normal", 0))
    model = build_anomaly_model(
        device, num_classes=len(label2id), normal_index=normal_index
    )
    load_anomaly_checkpoint(model, checkpoint_path, device)
    model.eval()

    frame_scores: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    video_preds: dict[str, list[tuple[float, float, float]]] = {}
    video_gts: dict[str, list[tuple[float, float]]] = {}
    missing = 0

    for ann in tqdm(selected, desc=f"Eval ({args.fold})"):
        path = _resolve_video(video_root, ann.video_name)
        if path is None:
            missing += 1
            continue
        # Duration proxy: use last span end or a default from frame count.
        duration = max((e for _, e in ann.spans_sec), default=0.0)
        if duration <= 0:
            # Normal videos — sample a short window.
            duration = float(max(args.num_frames - 1, 1))

        frames = load_video_frames(
            path,
            num_frames=args.num_frames,
            img_size=IMG_SIZE,
            start_sec=0.0 if ann.spans_sec else None,
            end_sec=duration if ann.spans_sec else None,
        ).unsqueeze(0).to(device)

        result = predict_from_videos(
            model,
            frames,
            id2label,
            window_duration=duration,
            nms_sigma=args.nms_sigma,
            deploy_threshold=args.deploy_threshold,
        )
        actionness = result.get("actionness")
        if actionness is None:
            # Recompute via model if stripped
            out = model.forward_from_cls(model.encode_cls(frames))
            actionness = out["actionness"][0].detach().cpu().numpy()
        else:
            actionness = np.asarray(actionness, dtype=np.float64)

        labels = frame_level_labels(duration, ann.spans_sec, len(actionness))
        frame_scores.append(actionness)
        frame_labels.append(labels)

        if ann.spans_sec:
            video_gts[ann.video_name] = list(ann.spans_sec)
            video_preds[ann.video_name] = [
                (s["start"], s["end"], s["confidence"])
                for s in result.get("segments") or []
            ]

    if not frame_scores:
        raise RuntimeError(
            f"No evaluable videos (missing={missing}). Check --video-root={video_root}"
        )

    auc = frame_auc(np.concatenate(frame_scores), np.concatenate(frame_labels))
    maps = mean_ap_at_tiou_range(video_preds, video_gts)
    print(f"Fold: {args.fold}  videos={len(frame_scores)} missing={missing}")
    print(f"Frame AUC: {auc:.4f}")
    for key, value in maps.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
