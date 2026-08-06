"""Native UCF-Crime temporal annotations — eval only (never used in training loss)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Official UCF-Crime videos are annotated at 30 fps.
UCF_ANNOTATION_FPS = 30.0

CLASS_ALIASES = {
    "RoadAccidents": "Traffic_accident",
    "roadaccidents": "Traffic_accident",
}

DEFAULT_ANNOTATION_NAMES = (
    "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    "Temporal_Anomaly_Annotation.txt",
)


@dataclass
class TemporalAnnotation:
    video_name: str
    anomaly_class: str
    spans_sec: list[tuple[float, float]]  # possibly empty for Normal


def normalize_class_name(name: str) -> str:
    return CLASS_ALIASES.get(name, CLASS_ALIASES.get(name.casefold(), name))


def find_temporal_annotation_file(
    search_roots: list[str | Path] | None = None,
) -> Path | None:
    roots = search_roots or [
        "data/UCF-Crime/hf",
        "data/UCF-Crime",
        "data/VAU-Bench",
    ]
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for name in DEFAULT_ANNOTATION_NAMES:
            direct = root_path / name
            if direct.is_file():
                return direct
        for path in root_path.rglob("Temporal_Anomaly_Annotation*.txt"):
            if path.is_file():
                return path
    return None


def parse_temporal_annotation_file(
    path: str | Path,
    fps: float = UCF_ANNOTATION_FPS,
) -> list[TemporalAnnotation]:
    """Parse native UCF test temporal annotations.

    Format per line:
      VideoName Class Start1 End1 Start2 End2
    Frame indices at ``fps``; negative means unused.
    """
    path = Path(path)
    rows: list[TemporalAnnotation] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        video_name = parts[0]
        anomaly_class = normalize_class_name(parts[1])
        frames = [int(float(x)) for x in parts[2:6]]
        spans: list[tuple[float, float]] = []
        for i in range(0, len(frames), 2):
            start_f, end_f = frames[i], frames[i + 1]
            if start_f < 0 or end_f < 0 or end_f <= start_f:
                continue
            spans.append((start_f / fps, end_f / fps))
        rows.append(
            TemporalAnnotation(
                video_name=video_name,
                anomaly_class=anomaly_class,
                spans_sec=spans,
            )
        )
    return rows


def split_hyperparam_fold(
    annotations: list[TemporalAnnotation],
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[TemporalAnnotation], list[TemporalAnnotation]]:
    """Split anomaly-annotated videos into hyperparam vs report folds.

    Normal videos (no spans) are attached to the report fold only so they
    never influence threshold tuning.
    """
    anom = [a for a in annotations if a.spans_sec]
    normals = [a for a in annotations if not a.spans_sec]
    rng = random.Random(seed)
    order = list(anom)
    rng.shuffle(order)
    n_hold = max(1, int(round(len(order) * holdout_frac))) if order else 0
    holdout = order[:n_hold]
    report = order[n_hold:] + normals
    return holdout, report


def frame_level_labels(
    duration_sec: float,
    spans_sec: list[tuple[float, float]],
    num_frames: int,
) -> np.ndarray:
    """Binary labels [T] for uniformly sampled frames over [0, duration]."""
    labels = np.zeros(num_frames, dtype=np.float32)
    if duration_sec <= 0 or num_frames <= 0:
        return labels
    for i in range(num_frames):
        t = 0.0 if num_frames == 1 else i / (num_frames - 1) * duration_sec
        for start, end in spans_sec:
            if start <= t <= end:
                labels[i] = 1.0
                break
    return labels


def frame_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """ROC AUC for binary frame scores (sklearn-free)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if scores.size == 0 or labels.max() == labels.min():
        return float("nan")
    # Rank-based AUC
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos = labels > 0.5
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[pos].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def segment_tiou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
    return inter / union if union > 0 else 0.0


def average_precision_at_tiou(
    predictions: list[tuple[float, float, float]],
    ground_truths: list[tuple[float, float]],
    tiou_threshold: float,
) -> float:
    """AP for one video (or pooled) at a single tIoU threshold.

    predictions: list of (start, end, score) sorted later by score.
    """
    if not ground_truths:
        return 0.0 if predictions else float("nan")
    if not predictions:
        return 0.0

    preds = sorted(predictions, key=lambda x: x[2], reverse=True)
    matched = [False] * len(ground_truths)
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)
    for i, (ps, pe, _score) in enumerate(preds):
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(ground_truths):
            if matched[j]:
                continue
            iou = segment_tiou((ps, pe), gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= tiou_threshold and best_j >= 0:
            tp[i] = 1.0
            matched[best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / len(ground_truths)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        ap += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 11.0


def mean_ap_at_tiou_range(
    video_predictions: dict[str, list[tuple[float, float, float]]],
    video_ground_truths: dict[str, list[tuple[float, float]]],
    tiou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Pool AP across videos at each tIoU, return per-threshold and mean."""
    if tiou_thresholds is None:
        tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    per_t: dict[str, float] = {}
    for thr in tiou_thresholds:
        aps = []
        for vid, gts in video_ground_truths.items():
            preds = video_predictions.get(vid, [])
            if not gts:
                continue
            aps.append(average_precision_at_tiou(preds, gts, thr))
        per_t[f"mAP@{thr:.1f}"] = float(np.nanmean(aps)) if aps else float("nan")
    vals = [v for v in per_t.values() if not math.isnan(v)]
    per_t["mAP@0.3:0.7"] = float(np.mean(vals)) if vals else float("nan")
    return per_t
