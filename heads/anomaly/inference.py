"""Anomaly localization inference for a single video."""

from __future__ import annotations

from pathlib import Path

import torch

from dinov3.utils.device import get_device
from heads.anomaly.decode import Segment
from heads.anomaly.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    IMG_SIZE,
    DINOv3AnomalyLocalizer,
    build_anomaly_model,
    load_anomaly_checkpoint,
    localize_from_cls,
)
from heads.vqa.dataset import load_video_frames
from heads.vqa.vau_dataset import load_label_maps


def resolve_anomaly_checkpoint(checkpoint_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    best_path = checkpoint_path.parent / f"{checkpoint_path.name}_best"
    if not checkpoint_path.exists() and best_path.exists():
        return best_path
    return checkpoint_path


def load_anomaly_label_maps(checkpoint_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    label_path = checkpoint_path / "label2id.json"
    classifier_path = checkpoint_path / "classifier.pt"
    if label_path.exists():
        return load_label_maps(label_path)
    if classifier_path.exists():
        state = torch.load(classifier_path, map_location="cpu", weights_only=False)
        label2id = {str(k): int(v) for k, v in state["label2id"].items()}
        id2label = {index: name for name, index in label2id.items()}
        return label2id, id2label
    raise FileNotFoundError(
        f"No label2id.json or classifier.pt under {checkpoint_path}"
    )


def _segments_to_dicts(segments: list[Segment]) -> list[dict]:
    return [
        {
            "start": s.start,
            "end": s.end,
            "class": s.class_name,
            "class_id": s.class_id,
            "confidence": s.confidence,
        }
        for s in segments
    ]


@torch.no_grad()
def predict_from_logits(
    logits: torch.Tensor,
    id2label: dict[int, str],
    top_k: int = 5,
) -> dict:
    """Convert video-level logits [B, C] or [C] into prediction + ranking."""
    if logits.dim() == 2:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)
    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    ranking = [
        {"class": id2label[int(index)], "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    return {
        "prediction": ranking[0]["class"],
        "top_k": ranking,
        "score": ranking[0]["probability"],
    }


@torch.no_grad()
def predict_from_cls(
    model: DINOv3AnomalyLocalizer,
    cls_tokens: torch.Tensor,
    id2label: dict[int, str],
    top_k: int = 5,
    window_duration: float = 8.0,
    nms_sigma: float = 0.5,
    nms_floor: float = 0.05,
    deploy_threshold: float = 0.0,
) -> dict:
    """Classify + localize from precomputed CLS tokens (shared-backbone path)."""
    result = localize_from_cls(
        model,
        cls_tokens,
        id2label,
        window_duration=window_duration,
        nms_sigma=nms_sigma,
        nms_floor=nms_floor,
        deploy_threshold=deploy_threshold,
    )
    ranking = {
        "prediction": result["prediction"],
        "score": result["score"],
        "top_k": result["top_k"][:top_k],
    }
    result["prediction"] = ranking["prediction"]
    result["score"] = ranking["score"]
    result["top_k"] = ranking["top_k"]
    result["segments"] = _segments_to_dicts(result["segments"])
    if hasattr(result.get("actionness"), "numpy"):
        result["actionness"] = result["actionness"].numpy()
    # Prefer top segment confidence as anomaly score when available
    if result["segments"]:
        top_seg = result["segments"][0]
        result["segment"] = top_seg
        if top_seg["class"].lower() not in {"normal", "normal_videos"}:
            result["prediction"] = top_seg["class"]
            result["score"] = top_seg["confidence"]
    else:
        result["segment"] = None
    return result


@torch.no_grad()
def predict_from_videos(
    model: DINOv3AnomalyLocalizer,
    videos: torch.Tensor,
    id2label: dict[int, str],
    top_k: int = 5,
    window_duration: float | None = None,
    **decode_kwargs,
) -> dict:
    """Localize a video tensor [B, T, 3, H, W] or [T, 3, H, W]."""
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    cls = model.encode_cls(videos)
    if window_duration is None:
        # Unknown duration — treat timesteps as unit-spaced seconds.
        window_duration = float(max(cls.shape[1] - 1, 1))
    return predict_from_cls(
        model,
        cls,
        id2label,
        top_k=top_k,
        window_duration=window_duration,
        **decode_kwargs,
    )


def run_inference(
    video_path: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    start_sec: float | None = None,
    end_sec: float | None = None,
    top_k: int = 5,
    model: DINOv3AnomalyLocalizer | None = None,
    id2label: dict[int, str] | None = None,
    deploy_threshold: float = 0.0,
) -> dict:
    device = get_device()

    if model is None or id2label is None:
        checkpoint_path = resolve_anomaly_checkpoint(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_dir}. "
                "Run python -m heads.anomaly.train first."
            )
        label2id, id2label = load_anomaly_label_maps(checkpoint_path)
        normal_index = label2id.get("Normal", label2id.get("normal", 0))
        model = build_anomaly_model(
            device, num_classes=len(label2id), normal_index=normal_index
        )
        load_anomaly_checkpoint(model, checkpoint_path, device)
        model.eval()

    frames = load_video_frames(
        video_path,
        num_frames=num_frames,
        img_size=IMG_SIZE,
        start_sec=start_sec,
        end_sec=end_sec,
    ).unsqueeze(0).to(device)

    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        window_duration = float(end_sec - start_sec)
    else:
        window_duration = float(max(num_frames - 1, 1))

    result = predict_from_videos(
        model,
        frames,
        id2label,
        top_k=top_k,
        window_duration=window_duration,
        deploy_threshold=deploy_threshold,
    )
    print(f"Predicted anomaly class: {result['prediction']} ({result['score']:.4f})")
    for item in result["top_k"]:
        print(f"  {item['class']}: {item['probability']:.4f}")
    if result.get("segments"):
        print("Segments:")
        for seg in result["segments"][:10]:
            print(
                f"  [{seg['start']:.2f}, {seg['end']:.2f}] "
                f"{seg['class']} conf={seg['confidence']:.3f}"
            )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WTAL anomaly localization inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--deploy-threshold", type=float, default=0.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    best = checkpoint.parent / f"{checkpoint.name}_best"
    if not checkpoint.exists() and not best.exists():
        print(
            f"Checkpoint not found at {args.checkpoint}. "
            "Run python -m heads.anomaly.train first."
        )
    else:
        run_inference(
            video_path=args.video,
            checkpoint_dir=args.checkpoint,
            num_frames=args.num_frames,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            top_k=args.top_k,
            deploy_threshold=args.deploy_threshold,
        )
