"""Anomaly class inference for a single video."""

from __future__ import annotations

from pathlib import Path

import torch

from dinov3.utils.device import get_device
from heads.anomaly.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    IMG_SIZE,
    DINOv3AnomalyClassifier,
    build_anomaly_model,
    load_anomaly_checkpoint,
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


@torch.no_grad()
def predict_from_logits(
    logits: torch.Tensor,
    id2label: dict[int, str],
    top_k: int = 5,
) -> dict:
    """Convert classifier logits [B, C] or [C] into prediction + ranking."""
    if logits.dim() == 2:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)
    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    ranking = [
        {"class": id2label[int(index)], "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    return {"prediction": ranking[0]["class"], "top_k": ranking, "score": ranking[0]["probability"]}


@torch.no_grad()
def predict_from_cls(
    model: DINOv3AnomalyClassifier,
    cls_tokens: torch.Tensor,
    id2label: dict[int, str],
    top_k: int = 5,
) -> dict:
    """
    Classify from precomputed CLS tokens (shared-backbone path).

    Args:
        cls_tokens: [B, T, D] or [T, D]
    """
    if cls_tokens.dim() == 2:
        cls_tokens = cls_tokens.unsqueeze(0)
    pooled = cls_tokens.mean(dim=1)
    logits = model.head(model.norm(pooled))
    return predict_from_logits(logits, id2label, top_k=top_k)


@torch.no_grad()
def predict_from_videos(
    model: DINOv3AnomalyClassifier,
    videos: torch.Tensor,
    id2label: dict[int, str],
    top_k: int = 5,
) -> dict:
    """Classify a video tensor [B, T, 3, H, W] or [T, 3, H, W]."""
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    logits = model(videos)
    return predict_from_logits(logits, id2label, top_k=top_k)


def run_inference(
    video_path: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    start_sec: float | None = None,
    end_sec: float | None = None,
    top_k: int = 5,
    model: DINOv3AnomalyClassifier | None = None,
    id2label: dict[int, str] | None = None,
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
        model = build_anomaly_model(device, num_classes=len(label2id))
        load_anomaly_checkpoint(model, checkpoint_path, device)
        model.eval()

    frames = load_video_frames(
        video_path,
        num_frames=num_frames,
        img_size=IMG_SIZE,
        start_sec=start_sec,
        end_sec=end_sec,
    ).unsqueeze(0).to(device)

    result = predict_from_videos(model, frames, id2label, top_k=top_k)
    print(f"Predicted anomaly class: {result['prediction']}")
    for item in result["top_k"]:
        print(f"  {item['class']}: {item['probability']:.4f}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAU-Bench anomaly class inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
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
        )
