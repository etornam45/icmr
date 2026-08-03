"""Anomaly class inference for a single video."""

from __future__ import annotations

from pathlib import Path

import torch

from dinov3.utils.device import get_device
from heads.anomaly.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    IMG_SIZE,
    build_anomaly_model,
    load_anomaly_checkpoint,
)
from heads.vqa.dataset import load_video_frames
from heads.vqa.vau_dataset import load_label_maps


def run_inference(
    video_path: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    num_frames: int = DEFAULT_NUM_FRAMES,
    start_sec: float | None = None,
    end_sec: float | None = None,
    top_k: int = 5,
) -> dict:
    device = get_device()

    checkpoint_path = Path(checkpoint_dir)
    best_path = checkpoint_path.parent / f"{checkpoint_path.name}_best"
    if not checkpoint_path.exists() and best_path.exists():
        checkpoint_path = best_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_dir}. "
            "Run python -m heads.anomaly.train first."
        )

    label_path = checkpoint_path / "label2id.json"
    classifier_path = checkpoint_path / "classifier.pt"
    if label_path.exists():
        label2id, id2label = load_label_maps(label_path)
    elif classifier_path.exists():
        state = torch.load(classifier_path, map_location="cpu", weights_only=False)
        label2id = {str(k): int(v) for k, v in state["label2id"].items()}
        id2label = {index: name for name, index in label2id.items()}
    else:
        raise FileNotFoundError(
            f"No label2id.json or classifier.pt under {checkpoint_path}"
        )

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

    with torch.no_grad():
        logits = model(frames)
        probs = torch.softmax(logits, dim=-1)[0]

    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    ranking = [
        {"class": id2label[int(index)], "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    pred = ranking[0]["class"]
    print(f"Predicted anomaly class: {pred}")
    for item in ranking:
        print(f"  {item['class']}: {item['probability']:.4f}")
    return {"prediction": pred, "top_k": ranking}


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
