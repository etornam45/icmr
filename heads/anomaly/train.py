"""Train DINOv3 patch-transformer anomaly classifier on VAU-Bench UCF labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from dinov3.utils.device import get_device
from heads.anomaly.dataset import build_train_label_maps
from heads.anomaly.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    build_anomaly_model,
    classifier_trainable_params,
    load_anomaly_checkpoint,
    save_anomaly_checkpoint,
)
from heads.backbone import build_backbone, encode_frames
from heads.vqa.vau_dataset import DEFAULT_CACHE_DIR, make_vau_class_dataloader
from logger import SQLiteLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 anomaly classifier on VAU-Bench (UCF)"
    )
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--skip-missing-videos", action="store_true")
    parser.add_argument(
        "--sources",
        type=str,
        default="ucf",
        help="Comma-separated source filter (default: ucf). Empty = all.",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        choices=("validation", "val", "test"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-db", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def train_epoch(model, backbone, loader, optimizer, device):
    model.train()
    backbone.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)
        feats = encode_frames(backbone, videos)

        optimizer.zero_grad()
        logits = model(feats["patch_tokens"])
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            print("Warning: non-finite loss, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier_trainable_params(model), 1.0)
        optimizer.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += n
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / max(total, 1):.3f}",
        )

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, backbone, loader, device):
    model.eval()
    backbone.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)
        feats = encode_frames(backbone, videos)
        logits = model(feats["patch_tokens"])
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for pred, label in zip(preds.tolist(), labels.tolist()):
            confusion[label][pred] += 1

    class_ids = sorted(confusion.keys())
    f1s = []
    recalls = []
    for class_id in class_ids:
        tp = confusion[class_id].get(class_id, 0)
        support = sum(confusion[class_id].values())
        predicted = sum(confusion[other].get(class_id, 0) for other in confusion)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)
        recalls.append(recall)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_recall": sum(recalls) / max(len(recalls), 1),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "num_samples": total,
    }


def main():
    args = parse_args()
    sources = args.sources if args.sources else None
    device = get_device()
    print(f"Using device: {device}")

    label2id, _id2label = build_train_label_maps(args.cache_dir, sources=sources)
    print(f"Anomaly classes ({len(label2id)}): {sorted(label2id)}")

    backbone = build_backbone(device)
    model = build_anomaly_model(
        device,
        num_classes=len(label2id),
        num_frames=args.num_frames,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    if args.resume:
        load_anomaly_checkpoint(model, args.resume, device)

    train_loader, _ = make_vau_class_dataloader(
        label2id,
        video_root=args.video_root,
        split="train",
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        skip_missing=args.skip_missing_videos,
        sources=sources,
    )
    print(f"Train batches: {len(train_loader)}")

    optimizer = optim.AdamW(
        classifier_trainable_params(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output)
    best_dir = output_dir.parent / f"{output_dir.name}_best"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    logger = None
    if args.log_db:
        logger = SQLiteLogger(
            args.log_db,
            head="anomaly",
            name=args.run_name,
            config=vars(args),
        )
        print(f"Logging run {logger.run_id} to {args.log_db}")

    try:
        for epoch in range(args.epochs):
            train_metrics = train_epoch(
                model, backbone, train_loader, optimizer, device
            )

            eval_loader, _ = make_vau_class_dataloader(
                label2id,
                video_root=args.video_root,
                split=args.eval_split,
                cache_dir=args.cache_dir,
                num_frames=args.num_frames,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                skip_missing=args.skip_missing_videos,
                sources=sources,
            )
            metrics = evaluate(model, backbone, eval_loader, device)
            del eval_loader

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.3f}, "
                f"eval_loss={metrics['loss']:.4f}, "
                f"eval_acc={metrics['accuracy']:.3f}, "
                f"macro_f1={metrics['macro_f1']:.3f}"
            )

            if logger is not None:
                logger.log_metrics(
                    {
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "eval/loss": metrics["loss"],
                        "eval/accuracy": metrics["accuracy"],
                        "eval/macro_f1": metrics["macro_f1"],
                        "eval/macro_recall": metrics["macro_recall"],
                    },
                    epoch=epoch + 1,
                )

            save_anomaly_checkpoint(model, output_dir, label2id)
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                save_anomaly_checkpoint(model, best_dir, label2id)
                print(f"  saved best checkpoint (eval_acc={best_acc:.3f})")

        if logger is not None:
            logger.finish(status="completed")
        print(f"Training complete. Checkpoints saved under {output_dir.parent}")
    except Exception:
        if logger is not None:
            logger.finish(status="failed")
        raise
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
