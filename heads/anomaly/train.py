"""Train DINOv3 anomaly classifier on VAU-Bench Anomaly Class."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn, optim
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
from heads.vqa.vau_dataset import DEFAULT_CACHE_DIR, make_vau_class_dataloader
from logger import SQLiteLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 video anomaly classifier on VAU-Bench"
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="Directory of VAU-Bench videos (default: <cache-dir>/videos)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Local cache for VAU-Bench annotations",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Train/evaluate only rows whose videos exist under video-root",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="ucf",
        help="Comma-separated source filter (default: ucf). Use empty string for all.",
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--log-db",
        type=str,
        default=None,
        help="SQLite database path for training/eval metrics (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for the logged training run",
    )
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.vision_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            print("Warning: non-finite loss, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier_trainable_params(model), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / max(total, 1):.3f}",
        )

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # confusion counts: label -> pred -> count
    confusion: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)
        logits = model(videos)
        loss = criterion(logits, labels)

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
    # Empty string disables the default UCF-only filter.
    sources = args.sources if args.sources else None
    device = get_device()
    print(f"Using device: {device}")

    label2id, _id2label = build_train_label_maps(args.cache_dir, sources=sources)
    print(f"Anomaly classes ({len(label2id)}): {sorted(label2id)}")

    model = build_anomaly_model(
        device,
        num_classes=len(label2id),
        hidden_dim=args.hidden_dim,
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

    criterion = nn.CrossEntropyLoss()
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
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
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
            metrics = evaluate(model, eval_loader, criterion, device)
            del eval_loader

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"eval_loss={metrics['loss']:.4f}, "
                f"eval_acc={metrics['accuracy']:.3f}, "
                f"macro_f1={metrics['macro_f1']:.3f}"
            )

            if logger is not None:
                logger.log_metrics(
                    {
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
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
