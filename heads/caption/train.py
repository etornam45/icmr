"""Train CaptionHead on VAU-Bench Description captions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from dinov3.utils.device import get_device
from heads.backbone import build_backbone, encode_frames
from heads.caption.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    build_caption_model,
    load_caption_checkpoint,
    save_caption_checkpoint,
)
from heads.caption.tokenizer import (
    DEFAULT_CACHE_DIR,
    DEFAULT_TOKENIZER_DIR,
    ensure_tokenizer,
)
from heads.caption.vau_dataset import make_vau_caption_dataloader
from logger import SQLiteLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 CaptionHead on VAU-Bench UCF-Crime descriptions"
    )
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Train only rows whose videos exist under video-root",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="ucf",
        help="Comma-separated source filter (default: ucf). Empty = all.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tokenizer-dir", type=str, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume weights from --output if caption_head.pt exists",
    )
    parser.add_argument("--log-db", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--force-retrain-tokenizer",
        action="store_true",
        help="Retrain the caption tokenizer even if tokenizer.json exists",
    )
    return parser.parse_args()


def _caption_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Cross-entropy on next-token targets, ignoring pad."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_token_id,
    )


def train_one_epoch(
    model,
    backbone,
    loader,
    optimizer,
    device: torch.device,
    pad_token_id: int,
    epoch: int,
) -> float:
    model.train()
    backbone.eval()
    total_loss = 0.0
    steps = 0
    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for batch in pbar:
        videos = batch["video"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            feats = encode_frames(backbone, videos)

        logits = model(
            feats["patch_tokens"],
            input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
        )
        loss = _caption_loss(logits, input_ids[:, 1:], pad_token_id)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(
    model,
    backbone,
    loader,
    device: torch.device,
    pad_token_id: int,
) -> float:
    model.eval()
    backbone.eval()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        videos = batch["video"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        feats = encode_frames(backbone, videos)
        logits = model(
            feats["patch_tokens"],
            input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
        )
        loss = _caption_loss(logits, input_ids[:, 1:], pad_token_id)
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def main():
    args = parse_args()
    device = get_device()
    sources = args.sources if args.sources else None

    tokenizer = ensure_tokenizer(
        args.tokenizer_dir,
        cache_dir=args.cache_dir,
        sources=sources,
        force_retrain=args.force_retrain_tokenizer,
    )

    backbone = build_backbone(device)
    model = build_caption_model(
        device,
        vocab_size=tokenizer.vocab_size,
        num_frames=args.num_frames,
        pad_token_id=tokenizer.pad_token_id,
    )

    output_dir = Path(args.output)
    if args.resume and (output_dir / "caption_head.pt").exists():
        load_caption_checkpoint(model, output_dir, device)
        print(f"Resumed caption head from {output_dir}")

    train_loader, n_train = make_vau_caption_dataloader(
        tokenizer,
        video_root=args.video_root,
        split="train",
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        skip_missing=args.skip_missing_videos,
        sources=sources,
    )
    if n_train == 0:
        raise SystemExit(
            "No training batches. Download UCF videos with:\n"
            "  python -m heads.caption.vau_dataset --download-ucf --vau-only"
        )

    eval_loader = None
    try:
        eval_loader, _ = make_vau_caption_dataloader(
            tokenizer,
            video_root=args.video_root,
            split="validation",
            cache_dir=args.cache_dir,
            num_frames=args.num_frames,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            skip_missing=args.skip_missing_videos,
            sources=sources,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: skipping validation loader ({exc})")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    logger = None
    if args.log_db:
        logger = SQLiteLogger(args.log_db, head="caption", name=args.run_name)

    best_eval = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            backbone,
            train_loader,
            optimizer,
            device,
            tokenizer.pad_token_id,
            epoch,
        )
        metrics = {"train/loss": train_loss}
        print(f"epoch {epoch}: train_loss={train_loss:.4f}")

        if eval_loader is not None:
            eval_loss = evaluate(
                model,
                backbone,
                eval_loader,
                device,
                tokenizer.pad_token_id,
            )
            metrics["eval/loss"] = eval_loss
            print(f"epoch {epoch}: eval_loss={eval_loss:.4f}")
            if eval_loss < best_eval:
                best_eval = eval_loss
                best_dir = output_dir.parent / f"{output_dir.name}_best"
                save_caption_checkpoint(
                    model, best_dir, tokenizer_dir=args.tokenizer_dir
                )
                print(f"  saved best checkpoint → {best_dir}")

        save_caption_checkpoint(
            model, output_dir, tokenizer_dir=args.tokenizer_dir
        )
        if logger is not None:
            logger.log_metrics(metrics, epoch=epoch)

    print(f"Done. Checkpoint saved to {output_dir}")
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
