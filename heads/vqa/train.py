import argparse
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from dinov3.utils.device import get_device
from heads.vqa.dataset import (
    DEFAULT_CACHE_DIR as CUVA_CACHE_DIR,
    encode_user_prompt,
    make_dataloader,
)
from heads.vqa.model import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_FRAMES,
    adapter_trainable_params,
    build_hybrid_model,
    decode_generated_answer,
    load_hybrid_checkpoint,
    save_hybrid_checkpoint,
    vision_adapter_params,
)
from heads.vqa.vau_dataset import (
    CAPTION_PROMPT,
    DEFAULT_CACHE_DIR as VAU_CACHE_DIR,
    make_vau_caption_dataloader,
)
from logger import SQLiteLogger

DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR
VAU_CHECKPOINT = "dinov3/checkpoints/model/vqa_vau_minicpm"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train DINOv3 spatial-pool + MiniCPM5-1B video VQA / captioning hybrid"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("cuva", "vau"),
        default="cuva",
        help="cuva: CUVA video QA; vau: VAU-Bench video-only Description",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="Video directory (default: <cache-dir>/videos)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Annotation cache dir (default: data/CUVA or data/VAU-Bench)",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download and extract CUVA video archives before training (~25.6 GB)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated CUVA task filter, e.g. Classification,Cause,Result",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Train/evaluate only rows whose videos exist under video-root",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "VAU only: comma-separated source filter (ucf, msad, ecva). "
            "Default for --dataset vau is ucf"
        ),
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Eval split (default: test for cuva, validation for vau)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--llm-lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument(
        "--adapter-lr", type=float, default=1e-4, help="Vision adapter learning rate"
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
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


def _resolve_defaults(args):
    if args.cache_dir is None:
        args.cache_dir = VAU_CACHE_DIR if args.dataset == "vau" else CUVA_CACHE_DIR
    if args.output is None:
        args.output = VAU_CHECKPOINT if args.dataset == "vau" else DEFAULT_CHECKPOINT
    if args.eval_split is None:
        args.eval_split = "validation" if args.dataset == "vau" else "test"
    if args.dataset == "vau" and args.download_videos:
        raise ValueError(
            "--download-videos is only supported for --dataset cuva. "
            "Download UCF videos with: python -m heads.vqa.vau_dataset --download-ucf"
        )
    if args.dataset == "vau" and args.tasks:
        raise ValueError("--tasks is only supported for --dataset cuva")
    if args.dataset == "cuva" and args.sources:
        raise ValueError("--sources is only supported for --dataset vau")
    if args.dataset == "vau" and args.sources is None:
        args.sources = "ucf"
    return args


def _make_loader(args, tokenizer, split: str, shuffle: bool):
    if args.dataset == "vau":
        return make_vau_caption_dataloader(
            tokenizer,
            video_root=args.video_root,
            split=split,
            cache_dir=args.cache_dir,
            skip_missing=args.skip_missing_videos,
            num_frames=args.num_frames,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sources=args.sources,
        )

    tasks = (
        [value.strip() for value in args.tasks.split(",") if value.strip()]
        if args.tasks
        else None
    )
    return make_dataloader(
        tokenizer,
        video_root=args.video_root,
        split=split,
        cache_dir=args.cache_dir,
        download_videos=args.download_videos and split == "train",
        tasks=tasks,
        skip_missing=args.skip_missing_videos,
        num_frames=args.num_frames,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    model.vision_model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        videos = batch["video"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(videos, input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            print("Warning: non-finite loss, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            1.0,
        )
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / num_batches:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_gen_samples: int = 8):
    model.eval()
    total_loss = 0.0
    references = []
    hypotheses = []
    questions = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        videos = batch["video"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(videos, input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        if len(references) < max_gen_samples:
            prompt = encode_user_prompt(tokenizer, [batch["question"][0]], device)
            prompt_len = prompt["input_ids"].shape[1]
            gen_ids = model.generate(
                videos[:1],
                prompt["input_ids"][:1],
                attention_mask=prompt["attention_mask"][:1],
                max_new_tokens=128,
                num_beams=1,
            )
            hypotheses.append(
                decode_generated_answer(
                    tokenizer,
                    gen_ids,
                    prompt_len,
                    num_visual_tokens=model.num_visual_tokens,
                )
            )
            references.append(batch["answer"][0])
            questions.append(batch["question"][0])

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, questions, references, hypotheses


def main():
    args = _resolve_defaults(parse_args())
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "vau":
        print(f"Caption prompt: {CAPTION_PROMPT!r}")

    model, tokenizer = build_hybrid_model(
        device,
        num_frames=args.num_frames,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    if args.resume:
        load_hybrid_checkpoint(model, args.resume, device, trainable_adapter=True)

    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()

    train_loader, _ = _make_loader(args, tokenizer, split="train", shuffle=True)
    print(f"Train batches: {len(train_loader)}")

    optimizer = optim.AdamW(
        [
            {
                "params": vision_adapter_params(model),
                "lr": args.adapter_lr,
            },
            {
                "params": adapter_trainable_params(model),
                "lr": args.llm_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output)
    best_dir = output_dir.parent / f"{output_dir.name}_best"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    logger = None
    if args.log_db:
        logger = SQLiteLogger(
            args.log_db,
            head="vqa",
            name=args.run_name,
            config=vars(args),
        )
        print(f"Logging run {logger.run_id} to {args.log_db}")

    try:
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)

            eval_loader, _ = _make_loader(
                args, tokenizer, split=args.eval_split, shuffle=False
            )
            eval_loss, questions, refs, hyps = evaluate(
                model, eval_loader, tokenizer, device
            )
            del eval_loader

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}"
            )
            if refs:
                print(f"  sample ref: {refs[0][:120]}...")
                print(f"  sample gen: {hyps[0][:120]}...")

            if logger is not None:
                logger.log_metrics(
                    {"train/loss": train_loss, "eval/loss": eval_loss},
                    epoch=epoch + 1,
                )
                if questions:
                    logger.log_records(
                        "eval_sample",
                        [
                            {
                                "question": question,
                                "reference": reference,
                                "prediction": prediction,
                            }
                            for question, reference, prediction in zip(
                                questions, refs, hyps
                            )
                        ],
                        epoch=epoch + 1,
                    )

            save_hybrid_checkpoint(model, str(output_dir))
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_hybrid_checkpoint(model, str(best_dir))
                print(f"  saved best checkpoint (eval_loss={eval_loss:.4f})")

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
