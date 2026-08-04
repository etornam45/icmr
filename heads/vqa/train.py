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
    DINOv3MiniCPMHybrid,
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
    parser.add_argument(
        "--adapter-epochs",
        type=int,
        default=2,
        help="Stage 1: train vision adapter only with LoRA frozen (0 to skip)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Stage 2: joint vision adapter + LoRA epochs",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument(
        "--llm-lr",
        type=float,
        default=5e-5,
        help="LoRA learning rate (joint stage)",
    )
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=5e-4,
        help="Vision adapter learning rate",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stop after this many joint-stage evals without improvement "
        "(0 disables)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=4,
        help="Number of spaced eval generations to print / log per epoch",
    )
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


def _set_lora_requires_grad(model: DINOv3MiniCPMHybrid, trainable: bool) -> int:
    """Enable/disable grads on LoRA parameters. Returns how many were toggled."""
    count = 0
    for name, param in model.llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = trainable
            count += 1
    return count


def build_optimizer(
    model: DINOv3MiniCPMHybrid,
    adapter_lr: float,
    llm_lr: float | None,
    weight_decay: float,
    include_lora: bool,
):
    groups = [
        {
            "params": vision_adapter_params(model),
            "lr": adapter_lr,
        }
    ]
    if include_lora:
        lora_params = adapter_trainable_params(model)
        if not lora_params:
            raise RuntimeError("No trainable LoRA parameters found for joint stage")
        groups.append({"params": lora_params, "lr": llm_lr})
    return optim.AdamW(groups, weight_decay=weight_decay)


def train_epoch(model, loader, optimizer, device, desc: str = "Training"):
    model.train()
    model.vision_model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
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


def _gen_sample_batch_indices(num_batches: int, max_samples: int) -> set[int]:
    """Evenly spaced batch indices so eval gens cover different clips."""
    if num_batches <= 0 or max_samples <= 0:
        return set()
    count = min(max_samples, num_batches)
    if count == 1:
        return {0}
    return {round(i * (num_batches - 1) / (count - 1)) for i in range(count)}


@torch.no_grad()
def evaluate(
    model,
    loader,
    tokenizer,
    device,
    max_gen_samples: int = 4,
):
    model.eval()
    total_loss = 0.0
    references = []
    hypotheses = []
    questions = []
    clip_ids = []

    sample_batches = _gen_sample_batch_indices(len(loader), max_gen_samples)

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        videos = batch["video"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(videos, input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        if batch_idx in sample_batches:
            prompt = encode_user_prompt(tokenizer, [batch["question"][0]], device)
            prompt_len = prompt["input_ids"].shape[1]
            gen_ids = model.generate(
                videos[:1],
                prompt["input_ids"][:1],
                attention_mask=prompt["attention_mask"][:1],
                max_new_tokens=128,
                num_beams=1,
            )
            num_visual = videos.shape[1] * model.tokens_per_frame
            hypotheses.append(
                decode_generated_answer(
                    tokenizer,
                    gen_ids,
                    prompt_len,
                    num_visual_tokens=num_visual,
                )
            )
            references.append(batch["answer"][0])
            questions.append(batch["question"][0])
            clip = batch.get("clip_id", batch.get("sample_id", [""]))[0]
            clip_ids.append(clip)

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, questions, references, hypotheses, clip_ids


def _print_eval_samples(refs, hyps, clip_ids, limit: int = 4):
    for index, (ref, hyp) in enumerate(zip(refs, hyps)):
        if index >= limit:
            break
        clip = clip_ids[index] if index < len(clip_ids) else ""
        tag = f" [{clip}]" if clip else ""
        print(f"  sample{index}{tag}")
        print(f"    ref: {ref[:140]}{'...' if len(ref) > 140 else ''}")
        print(f"    gen: {hyp[:140]}{'...' if len(hyp) > 140 else ''}")


def run_eval_epoch(
    args,
    model,
    tokenizer,
    device,
    logger,
    epoch_label: str,
    epoch_num: int,
    best_loss: float,
    output_dir: Path,
    best_dir: Path,
):
    eval_loader, _ = _make_loader(
        args, tokenizer, split=args.eval_split, shuffle=False
    )
    eval_loss, questions, refs, hyps, clip_ids = evaluate(
        model,
        eval_loader,
        tokenizer,
        device,
        max_gen_samples=args.eval_samples,
    )
    del eval_loader

    print(f"{epoch_label}: eval_loss={eval_loss:.4f}")
    if refs:
        _print_eval_samples(refs, hyps, clip_ids, limit=args.eval_samples)

    if logger is not None:
        logger.log_metrics({"eval/loss": eval_loss}, epoch=epoch_num)
        if questions:
            logger.log_records(
                "eval_sample",
                [
                    {
                        "clip_id": clip_id,
                        "question": question,
                        "reference": reference,
                        "prediction": prediction,
                    }
                    for question, reference, prediction, clip_id in zip(
                        questions, refs, hyps, clip_ids
                    )
                ],
                epoch=epoch_num,
            )

    save_hybrid_checkpoint(model, str(output_dir))
    improved = eval_loss < best_loss
    if improved:
        best_loss = eval_loss
        save_hybrid_checkpoint(model, str(best_dir))
        print(f"  saved best checkpoint (eval_loss={eval_loss:.4f})")
    return best_loss, improved


def main():
    args = _resolve_defaults(parse_args())
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(
        f"Schedule: adapter-only={args.adapter_epochs} epoch(s), "
        f"joint={args.epochs} epoch(s), "
        f"adapter_lr={args.adapter_lr}, llm_lr={args.llm_lr}, "
        f"patience={args.patience}"
    )
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

    global_epoch = 0
    try:
        # --- Stage 1: vision adapter only ---
        if args.adapter_epochs > 0:
            n_lora = _set_lora_requires_grad(model, trainable=False)
            print(
                f"Stage 1: vision adapter only "
                f"({args.adapter_epochs} epoch(s), LoRA frozen={n_lora} tensors)"
            )
            optimizer = build_optimizer(
                model,
                adapter_lr=args.adapter_lr,
                llm_lr=None,
                weight_decay=args.weight_decay,
                include_lora=False,
            )
            for stage_epoch in range(args.adapter_epochs):
                global_epoch += 1
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    desc=f"Adapter {stage_epoch + 1}/{args.adapter_epochs}",
                )
                print(
                    f"Adapter epoch {stage_epoch + 1}/{args.adapter_epochs}: "
                    f"train_loss={train_loss:.4f}"
                )
                if logger is not None:
                    logger.log_metrics(
                        {"train/loss": train_loss, "stage": 1},
                        epoch=global_epoch,
                    )
                best_loss, _ = run_eval_epoch(
                    args,
                    model,
                    tokenizer,
                    device,
                    logger,
                    epoch_label=(
                        f"Adapter epoch {stage_epoch + 1}/{args.adapter_epochs}"
                    ),
                    epoch_num=global_epoch,
                    best_loss=best_loss,
                    output_dir=output_dir,
                    best_dir=best_dir,
                )

        # --- Stage 2: joint LoRA + adapter ---
        if args.epochs > 0:
            n_lora = _set_lora_requires_grad(model, trainable=True)
            print(
                f"Stage 2: joint adapter + LoRA "
                f"({args.epochs} epoch(s), LoRA trainable={n_lora} tensors)"
            )
            optimizer = build_optimizer(
                model,
                adapter_lr=args.adapter_lr,
                llm_lr=args.llm_lr,
                weight_decay=args.weight_decay,
                include_lora=True,
            )
            stale = 0
            for stage_epoch in range(args.epochs):
                global_epoch += 1
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    desc=f"Joint {stage_epoch + 1}/{args.epochs}",
                )
                print(
                    f"Joint epoch {stage_epoch + 1}/{args.epochs}: "
                    f"train_loss={train_loss:.4f}"
                )
                if logger is not None:
                    logger.log_metrics(
                        {"train/loss": train_loss, "stage": 2},
                        epoch=global_epoch,
                    )
                best_loss, improved = run_eval_epoch(
                    args,
                    model,
                    tokenizer,
                    device,
                    logger,
                    epoch_label=f"Joint epoch {stage_epoch + 1}/{args.epochs}",
                    epoch_num=global_epoch,
                    best_loss=best_loss,
                    output_dir=output_dir,
                    best_dir=best_dir,
                )
                if args.patience > 0:
                    if improved:
                        stale = 0
                    else:
                        stale += 1
                        print(
                            f"  no eval improvement ({stale}/{args.patience})"
                        )
                        if stale >= args.patience:
                            print("Early stopping (patience exhausted)")
                            break

        if logger is not None:
            logger.finish(status="completed")
        print(
            f"Training complete. Best eval_loss={best_loss:.4f}. "
            f"Checkpoints under {output_dir.parent}"
        )
    except Exception:
        if logger is not None:
            logger.finish(status="failed")
        raise
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
