"""CaptionHead video captioning inference."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dinov3.utils.device import get_device
from heads.backbone import IMG_SIZE, build_backbone, encode_frames
from heads.caption.dataset import DEFAULT_NUM_FRAMES, load_video_frames
from heads.caption.model import (
    DEFAULT_CHECKPOINT_DIR,
    CaptionHead,
    build_caption_model,
    load_caption_checkpoint,
)
from heads.caption.tokenizer import (
    DEFAULT_TOKENIZER_DIR,
    CaptionTokenizer,
    ensure_tokenizer,
)


def resolve_caption_checkpoint(checkpoint_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    best_path = checkpoint_path.parent / f"{checkpoint_path.name}_best"
    if not checkpoint_path.exists() and best_path.exists():
        return best_path
    return checkpoint_path


# Back-compat alias for callers that still use the old name.
resolve_vqa_checkpoint = resolve_caption_checkpoint


def _peek_caption_hyperparams(checkpoint_dir: Path) -> dict:
    ckpt_path = checkpoint_dir / "caption_head.pt"
    if not ckpt_path.exists():
        return {}
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "num_frames": int(state.get("num_frames", DEFAULT_NUM_FRAMES)),
        "vocab_size": int(state.get("vocab_size", 0)) or None,
        "pad_token_id": int(state.get("pad_token_id", 0)),
        "tokenizer_dir": state.get("tokenizer_dir"),
    }


@torch.no_grad()
def generate_caption(
    model: CaptionHead,
    tokenizer: CaptionTokenizer,
    videos: torch.Tensor,
    max_new_tokens: int = 64,
    backbone: nn.Module | None = None,
    **_unused,
) -> str:
    """Caption a video tensor [B, T, 3, H, W] or [T, 3, H, W]."""
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)

    if backbone is None:
        backbone = build_backbone(videos.device)
    feats = encode_frames(backbone, videos)
    token_ids = model.generate(
        feats["patch_tokens"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(token_ids[0])


def run_inference(
    video_path: str,
    max_new_tokens: int = 64,
    num_frames: int = DEFAULT_NUM_FRAMES,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    tokenizer_dir: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model: CaptionHead | None = None,
    tokenizer: CaptionTokenizer | None = None,
    backbone: nn.Module | None = None,
) -> str:
    device = get_device()
    checkpoint_path = resolve_caption_checkpoint(checkpoint_dir)

    if model is None or tokenizer is None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_dir}. "
                "Run python -m heads.caption.train first."
            )
        hyper = _peek_caption_hyperparams(checkpoint_path)
        num_frames = hyper.get("num_frames") or num_frames
        tok_dir = tokenizer_dir or hyper.get("tokenizer_dir") or DEFAULT_TOKENIZER_DIR
        tokenizer = ensure_tokenizer(tok_dir)
        model = build_caption_model(
            device,
            vocab_size=tokenizer.vocab_size,
            num_frames=num_frames,
            pad_token_id=tokenizer.pad_token_id,
        )
        load_caption_checkpoint(model, checkpoint_path, device)
        model.eval()

    if backbone is None:
        backbone = build_backbone(device)

    frames = load_video_frames(
        video_path,
        num_frames=num_frames,
        img_size=IMG_SIZE,
        start_sec=start_sec,
        end_sec=end_sec,
    ).unsqueeze(0).to(device)

    caption = generate_caption(
        model,
        tokenizer,
        frames,
        max_new_tokens=max_new_tokens,
        backbone=backbone,
    )
    print(f"Generated caption: {caption}")
    return caption


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video captioning inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video clip")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tokenizer-dir", type=str, default=None)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        max_new_tokens=args.max_new_tokens,
        num_frames=args.num_frames,
        checkpoint_dir=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
