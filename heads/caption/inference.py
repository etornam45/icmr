from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from dinov3.utils.device import get_device
from heads.backbone import IMG_SIZE, build_backbone, encode_frames
from heads.vqa.dataset import (
    DEFAULT_NUM_FRAMES,
    encode_user_prompt,
    load_video_frames,
)
from heads.vqa.model import (
    DEFAULT_CHECKPOINT_DIR,
    DINOv3QwenXAttn,
    build_hybrid_model,
    decode_generated_answer,
    load_hybrid_checkpoint,
)
from heads.vqa.vau_dataset import CAPTION_PROMPT

VAU_CHECKPOINT = "dinov3/checkpoints/model/vqa_vau_qwen"


def resolve_vqa_checkpoint(checkpoint_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    best_path = checkpoint_path.parent / f"{checkpoint_path.name}_best"
    if not checkpoint_path.exists() and best_path.exists():
        return best_path
    return checkpoint_path


@torch.no_grad()
def generate_caption(
    model: DINOv3QwenXAttn,
    tokenizer: PreTrainedTokenizer,
    videos: torch.Tensor,
    question: str | None = None,
    max_new_tokens: int = 128,
    backbone: nn.Module | None = None,
) -> str:
    """Caption a video tensor [B, T, 3, H, W] or [T, 3, H, W] with a loaded model."""
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    if question is None:
        question = CAPTION_PROMPT

    device = videos.device
    if backbone is None:
        backbone = build_backbone(device)
    feats = encode_frames(backbone, videos)

    prompt = encode_user_prompt(tokenizer, [question], device)
    prompt_len = prompt["input_ids"].shape[1]

    gen_ids = model.generate(
        feats["patch_tokens"],
        prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        cls_tokens=feats["cls_tokens"],
        max_new_tokens=max_new_tokens,
        num_beams=1,
    )
    return decode_generated_answer(tokenizer, gen_ids, prompt_len)


def run_inference(
    video_path: str,
    question: str | None = None,
    max_new_tokens: int = 128,
    num_frames: int = DEFAULT_NUM_FRAMES,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    start_sec: float | None = None,
    end_sec: float | None = None,
    model: DINOv3QwenXAttn | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    backbone: nn.Module | None = None,
) -> str:
    device = get_device()

    if question is None:
        question = CAPTION_PROMPT

    if model is None or tokenizer is None:
        checkpoint_path = resolve_vqa_checkpoint(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_dir}. Run python -m heads.vqa.train first."
            )
        model, tokenizer = build_hybrid_model(device, num_frames=num_frames)
        load_hybrid_checkpoint(
            model, str(checkpoint_path), device, trainable_adapter=False
        )
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

    answer = generate_caption(
        model,
        tokenizer,
        frames,
        question=question,
        max_new_tokens=max_new_tokens,
        backbone=backbone,
    )
    print(f"Generated answer: {answer}")
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video VQA / captioning inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video clip")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional question; defaults to the VAU caption prompt when omitted",
    )
    parser.add_argument(
        "--no-question",
        action="store_true",
        help="Force the default caption prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    args = parser.parse_args()

    question = None if args.no_question else args.question
    run_inference(
        video_path=args.video,
        question=question,
        max_new_tokens=args.max_new_tokens,
        num_frames=args.num_frames,
        checkpoint_dir=args.checkpoint,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
