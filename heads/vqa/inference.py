from pathlib import Path

import torch

from dinov3.utils.device import get_device
from heads.vqa.dataset import (
    DEFAULT_NUM_FRAMES,
    encode_user_prompt,
    load_video_frames,
)
from heads.vqa.model import (
    DEFAULT_CHECKPOINT_DIR,
    IMG_SIZE,
    build_hybrid_model,
    decode_generated_answer,
    load_hybrid_checkpoint,
)


def run_inference(
    video_path: str,
    question: str,
    max_new_tokens: int = 128,
    num_frames: int = DEFAULT_NUM_FRAMES,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> str:
    device = get_device()

    checkpoint_path = Path(checkpoint_dir)
    best_path = checkpoint_path.parent / f"{checkpoint_path.name}_best"
    if not checkpoint_path.exists() and best_path.exists():
        checkpoint_path = best_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_dir}. Run python -m heads.vqa.train first."
        )

    model, tokenizer = build_hybrid_model(device, num_frames=num_frames)
    load_hybrid_checkpoint(model, str(checkpoint_path), device, trainable_adapter=False)
    model.eval()

    frames = load_video_frames(
        video_path, num_frames=num_frames, img_size=IMG_SIZE
    ).unsqueeze(0).to(device)

    prompt = encode_user_prompt(tokenizer, [question], device)
    prompt_len = prompt["input_ids"].shape[1]

    gen_ids = model.generate(
        frames,
        prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=1,
    )

    answer = decode_generated_answer(
        tokenizer,
        gen_ids,
        prompt_len,
        num_visual_tokens=model.num_visual_tokens,
    )
    print(f"Generated answer: {answer}")
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video VQA inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video clip")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    best = checkpoint.parent / f"{checkpoint.name}_best"
    if not checkpoint.exists() and not best.exists():
        print(
            f"Checkpoint not found at {args.checkpoint}. "
            "Run python -m heads.vqa.train first."
        )
    else:
        run_inference(
            video_path=args.video,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            num_frames=args.num_frames,
            checkpoint_dir=args.checkpoint,
        )
