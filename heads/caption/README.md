# Caption Head

Video captioning on **VAU-Bench** descriptions with **UCF-Crime** videos.
Uses shared DINOv3 patch tokens, a BPE `CaptionTokenizer`, and `CaptionHead`.

## Setup

```bash
python -m heads.caption.vau_dataset --annotations-only --split train
python -m heads.caption.vau_dataset --download-ucf --vau-only
python -m heads.caption.vau_dataset --verify-videos --sources ucf --split train
```

## Tokenizer

Trains once from UCF train-split descriptions (if missing):

```bash
python -m heads.caption.tokenizer --sources ucf
```

Default path: `dinov3/checkpoints/model/caption/tokenizer`.

## Train

```bash
python -m heads.caption.train \
  --sources ucf \
  --skip-missing-videos \
  --epochs 5 \
  --batch-size 4 \
  --num-frames 16 \
  --output dinov3/checkpoints/model/caption
```

## Inference

```bash
python -m heads.caption.inference \
  --video path/to/video.mp4 \
  --checkpoint dinov3/checkpoints/model/caption
```

## Checks

```bash
python -m heads.caption.model
python -m heads.caption.tokenizer --sources ucf
```
