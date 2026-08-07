# ICMR — DINOv3 Multi-Head Video Understanding

Frozen [DINOv3](https://github.com/facebookresearch/dinov3) ViT backbone with
task heads for **object detection**, **anomaly classification**, and **video
captioning**. One shared encode produces patch/CLS tokens; heads never run the
backbone themselves.

Supports CUDA, Apple Silicon (MPS), and CPU (`dinov3.utils.device.get_device`).

## Architecture

```
frames [B,T,3,H,W]
        │
        ▼
 heads.backbone.encode_frames  →  patch_tokens / cls_tokens
        │
        ├── DETR          (COCO detection)
        ├── Anomaly       (VAU-Bench / UCF-Crime class labels)
        └── Caption       (VAU-Bench / UCF-Crime descriptions)
```

| Head | Input | Model | Data |
|------|--------|--------|------|
| **DETR** | last-frame patches | Deformable decoder | COCO 2017 |
| **Anomaly** | `T×P` patches | TransformerEncoder + MLP | VAU-Bench UCF labels |
| **Caption** | `T×P` patches | TransformerDecoder + BPE tokenizer | VAU-Bench UCF descriptions |

## Installation

```bash
pip install -e .
```

Optional extras used by training/logging: `tokenizers` (caption BPE).

## Quick start — backbone

```python
import torch
from heads.backbone import build_backbone, encode_frames
from dinov3.utils.device import get_device

device = get_device()
backbone = build_backbone(device)
videos = torch.rand(1, 16, 3, 224, 224, device=device)  # [0, 1]
feats = encode_frames(backbone, videos)
# feats["patch_tokens"]: [1, 16, 196, 384]
# feats["cls_tokens"]:   [1, 16, 384]
```

## Datasets

### COCO (DETR)

Downloaded automatically on first DETR train (or see `heads/detr/README.md`).

### VAU-Bench + UCF-Crime (anomaly & caption)

Annotations from [VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench);
videos from [`etornam/ufc-crime-videos`](https://huggingface.co/datasets/etornam/ufc-crime-videos):

```bash
python -m heads.caption.vau_dataset --annotations-only --split train
python -m heads.caption.vau_dataset --download-ucf --vau-only
python -m heads.caption.vau_dataset --verify-videos --sources ucf --split train
```

## Training

### DETR

```bash
python -m heads.detr.train
```

### Anomaly classifier

```bash
python -m heads.anomaly.train \
  --sources ucf \
  --skip-missing-videos \
  --epochs 10 \
  --batch-size 4 \
  --num-frames 16
```

Checkpoints: `dinov3/checkpoints/model/anomaly_vau/` (`classifier.pt` + `label2id.json`).

### Caption head

Trains (or loads) a BPE tokenizer on UCF descriptions, then the decoder:

```bash
python -m heads.caption.tokenizer --sources ucf   # optional; also auto-runs in train
python -m heads.caption.train \
  --sources ucf \
  --skip-missing-videos \
  --epochs 5 \
  --batch-size 4 \
  --num-frames 16
```

Checkpoints: `dinov3/checkpoints/model/caption/` (`caption_head.pt` + tokenizer/).

## Inference

```bash
# DETR (edit image_path in heads/detr/inference.py or call run_inference)
python -m heads.detr.inference

# Anomaly class
python -m heads.anomaly.inference --video path/to/video.mp4

# Caption
python -m heads.caption.inference --video path/to/video.mp4
```

## Live demo server

Shared backbone + DETR + optional anomaly/caption heads:

```bash
python -m server
```

Useful env vars: `ICMR_BACKBONE`, `ICMR_DETR`, `ICMR_ANOMALY`, `ICMR_CAPTION`,
`ICMR_ANOMALY_THRESHOLD`, `ICMR_HOST`, `ICMR_PORT`. See `server/config.py`.

## Repository layout

```
dinov3/                 # Vendored ViT + checkpoint helpers
heads/
  backbone.py           # Shared frozen encode (normalize → tokens)
  detr/                 # Detection head
  anomaly/              # Video anomaly classifier
  caption/              # Caption tokenizer + decoder (VAU/UCF)
server/                 # FastAPI live pipeline
logger/                 # SQLite experiment logging
```

## Smoke checks

```bash
python -m heads.caption.model
python -m heads.anomaly.model
python -m heads.caption.tokenizer --sources ucf
```

## Acknowledgments

DINOv3 backbone code is vendored from Meta's
[official DINOv3 repository](https://github.com/facebookresearch/dinov3)
under the DINOv3 License Agreement.
