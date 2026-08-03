# DINOv3 VAU-Bench Anomaly Classifier

Multi-class anomaly detection on
[7xiang/VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench):

- **Input**: video only (frames trimmed to `[Start Time, End Time]` when available)
- **Output**: `Anomaly Class` label (~20 classes including Normal)
- **Vision**: frozen DINOv3 ViT-S CLS token per sampled frame
- **Head**: LayerNorm + MLP over temporal mean-pooled CLS features

Video-only Description captioning lives under [`heads/vqa`](../vqa) with
`--dataset vau`.

## Dataset setup

Annotations come from Hugging Face. **UCF-Crime videos** come from
[`etornam/ufc-crime-videos`](https://huggingface.co/datasets/etornam/ufc-crime-videos)
(~57 GB with `--vau-only` for train+val; ~105 GB full) and are staged as
`ucf_*` under `data/VAU-Bench/videos/`. Training defaults to **UCF-only**
(`--sources ucf`).

```bash
# 1. Download annotations
python -m heads.vqa.vau_dataset --annotations-only --split train

# 2. Download only UCF videos named in VAU train+val (~57 GB) and stage as ucf_*
python -m heads.vqa.vau_dataset --download-ucf --vau-only
# Full mirror (~105 GB): omit --vau-only
# python -m heads.vqa.vau_dataset --download-ucf --stage-only  # if already downloaded

# 3. Verify UCF coverage
python -m heads.vqa.vau_dataset --verify-videos --sources ucf --split train
python -m heads.vqa.vau_dataset --verify-videos --sources ucf --split validation
```

Train with `--skip-missing-videos` so incomplete downloads are skipped.

Original UCF citation:
[UCF-Crime project page](https://www.crcv.ucf.edu/projects/real-world/).

When both `Start Time` and `End Time` are ≥ 0, the loader samples frames only
inside that window. A value of `-1` means the temporal annotation is unavailable
and the full video is used.

See [`heads/vqa/README.md`](../vqa/README.md#vau-bench-description) for the
shared layout diagram and download details.

## Train

Defaults to UCF-only:

```bash
python -m heads.anomaly.train \
  --sources ucf \
  --skip-missing-videos \
  --epochs 10 \
  --batch-size 4 \
  --num-frames 16
```

Custom video root:

```bash
python -m heads.anomaly.train \
  --video-root /path/to/VAU-Bench/videos \
  --sources ucf \
  --skip-missing-videos \
  --epochs 10
```

Checkpoints are written to `dinov3/checkpoints/model/anomaly_vau/` (and
`anomaly_vau_best/`), including `classifier.pt` and `label2id.json`.

## Inference

```bash
python -m heads.anomaly.inference \
  --video data/VAU-Bench/videos/ucf_Abuse001_x264.mp4 \
  --start-sec 23 \
  --end-sec 31
```

```python
from heads.anomaly.inference import run_inference

result = run_inference(
    video_path="data/VAU-Bench/videos/ucf_Abuse001_x264.mp4",
    start_sec=23,
    end_sec=31,
)
print(result["prediction"])
```

## Checks

```bash
python -m py_compile \
  heads/anomaly/dataset.py \
  heads/anomaly/model.py \
  heads/anomaly/train.py \
  heads/anomaly/inference.py \
  heads/vqa/vau_dataset.py
```
