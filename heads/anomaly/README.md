# DINOv3 Patch-Transformer Anomaly Classifier

Video-level anomaly classification on
[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) via
[VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench) labels.

Architecture (`ARCHITECTURE = patch_transformer_v1`):

- **Vision**: shared frozen DINOv3 via [`heads/backbone.py`](../backbone.py)
  (`encode_frames` → patch tokens); the anomaly head does **not** own a backbone
- **Positional encodings**: sinusoidal spatial (per patch) + temporal (per frame)
- **Aggregator**: TransformerEncoder over flattened `T × P` tokens
- **Head**: mean-pool → MLP → class logits
- **Train loss**: cross-entropy on video-level category labels

## Dataset setup

Same UCF staging as caption — annotations from VAU-Bench, videos from
[`etornam/ufc-crime-videos`](https://huggingface.co/datasets/etornam/ufc-crime-videos):

```bash
python -m heads.caption.vau_dataset --annotations-only --split train
python -m heads.caption.vau_dataset --download-ucf --vau-only
python -m heads.caption.vau_dataset --verify-videos --sources ucf --split train
```

Training defaults to `--sources ucf`.

## Train

```bash
python -m heads.anomaly.train \
  --sources ucf \
  --skip-missing-videos \
  --epochs 10 \
  --batch-size 4 \
  --num-frames 16
```

Checkpoints land in `dinov3/checkpoints/model/anomaly_vau/` (and
`anomaly_vau_best/`) as `classifier.pt` + `label2id.json`. Older WTAL
checkpoints are rejected — retrain is required.

## Inference

```bash
python -m heads.anomaly.inference \
  --video data/VAU-Bench/videos/ucf_Abuse001_x264.mp4
```

Returns video-level class prediction and top-k ranking.

## Server wiring

Live path shares the DINOv3 backbone with DETR/VQA, feeds patch tokens into
the anomaly Transformer, and gates events with `ICMR_ANOMALY_THRESHOLD`
(clip-level confidence). Optional class hint for VQA:

```bash
ICMR_VQA_CLASS_HINT=1
```

## Checks

```bash
python -m py_compile \
  heads/anomaly/model.py \
  heads/anomaly/train.py \
  heads/anomaly/inference.py \
  server/pipeline.py \
  server/runtime.py

python -m heads.anomaly.model
```
