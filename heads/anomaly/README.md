# DINOv3 WTAL Anomaly Localizer

Weakly-supervised temporal anomaly localization on
[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) via
[VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench) video-level labels.

Architecture (`ARCHITECTURE = temporal_wtal_v1`):

- **Vision**: frozen DINOv3 ViT-S CLS token per sampled frame
- **Temporal aggregator**: residual 1D conv stack over the frame sequence
- **Feature pyramid**: strides `{1, 2, 4, 8}`
- **Classification head**: shared MLP → class activation sequence (CAS)
- **Boundary head**: stubbed in-code, **inactive** (no span-labeled train data)
- **Train losses**: top-k MIL (video-level category) + binary MIL ranking + SVDD
- **Inference**: multi-threshold proposals + OIC scoring + soft-NMS → segments

Native `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` is **eval-only**
(frame AUC / tIoU mAP). It never enters the training loss.

## Dataset setup

Same UCF staging as before — annotations from VAU-Bench, videos from
[`etornam/ufc-crime-videos`](https://huggingface.co/datasets/etornam/ufc-crime-videos):

```bash
python -m heads.vqa.vau_dataset --annotations-only --split train
python -m heads.vqa.vau_dataset --download-ucf --vau-only
python -m heads.vqa.vau_dataset --verify-videos --sources ucf --split train
```

Training defaults to `--sources ucf`. VAU UCF train/val have **no** temporal
spans; supervision is video-level category + normal/abnormal bags only.

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
`anomaly_vau_best/`) as `classifier.pt` + `label2id.json`. Legacy mean-pool
checkpoints are rejected — retrain is required.

Loss weights (CLI): `--lambda-topk 1.0 --lambda-mil 1.0 --lambda-svdd 0.1`.

## Inference

```bash
python -m heads.anomaly.inference \
  --video data/VAU-Bench/videos/ucf_Abuse001_x264.mp4 \
  --deploy-threshold 0.15
```

Returns video-level class plus decoded segments `{start, end, class, confidence}`.

## Server wiring

Live path uses segment confidence ≥ `deploy_threshold` (`ICMR_DEPLOY_THRESHOLD`)
to emit an `AnomalyEvent` with `start_ts` / `end_ts` / `svdd_score`, then samples
the ring buffer in `[start - pad, end + pad]` for VQA. Optional class hint:

```bash
ICMR_VQA_CLASS_HINT=1
```

## Checks

```bash
python -m py_compile \
  heads/anomaly/temporal.py \
  heads/anomaly/pyramid.py \
  heads/anomaly/heads.py \
  heads/anomaly/losses.py \
  heads/anomaly/decode.py \
  heads/anomaly/ucf_temporal.py \
  heads/anomaly/model.py \
  heads/anomaly/train.py \
  heads/anomaly/inference.py \
  server/pipeline.py \
  server/monitor.py \
  server/events.py \
  server/config.py \
  server/runtime.py

python -m heads.anomaly.model
```
