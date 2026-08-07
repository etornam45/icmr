# DINOv3 + Qwen2.5-1.5B Video VQA / Captioning (Flamingo-style)

Supports two datasets:

1. **CUVA** ([fesvhtr/CUVA](https://huggingface.co/datasets/fesvhtr/CUVA)) —
   video question answering (question + answer)
2. **VAU-Bench** ([7xiang/VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench)) —
   video-only Description captioning (no dataset question)

Architecture (`video_xattn_v1`):

- **Vision**: frozen DINOv3 ViT-S — full per-frame patch tokens (+ CLS)
- **Resampler**: Perceiver with 64 learned queries, 3 layers, learned temporal
  embeddings + 2D sincos spatial PE
- **Language**: frozen
  [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  with Flamingo-style **tanh-gated cross-attention** inserted every 4th decoder
  layer (gates init at 0); optional LoRA in training stage 2
- Visual tokens stay in the cross-attn pathway only (no prefix-concat into the
  LM context)

Default: 16 frames → **64** visual latents. Older `video_cls` /
`video_spatial*` / MiniCPM prefix-concat checkpoints are incompatible — retrain
required.

---

## CUVA Video QA

CUVA tasks: Detection, Classification, Cause, Result, Timestamp, and Description.

### Dataset

CUVA provides annotations and original videos in the same Hugging Face
repository. Its schema is:

| Column | Use |
| --- | --- |
| `instruction` | Question/instruction |
| `visual_input` | Video filename |
| `output` | Target answer |
| `task` | Task type |
| `ID` | Sample identifier |

The repository provides `full` and `test` splits. Upstream `full` includes
older rows for the test videos under different IDs, as well as the test
records themselves. This loader defines a leakage-safe `train` split by
removing every `visual_input` found in `test`. You can still request the
original `full` split explicitly.

The dataset is about 25.6 GB. CUVA is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
commercial use is not permitted and derivatives must use the same license.

### Download

#### Recommended: project helper

Download annotations only (small parquet files):

```bash
python -m heads.vqa.dataset --annotations-only --split train
python -m heads.vqa.dataset --annotations-only --split test
```

Download and extract all four video archives:

```bash
python -m heads.vqa.dataset --download-videos
```

The four archives are approximately 7.6 GB, 7.6 GB, 7.7 GB, and 2.7 GB.
Download one group at a time if disk space or network reliability is limited:

```bash
python -m heads.vqa.dataset --download-videos --groups 0
python -m heads.vqa.dataset --download-videos --groups 1
python -m heads.vqa.dataset --download-videos --groups 2
python -m heads.vqa.dataset --download-videos --groups 3
```

You can train on only the groups already extracted. Missing-video rows must be
skipped explicitly:

```bash
python -m heads.vqa.train \
  --video-root data/CUVA/videos \
  --skip-missing-videos \
  --epochs 5
```

Delete archives after extraction if needed:

```bash
rm data/CUVA/raw/group_{0,1,2,3}.zip
```

By default, files are placed under:

```text
data/CUVA/
  data/all.parquet
  data/test.parquet
  raw/group_0.zip
  raw/group_1.zip
  raw/group_2.zip
  raw/group_3.zip
  videos/
    .../00001.mp4
    .../00002.mp4
```

### Train (CUVA)

```bash
python -m heads.vqa.train \
  --dataset cuva \
  --adapter-epochs 2 \
  --epochs 5 \
  --batch-size 2 \
  --num-frames 16 \
  --adapter-lr 1e-4 \
  --llm-lr 1e-5
```

Download videos automatically before training:

```bash
python -m heads.vqa.train --dataset cuva --download-videos --epochs 5
```

Train only selected tasks:

```bash
python -m heads.vqa.train \
  --dataset cuva \
  --tasks Classification,Cause,Result \
  --epochs 5
```

Checkpoints are saved under
`dinov3/checkpoints/model/vqa_cuva_qwen/`, with the best eval checkpoint
under `vqa_cuva_qwen_best/`.

### Inference (CUVA)

```bash
python -m heads.vqa.inference \
  --video data/CUVA/videos/path/to/00001.mp4 \
  --question "Please give a detailed description of the anomalous event."
```

---

## VAU-Bench Description

Video-only captioning: the model does **not** use the VAU-Bench `Question` /
options. A fixed prompt is used instead:

> Describe what happens in the video.

Training targets the `Description` column. Frames are trimmed to
`[Start Time, End Time]` when both values are ≥ 0; `-1` means use the full
video. Duplicate QA rows for the same `Video Name` are collapsed to one sample.

Anomaly Class classification uses the same videos via
[`heads/anomaly`](../anomaly).

### Dataset setup

Annotations come from [7xiang/VAU-Bench](https://huggingface.co/datasets/7xiang/VAU-Bench).
**UCF-Crime videos** come from Hugging Face
[`etornam/ufc-crime-videos`](https://huggingface.co/datasets/etornam/ufc-crime-videos)
(~105 GB full; ~57 GB with `--vau-only` for train+val). Training defaults to
**UCF-only** (`--sources ucf`), which covers about 1194 train / 299 validation
videos. MSAD / ECVA are optional and not downloaded by the helpers below.

Original UCF citation:
[UCF-Crime project page](https://www.crcv.ucf.edu/projects/real-world/).

```bash
# 1. Download annotations
python -m heads.vqa.vau_dataset --annotations-only --split train
python -m heads.vqa.vau_dataset --annotations-only --split validation

# 2. Download only UCF videos named in VAU train+val (~57 GB) and stage as ucf_*
python -m heads.vqa.vau_dataset --download-ucf --vau-only
# Full mirror (~105 GB): omit --vau-only
# If already downloaded to data/UCF-Crime/hf:
# python -m heads.vqa.vau_dataset --download-ucf --stage-only

# 3. Verify UCF coverage
python -m heads.vqa.vau_dataset --verify-videos --sources ucf --split train
python -m heads.vqa.vau_dataset --verify-videos --sources ucf --split validation
```

Train with `--skip-missing-videos` so any files that fail to download are skipped:

```bash
python -m heads.vqa.train --dataset vau --sources ucf --skip-missing-videos --epochs 5
```

Expected layout:

```text
data/UCF-Crime/hf/           # HF snapshot (~57 GB with --vau-only)
  Anomaly-Videos-Part-1/...
  Training-Normal-Videos-Part-1/...
data/VAU-Bench/
  hf_cache/                  # VAU annotations
  videos/
    ucf_Abuse001_x264.mp4    # staged (symlink or copy)
```

### Train (VAU Description)

Defaults to UCF-only when `--dataset vau`. Training is **two-stage** by default:

1. Resampler + gated cross-attn only (`--adapter-epochs`, LoRA frozen, lr `1e-4`)
2. Joint visual pathway + LoRA (`--epochs`, LoRA lr `1e-5`) with early stopping

```bash
python -m heads.vqa.train \
  --dataset vau \
  --sources ucf \
  --skip-missing-videos \
  --adapter-epochs 2 \
  --epochs 5 \
  --batch-size 2 \
  --num-frames 16 \
  --adapter-lr 1e-4 \
  --llm-lr 1e-5 \
  --patience 2 \
  --eval-samples 4
```

On MPS, consider a sparser cross-attn insert for memory, e.g. `--xattn-every-n 6`.

Skip stage 1 with `--adapter-epochs 0`. Checkpoints default to `dinov3/checkpoints/model/vqa_vau_qwen/`.

### Inference (VAU Description)

```bash
python -m heads.vqa.inference \
  --video data/VAU-Bench/videos/ucf_Abuse001_x264.mp4 \
  --no-question \
  --checkpoint dinov3/checkpoints/model/vqa_vau_qwen \
  --start-sec 23 \
  --end-sec 31
```

Omitting `--question` (or passing `--no-question`) uses the fixed caption
prompt. Optional `--start-sec` / `--end-sec` trim frame sampling to the
annotated anomaly window.

```python
from heads.vqa.inference import run_inference

answer = run_inference(
    video_path="data/VAU-Bench/videos/ucf_Abuse001_x264.mp4",
    question=None,  # uses fixed caption prompt
    checkpoint_dir="dinov3/checkpoints/model/vqa_vau_qwen",
    start_sec=23,
    end_sec=31,
)
```

---

## Logging

Log epoch losses and eval samples to SQLite:

```bash
python -m heads.vqa.train \
  --dataset vau \
  --skip-missing-videos \
  --log-db logs/vqa.db \
  --run-name vau-description \
  --adapter-epochs 2 \
  --epochs 5
```

See [`logger/README.md`](../../logger/README.md) for the schema and SQL queries.

## Checks

```bash
python -m py_compile \
  heads/vqa/llm_loader.py \
  heads/vqa/resampler.py \
  heads/vqa/gated_xattn.py \
  heads/vqa/dataset.py \
  heads/vqa/vau_dataset.py \
  heads/vqa/model.py \
  heads/vqa/train.py \
  heads/vqa/inference.py

python -m heads.vqa.model   # smoke: encode + forward loss on device
```
