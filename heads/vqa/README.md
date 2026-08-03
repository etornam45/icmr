# DINOv3 + MiniCPM5-1B CUVA Video QA

Video question answering on
[fesvhtr/CUVA](https://huggingface.co/datasets/fesvhtr/CUVA):

- **Vision**: frozen DINOv3 ViT-S CLS token for each sampled frame
- **Adapter**: linear 384 → 1536, LayerNorm, and 1D temporal position encoding
- **Language**: LoRA-tuned
  [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
- **Tasks**: Detection, Classification, Cause, Result, Timestamp, and Description

Each of the 16 uniformly sampled frames becomes one ordered MiniCPM visual
prefix token. Existing image-VQA checkpoints are incompatible.

## Dataset

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

## Download

### Recommended: project helper

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

delete 

```bash
rm data/CUVA/raw/group_{0,1,2,3}.zip
```

This uses every available video under `video-root`; evaluation is consequently
performed only on the available subset as well.

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

Keep enough free space for both the archives and extracted videos. Delete the
ZIP files after successful extraction if you need to reclaim space.

### Direct Hugging Face download

Download a single archive:

```python
from huggingface_hub import hf_hub_download

zip_path = hf_hub_download(
    repo_id="fesvhtr/CUVA",
    repo_type="dataset",
    filename="raw/group_0.zip",
    local_dir="data/CUVA",
)
print(zip_path)
```

Download the complete repository:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fesvhtr/CUVA",
    repo_type="dataset",
    local_dir="data/CUVA",
)
```

The full snapshot downloads all annotations and archives but does not extract
the ZIP files. Extract them into one searchable video root:

```bash
mkdir -p data/CUVA/videos
for archive in data/CUVA/raw/group_*.zip; do
  unzip "$archive" -d data/CUVA/videos
done
```

## Inspect annotations

No video download is required:

```python
from heads.vqa.dataset import load_cuva_samples

train = load_cuva_samples("train")
test = load_cuva_samples("test")
print(train[0])
```

Filter specific task types:

```python
samples = load_cuva_samples(
    "train",
    tasks=["Classification", "Cause", "Result"],
)
```

## Train

If videos were downloaded with the project helper:

```bash
python -m heads.vqa.train \
  --epochs 5 \
  --batch-size 2 \
  --num-frames 16 \
  --llm-lr 2e-4 \
  --adapter-lr 1e-4
```

Download videos automatically before training:

```bash
python -m heads.vqa.train --download-videos --epochs 5
```

Use already extracted videos elsewhere:

```bash
python -m heads.vqa.train \
  --video-root /path/to/CUVA/videos \
  --epochs 5
```

Train only selected tasks:

```bash
python -m heads.vqa.train \
  --tasks Classification,Cause,Result \
  --epochs 5
```

Resume:

```bash
python -m heads.vqa.train \
  --resume dinov3/checkpoints/model/vqa_cuva_minicpm \
  --epochs 5
```

Log epoch losses and eval samples to SQLite:

```bash
python -m heads.vqa.train \
  --skip-missing-videos \
  --log-db logs/vqa.db \
  --run-name cuva-baseline \
  --epochs 5
```

See [`logger/README.md`](../../logger/README.md) for the schema and SQL queries.

Checkpoints are saved under
`dinov3/checkpoints/model/vqa_cuva_minicpm/`, with the best eval checkpoint
under `vqa_cuva_minicpm_best/`.

## Inference

```bash
python -m heads.vqa.inference \
  --video data/CUVA/videos/path/to/00001.mp4 \
  --question "Please give a detailed description of the anomalous event."
```

```python
from heads.vqa.inference import run_inference

answer = run_inference(
    video_path="data/CUVA/videos/path/to/00001.mp4",
    question="What caused the anomalous event?",
)
```

## Checks

```bash
python -m py_compile \
  heads/vqa/dataset.py \
  heads/vqa/model.py \
  heads/vqa/train.py \
  heads/vqa/inference.py
```
