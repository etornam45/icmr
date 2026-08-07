# SQLite Training Logger

Small, head-agnostic experiment logger. Stores runs, numeric metrics, and
arbitrary JSON records in one SQLite file. Depends only on the Python standard
library (`sqlite3`).

## Quick start

```python
from logger import SQLiteLogger

with SQLiteLogger(
    "logs/experiments.db",
    head="vqa",  # or "detr", or any label
    name="baseline",
    config={"lr": 1e-4, "epochs": 5},
) as log:
    log.log_metrics({"train/loss": 1.23, "eval/loss": 1.10}, epoch=1)
    log.log_record(
        "eval_sample",
        {
            "question": "What happened?",
            "reference": "A theft",
            "prediction": "A robbery",
        },
        epoch=1,
    )
```

Without a context manager:

```python
log = SQLiteLogger("logs/experiments.db", head="detr", name="run-1")
try:
    log.log_metric("train/loss", 2.5, epoch=1, split="train")
    log.finish(status="completed")
finally:
    log.close()
```

## Schema

```sql
runs(id, head, name, status, config_json, started_at, ended_at)
metrics(id, run_id, epoch, step, split, name, value, created_at)
records(id, run_id, epoch, step, kind, payload_json, created_at)
```

`kind` on `records` is free-form (`eval_sample`, `prediction`, etc.). Payloads
are JSON, so heads can log different fields without schema changes.

## Training integrations

### Caption

```bash
python -m heads.caption.train \
  --skip-missing-videos \
  --log-db logs/caption.db \
  --run-name vau-caption \
  --epochs 5
```

Logs per epoch: `train/loss`, `eval/loss`.

### DETR

```bash
python -m heads.detr.train \
  --log-db logs/detr.db \
  --run-name coco-detr \
  --epochs 50
```

Logs per epoch: `train/loss` (no separate eval loop in the current script).

## Useful queries

```sql
-- Latest runs
SELECT id, head, name, status, started_at, ended_at
FROM runs
ORDER BY id DESC;

-- Loss curves for a run
SELECT epoch, name, value
FROM metrics
WHERE run_id = 1
ORDER BY epoch, name;

-- Eval samples
SELECT epoch, payload_json
FROM records
WHERE run_id = 1 AND kind = 'eval_sample';
```
