"""Head-agnostic SQLite logger for training metrics and structured records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class SQLiteLogger:
    """Minimal experiment logger backed by a single SQLite database file.

    Schema
    ------
    runs(id, head, name, status, config_json, started_at, ended_at)
    metrics(id, run_id, epoch, step, split, name, value, created_at)
    records(id, run_id, epoch, step, kind, payload_json, created_at)
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        head: str | None = None,
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        self.run_id = self._start_run(head=head, name=name, config=config)
        self._finished = False

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                head TEXT,
                name TEXT,
                status TEXT NOT NULL,
                config_json TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER,
                step INTEGER,
                split TEXT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER,
                step INTEGER,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id);
            """
        )
        self._conn.commit()

    def _start_run(
        self,
        *,
        head: str | None,
        name: str | None,
        config: Mapping[str, Any] | None,
    ) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO runs (head, name, status, config_json, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                head,
                name,
                "running",
                _json_dumps(dict(config or {})),
                _utc_now(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        epoch: int | None = None,
        step: int | None = None,
        split: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO metrics (run_id, epoch, step, split, name, value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (self.run_id, epoch, step, split, name, float(value), _utc_now()),
        )
        self._conn.commit()

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        epoch: int | None = None,
        step: int | None = None,
        split: str | None = None,
    ) -> None:
        now = _utc_now()
        self._conn.executemany(
            """
            INSERT INTO metrics (run_id, epoch, step, split, name, value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (self.run_id, epoch, step, split, name, float(value), now)
                for name, value in metrics.items()
            ],
        )
        self._conn.commit()

    def log_record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO records (run_id, epoch, step, kind, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                epoch,
                step,
                kind,
                _json_dumps(dict(payload)),
                _utc_now(),
            ),
        )
        self._conn.commit()

    def log_records(
        self,
        kind: str,
        payloads: Sequence[Mapping[str, Any]],
        *,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        now = _utc_now()
        self._conn.executemany(
            """
            INSERT INTO records (run_id, epoch, step, kind, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    self.run_id,
                    epoch,
                    step,
                    kind,
                    _json_dumps(dict(payload)),
                    now,
                )
                for payload in payloads
            ],
        )
        self._conn.commit()

    def finish(self, status: str = "completed") -> None:
        if self._finished:
            return
        self._conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            (status, _utc_now(), self.run_id),
        )
        self._conn.commit()
        self._finished = True

    def close(self) -> None:
        if not self._finished:
            self.finish(status="failed")
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finish(status="completed")
        else:
            self.finish(status="failed")
        self._conn.close()
