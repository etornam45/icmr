"""SQLite event store for anomaly + caption timeline."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EventRecord:
    id: int
    timestamp: float
    source: str
    anomaly_class: str
    score: float
    caption: str | None
    thumbnail_path: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    svdd_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "source": self.source,
            "anomaly_class": self.anomaly_class,
            "score": self.score,
            "caption": self.caption,
            "thumbnail_path": self.thumbnail_path,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "svdd_score": self.svdd_score,
        }


class EventStore:
    def __init__(self, db_path: str, max_events: int = 200):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        for col, decl in (
            ("start_ts", "REAL"),
            ("end_ts", "REAL"),
            ("svdd_score", "REAL"),
        ):
            if col not in existing:
                conn.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        source TEXT NOT NULL,
                        anomaly_class TEXT NOT NULL,
                        score REAL NOT NULL,
                        caption TEXT,
                        thumbnail_path TEXT,
                        start_ts REAL,
                        end_ts REAL,
                        svdd_score REAL
                    )
                    """
                )
                self._ensure_columns(conn)
                conn.commit()
            finally:
                conn.close()

    def add_event(
        self,
        source: str,
        anomaly_class: str,
        score: float,
        caption: str | None = None,
        thumbnail_path: str | None = None,
        timestamp: float | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        svdd_score: float | None = None,
    ) -> EventRecord:
        ts = time.time() if timestamp is None else timestamp
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO events
                        (timestamp, source, anomaly_class, score, caption,
                         thumbnail_path, start_ts, end_ts, svdd_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        source,
                        anomaly_class,
                        float(score),
                        caption,
                        thumbnail_path,
                        start_ts,
                        end_ts,
                        svdd_score,
                    ),
                )
                event_id = int(cur.lastrowid)
                conn.execute(
                    """
                    DELETE FROM events WHERE id NOT IN (
                        SELECT id FROM events ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (self.max_events,),
                )
                conn.commit()
            finally:
                conn.close()
        return EventRecord(
            id=event_id,
            timestamp=ts,
            source=source,
            anomaly_class=anomaly_class,
            score=float(score),
            caption=caption,
            thumbnail_path=thumbnail_path,
            start_ts=start_ts,
            end_ts=end_ts,
            svdd_score=svdd_score,
        )

    def update_caption(self, event_id: int, caption: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE events SET caption = ? WHERE id = ?",
                    (caption, event_id),
                )
                conn.commit()
            finally:
                conn.close()

    def list_events(self, limit: int = 50) -> list[EventRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, source, anomaly_class, score, caption,
                           thumbnail_path, start_ts, end_ts, svdd_score
                    FROM events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
        return [
            EventRecord(
                id=int(row["id"]),
                timestamp=float(row["timestamp"]),
                source=str(row["source"]),
                anomaly_class=str(row["anomaly_class"]),
                score=float(row["score"]),
                caption=row["caption"],
                thumbnail_path=row["thumbnail_path"],
                start_ts=row["start_ts"],
                end_ts=row["end_ts"],
                svdd_score=row["svdd_score"],
            )
            for row in rows
        ]
