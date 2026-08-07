"""Orchestrates capture → inference → websocket broadcast → caption events."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

from fastapi import WebSocket

from heads.caption.inference import generate_caption
from server.capture import CaptureLoop, FrameBuffer, SourceSpec
from server.config import ServerConfig
from server.events import EventStore
from server.pipeline import run_pipeline
from server.runtime import ModelRuntime

logger = logging.getLogger(__name__)


class MonitorService:
    def __init__(
        self,
        runtime: ModelRuntime,
        cfg: ServerConfig,
        events: EventStore,
    ):
        self.runtime = runtime
        self.cfg = cfg
        self.events = events
        self.buffer = FrameBuffer(
            maxlen=max(int(cfg.buffer_seconds * 30), max(cfg.num_frames, 48) * 3)
        )
        self.capture = CaptureLoop(self.buffer)
        self.overlay_mode = cfg.default_overlay
        self._clients: set[WebSocket] = set()
        self._loop_task: asyncio.Task | None = None
        self._last_anomaly_ts = 0.0
        self._caption_busy = False
        self._latest_meta: dict[str, Any] = {
            "anomaly": None,
            "score": None,
            "detections": [],
            "source": None,
            "running": False,
            "is_anomaly": False,
            "segments": [],
        }
        self._lock = asyncio.Lock()

    @property
    def latest_meta(self) -> dict[str, Any]:
        return dict(self._latest_meta)

    async def start_source(self, source_type: str, uri: str) -> dict[str, Any]:
        from server.capture import classify_uri

        uri = uri.strip()
        if not uri:
            raise ValueError("uri is required")

        if source_type == "rtsp":
            source_type = classify_uri(uri)
            if source_type == "file":
                raise ValueError("Stream URL must start with rtsp://, http://, or https://")
        elif source_type == "stream":
            source_type = classify_uri(uri)
            if source_type == "file":
                source_type = "stream"
        elif source_type != "file":
            raise ValueError("source type must be 'file', 'rtsp', or 'stream'")

        async with self._lock:
            self.capture.start(SourceSpec(type=source_type, uri=uri))  # type: ignore[arg-type]
            if self._loop_task is None or self._loop_task.done():
                self._loop_task = asyncio.create_task(
                    self._inference_loop(), name="icmr-inference"
                )
        return {"ok": True, "type": source_type, "uri": uri}

    async def stop_source(self) -> dict[str, Any]:
        async with self._lock:
            self.capture.stop()
            if self._loop_task is not None:
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
                self._loop_task = None
        self._latest_meta["running"] = False
        self._latest_meta["source"] = None
        return {"ok": True}

    def set_overlay(self, mode: str) -> dict[str, Any]:
        if mode not in {"none", "detection", "pca"}:
            raise ValueError("overlay mode must be none|detection|pca")
        self.overlay_mode = mode
        self._latest_meta["overlay_mode"] = mode
        return {"ok": True, "mode": mode}

    async def register_client(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def unregister_client(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for client in list(self._clients):
            try:
                await client.send_json(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    async def _inference_loop(self) -> None:
        logger.info("Inference loop started")
        try:
            while True:
                source = self.capture.source
                if source is None:
                    await asyncio.sleep(0.2)
                    continue

                if self.capture.error:
                    await self._broadcast(
                        {"type": "error", "message": self.capture.error}
                    )
                    await asyncio.sleep(1.0)
                    continue

                latest = self.buffer.latest()
                if latest is None:
                    await asyncio.sleep(0.05)
                    continue

                span = self.buffer.time_span()
                window_start_ts = span[0] if span else None
                window_end_ts = span[1] if span else None

                frames = self.buffer.sample_uniform(self.cfg.num_frames)
                if len(frames) < max(1, self.cfg.num_frames // 4):
                    await asyncio.sleep(0.05)
                    continue

                while len(frames) < self.cfg.num_frames:
                    frames.append(frames[-1])

                _, preview = latest
                try:
                    result = await asyncio.to_thread(
                        run_pipeline,
                        self.runtime,
                        frames,
                        preview,
                        self.cfg,
                        window_start_ts,
                        window_end_ts,
                    )
                except Exception:
                    logger.exception("Pipeline tick failed")
                    await asyncio.sleep(self.cfg.inference_interval_sec)
                    continue

                frames_b64 = {
                    key: base64.b64encode(jpeg).decode("ascii")
                    for key, jpeg in result.jpegs.items()
                }
                meta = {
                    "type": "frame",
                    "frames": frames_b64,
                    "anomaly": result.anomaly_class,
                    "score": result.anomaly_score,
                    "top_k": result.top_k,
                    "detections": result.detections,
                    "segments": result.segments,
                    "segment": result.segment,
                    "svdd_score": result.svdd_score,
                    "source": {"type": source.type, "uri": source.uri},
                    "running": True,
                    "is_anomaly": result.is_anomaly,
                }
                self._latest_meta = {
                    k: v for k, v in meta.items() if k != "frames"
                }
                await self._broadcast(meta)

                if result.is_anomaly:
                    await self._maybe_trigger_caption(
                        source_uri=source.uri,
                        anomaly_class=result.anomaly_class or "unknown",
                        score=float(result.anomaly_score or 0.0),
                        videos_tensor=result.videos_tensor,
                        segment=result.segment,
                        svdd_score=result.svdd_score,
                        window_start_ts=result.window_start_ts,
                        window_end_ts=result.window_end_ts,
                    )

                await asyncio.sleep(self.cfg.inference_interval_sec)
        except asyncio.CancelledError:
            logger.info("Inference loop cancelled")
            raise

    async def _maybe_trigger_caption(
        self,
        source_uri: str,
        anomaly_class: str,
        score: float,
        videos_tensor,
        segment: dict[str, Any] | None = None,
        svdd_score: float | None = None,
        window_start_ts: float | None = None,
        window_end_ts: float | None = None,
    ) -> None:
        now = time.time()
        if now - self._last_anomaly_ts < self.cfg.anomaly_cooldown_sec:
            return
        if self._caption_busy:
            return

        self._last_anomaly_ts = now

        # Map relative segment times onto absolute buffer timestamps.
        event_start = None
        event_end = None
        if (
            segment is not None
            and window_start_ts is not None
            and window_end_ts is not None
        ):
            event_start = window_start_ts + float(segment["start"])
            event_end = window_start_ts + float(segment["end"])
            pad = self.cfg.span_pad_sec
            event_start = max(window_start_ts, event_start - pad)
            event_end = min(window_end_ts, event_end + pad)

        event = self.events.add_event(
            source=source_uri,
            anomaly_class=anomaly_class,
            score=score,
            caption=None if self.runtime.has_caption else "(caption unavailable)",
            start_ts=event_start,
            end_ts=event_end,
            svdd_score=svdd_score,
        )
        await self._broadcast(
            {
                "type": "event",
                "event": event.to_dict(),
            }
        )

        if not self.runtime.has_caption:
            return

        self._caption_busy = True
        n_frames = max(self.runtime.caption_num_frames, self.cfg.num_frames)

        caption_frames: list = []
        if event_start is not None and event_end is not None:
            caption_frames = self.buffer.sample_time_range(
                event_start, event_end, n_frames
            )
        if not caption_frames:
            caption_frames = self.buffer.sample_uniform(n_frames)

        if not caption_frames and videos_tensor is not None:
            clip = videos_tensor
        elif caption_frames:
            while len(caption_frames) < n_frames:
                caption_frames.append(caption_frames[-1])
            from server.pipeline import bgr_frames_to_tensor

            clip = bgr_frames_to_tensor(
                caption_frames, self.cfg.img_size, self.runtime.device
            ).cpu()
        else:
            self._caption_busy = False
            return

        async def _run_caption() -> None:
            try:
                caption = await asyncio.to_thread(
                    generate_caption,
                    self.runtime.caption,
                    self.runtime.caption_tokenizer,
                    clip.to(self.runtime.device),
                    backbone=self.runtime.backbone,
                )
                self.events.update_caption(event.id, caption)
                updated = event.to_dict()
                updated["caption"] = caption
                await self._broadcast({"type": "event", "event": updated})
            except Exception:
                logger.exception("Caption failed for event %s", event.id)
                self.events.update_caption(event.id, "(caption failed)")
            finally:
                self._caption_busy = False

        asyncio.create_task(_run_caption(), name=f"caption-event-{event.id}")
