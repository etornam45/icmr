"""FastAPI app for ICMR CCTV monitoring."""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.config import ServerConfig, load_config
from server.events import EventStore
from server.monitor import MonitorService
from server.runtime import ModelRuntime, load_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


class SourceRequest(BaseModel):
    type: Literal["file", "rtsp", "stream"]
    uri: str = Field(..., min_length=1)


class OverlayRequest(BaseModel):
    mode: Literal["none", "detection", "pca"]


class AppState:
    config: ServerConfig
    runtime: ModelRuntime
    events: EventStore
    monitor: MonitorService


state = AppState()


def _safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\-]+", "_", base).strip("._")
    return cleaned or "upload.mp4"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    state.config = cfg
    Path(cfg.uploads_dir).mkdir(parents=True, exist_ok=True)
    state.events = EventStore(cfg.events_db_path, max_events=cfg.max_events)
    logger.info("Loading models…")
    state.runtime = load_runtime(cfg)
    state.monitor = MonitorService(state.runtime, cfg, state.events)
    logger.info(
        "ICMR server ready (anomaly=%s vqa=%s)",
        state.runtime.has_anomaly,
        state.runtime.has_vqa,
    )
    yield
    await state.monitor.stop_source()


app = FastAPI(title="ICMR", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=load_config().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "anomaly": state.runtime.has_anomaly if hasattr(state, "runtime") else False,
        "vqa": state.runtime.has_vqa if hasattr(state, "runtime") else False,
        "overlay": state.monitor.overlay_mode if hasattr(state, "monitor") else None,
        "source": state.monitor.latest_meta.get("source")
        if hasattr(state, "monitor")
        else None,
    }


@app.post("/source")
async def set_source(body: SourceRequest):
    try:
        return await state.monitor.start_source(body.type, body.uri)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/source/upload")
async def upload_source(file: UploadFile = File(...)):
    """Accept a browser-uploaded video, save it, and start file playback."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video type {suffix or '(none)'}. "
                f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            ),
        )

    uploads = Path(state.config.uploads_dir)
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex[:10]}_{_safe_filename(file.filename)}"

    max_bytes = state.config.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {state.config.max_upload_mb} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload")

    logger.info("Uploaded video %s (%d bytes) → %s", file.filename, written, dest)
    result = await state.monitor.start_source("file", str(dest.resolve()))
    return {
        **result,
        "filename": file.filename,
        "saved_as": str(dest),
        "bytes": written,
    }


@app.post("/source/stop")
async def stop_source():
    return await state.monitor.stop_source()


@app.post("/overlay")
def set_overlay(body: OverlayRequest):
    try:
        return state.monitor.set_overlay(body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/events")
def list_events(limit: int = 50):
    limit = max(1, min(limit, 200))
    return {"events": [e.to_dict() for e in state.events.list_events(limit=limit)]}


@app.get("/status")
def status():
    return state.monitor.latest_meta


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    import asyncio

    await state.monitor.register_client(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except TimeoutError:
                # Connection still open; client may be receive-only.
                continue
    except WebSocketDisconnect:
        state.monitor.unregister_client(websocket)
    except Exception:
        state.monitor.unregister_client(websocket)
