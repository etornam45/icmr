"""Video capture from file, RTSP, or HTTP/HLS into a frame ring buffer."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SourceType = Literal["file", "rtsp", "stream"]

_FFMPEG_CAPTURE_OPTIONS = (
    "allowed_extensions;ALL|"
    "protocol_whitelist;file,http,https,tcp,tls,crypto,rtsp,rtp,udp,concat"
)

_PROTOCOL_WHITELIST = "file,http,https,tcp,tls,crypto,rtsp,rtp,udp,concat"


@dataclass
class SourceSpec:
    type: SourceType
    uri: str


def classify_uri(uri: str) -> SourceType:
    """Infer source type from a URI string."""
    lower = uri.strip().lower()
    if lower.startswith("rtsp://") or lower.startswith("rtsps://"):
        return "rtsp"
    if lower.startswith(("http://", "https://")) or lower.endswith(".m3u8"):
        return "stream"
    return "file"


def _ensure_ffmpeg_options() -> None:
    existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    if "allowed_extensions" not in existing:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"{existing}|{_FFMPEG_CAPTURE_OPTIONS}" if existing else _FFMPEG_CAPTURE_OPTIONS
        )


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


def probe_video_size(uri: str) -> tuple[int, int] | None:
    """Return (width, height) via ffprobe, or None on failure."""
    ffprobe = _ffprobe_bin()
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-allowed_extensions",
        "ALL",
        "-protocol_whitelist",
        _PROTOCOL_WHITELIST,
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        uri,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=20)
        text = out.decode("utf-8", errors="ignore").strip().splitlines()[0]
        width_s, height_s = text.split("x")
        width, height = int(width_s), int(height_s)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        logger.exception("ffprobe failed for %s", uri)
    return None


class FFmpegPipeCapture:
    """Read BGR frames from an ffmpeg stdout pipe (needed for some HLS cams)."""

    def __init__(self, uri: str, width: int, height: int):
        ffmpeg = _ffmpeg_bin()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH")
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._uri = uri
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-allowed_extensions",
            "ALL",
            "-protocol_whitelist",
            _PROTOCOL_WHITELIST,
            "-i",
            uri,
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 2,
        )

    def isOpened(self) -> bool:
        return self._proc.poll() is None and self._proc.stdout is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._proc.stdout is None or self._proc.poll() is not None:
            return False, None
        raw = self._proc.stdout.read(self.frame_bytes)
        if not raw or len(raw) < self.frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return True, frame.copy()

    def release(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._proc.stdout:
            self._proc.stdout.close()
        if self._proc.stderr:
            self._proc.stderr.close()

    def get(self, _prop: int) -> float:
        return 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return False


def open_capture(uri: str, source_type: SourceType):
    """Open a capture handle suited to the source type.

    HTTP/HLS streams use an ffmpeg pipe because OpenCV often cannot pass
    ``allowed_extensions=ALL`` (needed for cameras that serve ``.cmfv`` segments).
    """
    if source_type == "stream" and _ffmpeg_bin() is not None:
        size = probe_video_size(uri)
        if size is not None:
            width, height = size
            try:
                pipe = FFmpegPipeCapture(uri, width, height)
                if pipe.isOpened():
                    logger.info(
                        "Opened HLS/HTTP via ffmpeg pipe (%dx%d): %s",
                        width,
                        height,
                        uri,
                    )
                    return pipe
                pipe.release()
            except Exception:
                logger.exception("ffmpeg pipe open failed; falling back to OpenCV")

    _ensure_ffmpeg_options()
    if source_type in {"rtsp", "stream"}:
        cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    return cv2.VideoCapture(uri)


class FrameBuffer:
    """Thread-safe ring buffer of (timestamp, bgr_frame) tuples."""

    def __init__(self, maxlen: int = 240):
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def push(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> None:
        ts = time.time() if timestamp is None else timestamp
        with self._lock:
            self._frames.append((ts, frame_bgr.copy()))

    def latest(self) -> tuple[float, np.ndarray] | None:
        with self._lock:
            if not self._frames:
                return None
            ts, frame = self._frames[-1]
            return ts, frame.copy()

    def sample_uniform(self, n: int) -> list[np.ndarray]:
        """Return n frames uniformly sampled from the buffer (oldest→newest)."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return []
        if len(frames) <= n:
            return [f.copy() for _, f in frames]
        indices = np.linspace(0, len(frames) - 1, n)
        return [frames[int(i)][1].copy() for i in indices]

    def time_span(self) -> tuple[float, float] | None:
        """Oldest and newest timestamps currently in the buffer."""
        with self._lock:
            if not self._frames:
                return None
            return self._frames[0][0], self._frames[-1][0]

    def sample_time_range(
        self,
        start_ts: float,
        end_ts: float,
        n: int,
    ) -> list[np.ndarray]:
        """Uniformly sample ``n`` frames whose timestamps fall in [start, end]."""
        with self._lock:
            frames = [(ts, frame) for ts, frame in self._frames if start_ts <= ts <= end_ts]
        if not frames:
            return []
        if len(frames) <= n:
            return [f.copy() for _, f in frames]
        indices = np.linspace(0, len(frames) - 1, n)
        return [frames[int(i)][1].copy() for i in indices]

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


class CaptureLoop:
    """Background capture writing into a FrameBuffer."""

    def __init__(self, buffer: FrameBuffer):
        self.buffer = buffer
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._source: SourceSpec | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def source(self) -> SourceSpec | None:
        with self._lock:
            return self._source

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source: SourceSpec) -> None:
        self.stop()
        with self._lock:
            self._source = source
            self._error = None
        self.buffer.clear()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="icmr-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self._source = None

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message
        logger.error(message)

    def _run(self) -> None:
        source = self.source
        if source is None:
            return

        uri = source.uri
        is_network = source.type in {"rtsp", "stream"}
        cap = open_capture(uri, source.type)

        if not cap.isOpened():
            hint = ""
            if source.type == "stream" and _ffmpeg_bin() is None:
                hint = " (install ffmpeg for HLS/.m3u8 streams)"
            self._set_error(f"Failed to open source: {uri}{hint}")
            return

        logger.info("Capture opened (%s): %s", source.type, uri)
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if source.type == "file":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.01)
                        continue

                    cap.release()
                    time.sleep(0.75)
                    if self._stop.is_set():
                        break
                    logger.warning("Reconnecting stream: %s", uri)
                    cap = open_capture(uri, source.type)
                    if not cap.isOpened():
                        self._set_error(f"Lost stream: {uri}")
                        time.sleep(1.5)
                    continue

                self.buffer.push(frame)

                if source.type == "file":
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if fps > 1.0:
                        time.sleep(1.0 / fps)
                elif is_network:
                    time.sleep(0.001)
        finally:
            cap.release()
