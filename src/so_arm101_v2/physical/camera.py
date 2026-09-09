"""Wrist webcam grabber: one V4L2 open per device, a background reader, newest frame on demand.

Moved here from tools/camera_preview.py (which re-exports it) so the physical
runner can depend on it without importing the tkinter preview. Frames are the
raw OpenCV BGR uint8 arrays; consumers that feed a policy must flip to RGB.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional


class FrameGrabber:
    """Background reader that always holds the newest frame from one V4L2 device."""

    def __init__(self, device: int = 0, width: int = 1920, height: int = 1080, fps: int = 30, warmup: int = 20) -> None:
        import cv2

        self._cv2 = cv2
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        frame = None
        for _ in range(warmup):
            ok, frame = cap.read()
        if not cap.isOpened() or frame is None:
            cap.release()
            raise RuntimeError(f"camera device {device} did not deliver frames (is /dev/video{device} attached to WSL?)")
        self.device = device
        self.properties = dict(
            fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace"),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )
        self._cap = cap
        self._lock = threading.Lock()
        self._seq = 0
        self._frame = frame
        self._timestamp = time.time()
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="frame-grabber", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                with self._lock:
                    self._seq += 1
                    self._frame = frame
                    self._timestamp = time.time()
        except BaseException as exc:  # surfaced to the consumer on the next latest()
            self._error = exc

    def latest(self):
        """Newest ``(sequence, unix_time, bgr_frame)``; the frame is the grabber's own buffer, copy before mutating."""
        if self._error is not None:
            raise RuntimeError(f"camera grabber failed: {self._error!r}")
        with self._lock:
            return self._seq, self._timestamp, self._frame

    def wait_for_new(self, after_seq: int, timeout: float = 1.0):
        """Block until a frame newer than ``after_seq`` exists (or timeout), then return ``latest()``."""
        deadline = time.time() + timeout
        while True:
            seq, stamp, frame = self.latest()
            if seq > after_seq or time.time() >= deadline:
                return seq, stamp, frame
            time.sleep(0.002)

    def measure_rate(self, seconds: float = 1.0) -> float:
        """Frames per second actually delivered over ``seconds`` (preflight check)."""
        start_seq, _, _ = self.latest()
        time.sleep(seconds)
        end_seq, _, _ = self.latest()
        return (end_seq - start_seq) / seconds

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self) -> "FrameGrabber":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["FrameGrabber"]
