"""Shared webcam grabber and a live preview window for the physical capture tools.

Two pieces, both dependency-free beyond what the tools already use (OpenCV for
V4L2 capture, Pillow, and the standard-library tkinter):

``FrameGrabber``
    Owns the single ``cv2.VideoCapture`` of the wrist camera (V4L2 allows one
    open per device) on a background thread that reads continuously and keeps
    only the newest frame. Consumers call ``latest()`` and get
    ``(sequence, timestamp, frame)``; the sequence number lets a slow consumer
    (the chessboard detector takes ~100 ms) skip frames it has already seen.
    This replaces the manual "read three frames and keep the last" buffer
    drain the capture tools used before.

``PreviewWindow``
    A tkinter window (must live on the main thread) showing the stream at half
    resolution with a status line, and a red dot that flashes in the top-right
    corner whenever ``flash()`` is called (the tools call it when a frame is
    saved). If the shell has no display (no DISPLAY/WAYLAND_DISPLAY), it
    degrades to a no-op so every tool still works headless.

``run_with_preview(worker, preview)`` runs a tool's capture loop on a worker
thread while the window's event loop owns the main thread; either side
finishing stops the other, and worker exceptions are re-raised.

The OpenCV in this environment is the headless build, so ``cv2.imshow`` is not
an option; tkinter + Pillow is.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

import numpy as np


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

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self) -> "FrameGrabber":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def compose_view(frame, step: int, status: str, overlays, flash: bool):
    """Half-resolution PIL image of ``frame`` with the status bar, target overlays and the save dot.

    ``overlays`` is a list of dicts with full-resolution ``box`` [x0, y0, x1, y1], optional ``label``
    and ``color`` (RGB tuple); they are drawn as thick rectangles so a viewer sees where the board
    should go.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.ascontiguousarray(frame[::step, ::step, ::-1]))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for item in overlays or ():
        x0, y0, x1, y1 = [v / step for v in item["box"]]
        color = tuple(item.get("color", (255, 220, 0)))
        for k in range(3):
            draw.rectangle([x0 + k, y0 + k, x1 - k, y1 - k], outline=color)
        label = item.get("label")
        if label:
            draw.rectangle([x0, max(0, y0 - 18), x0 + 8 * len(label) + 8, max(0, y0 - 18) + 18], fill=color)
            draw.text((x0 + 4, max(0, y0 - 18) + 3), label, fill=(0, 0, 0))
    if status:
        draw.rectangle([0, 0, width, 22], fill=(0, 0, 0))
        draw.text((6, 4), status[:170], fill=(255, 255, 255))
    if flash:
        r = max(8, int(0.02 * width))
        cx, cy = width - r - 12, r + 12
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 40, 40), outline=(255, 255, 255))
    return image


class SnapshotWriter:
    """Writes the composed view to ``path`` atomically at most every ``interval`` seconds (for a remote viewer)."""

    def __init__(self, path, interval: float = 1.0) -> None:
        self.path = path
        self.interval = float(interval)
        self._last = 0.0

    def maybe_write(self, image) -> bool:
        now = time.time()
        if now - self._last < self.interval:
            return False
        tmp = str(self.path) + ".tmp"
        image.save(tmp, format="JPEG", quality=80)
        os.replace(tmp, self.path)
        self._last = now
        return True


class _HeadlessPreview:
    """Stand-in with the PreviewWindow interface when no display exists (still writes snapshots if asked)."""

    closed = False
    headless = True

    def __init__(self, title: str = "", grabber: Optional[FrameGrabber] = None, scale: float = 0.5, snapshot=None, **kwargs: Any) -> None:
        print("no display: DISPLAY/WAYLAND_DISPLAY are unset in this shell, so the preview window cannot open; "
              "capturing without it (run from an interactive WSLg terminal to see the live stream).", flush=True)
        self.grabber = grabber
        self._step = max(1, int(round(1.0 / float(scale))))
        self._status = ""
        self._overlays: list = []
        self._flash_until = 0.0
        self._snapshot = SnapshotWriter(snapshot) if snapshot else None

    def flash(self, seconds: float = 0.4) -> None:
        self._flash_until = time.time() + float(seconds)

    def set_status(self, text: str) -> None:
        self._status = str(text)

    def set_overlays(self, overlays) -> None:
        self._overlays = list(overlays or ())

    def run(self, until: Callable[[], bool]) -> None:
        while not until():
            if self._snapshot is not None and self.grabber is not None:
                _seq, _stamp, frame = self.grabber.latest()
                if frame is not None:
                    self._snapshot.maybe_write(compose_view(frame, self._step, self._status, self._overlays, time.time() < self._flash_until))
            time.sleep(0.05)

    def close(self) -> None:
        self.closed = True


class PreviewWindow:
    """Live view of a FrameGrabber with a status line and a flashing save indicator."""

    headless = False

    def __new__(cls, *args: Any, **kwargs: Any):
        if not display_available():
            return _HeadlessPreview()
        return super().__new__(cls)

    def __init__(self, title: str, grabber: FrameGrabber, scale: float = 0.5, interval_ms: int = 15, snapshot=None) -> None:
        import tkinter as tk

        from PIL import Image, ImageDraw, ImageTk

        self._tk, self._Image, self._ImageDraw, self._ImageTk = tk, Image, ImageDraw, ImageTk
        self.grabber = grabber
        self.scale = float(scale)
        self.interval_ms = int(interval_ms)
        self.closed = False
        self._flash_until = 0.0
        self._status = ""
        self._overlays: list = []
        self._snapshot = SnapshotWriter(snapshot) if snapshot else None
        self._shown_seq = -1
        self._root = tk.Tk()
        self._root.title(title)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.bind("<Escape>", lambda _e: self.close())
        self._root.bind("q", lambda _e: self.close())
        # Integer decimation (frame[::step, ::step]) is ~10x cheaper than a PIL resize at 1080p and
        # is all a live preview needs; scale is rounded to the nearest 1/step.
        self._step = max(1, int(round(1.0 / self.scale)))
        width = -(-grabber.properties["width"] // self._step)
        height = -(-grabber.properties["height"] // self._step)
        self._size = (width, height)
        self._label = tk.Label(self._root, width=width, height=height, bg="black")
        self._label.pack()
        self._photo = None
        self.frames_shown = 0

    # Thread-safe: only assigns floats/strings read by the Tk tick.
    def flash(self, seconds: float = 0.4) -> None:
        self._flash_until = time.time() + float(seconds)

    def set_status(self, text: str) -> None:
        self._status = str(text)

    def set_overlays(self, overlays) -> None:
        """Full-resolution boxes to highlight (list of {box:[x0,y0,x1,y1], label, color}); replaces the previous set."""
        self._overlays = list(overlays or ())

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._root.destroy()
            except Exception:
                pass

    def _tick(self, until: Callable[[], bool]) -> None:
        if self.closed:
            return
        if until():
            self.close()
            return
        seq, _stamp, frame = self.grabber.latest()
        if seq != self._shown_seq and frame is not None:
            self._shown_seq = seq
            image = compose_view(frame, self._step, self._status, self._overlays, time.time() < self._flash_until)
            if self._snapshot is not None:
                self._snapshot.maybe_write(image)
            if self._photo is None or self._photo.width() != image.width or self._photo.height() != image.height:
                self._photo = self._ImageTk.PhotoImage(image)
                self._label.configure(image=self._photo)
            else:
                self._photo.paste(image)
            self.frames_shown += 1
        self._root.after(self.interval_ms, self._tick, until)

    def run(self, until: Callable[[], bool]) -> None:
        """Own the main thread until ``until()`` is true or the window is closed."""
        self._root.after(self.interval_ms, self._tick, until)
        try:
            self._root.mainloop()
        finally:
            self.close()


def run_with_preview(worker: Callable[[], Any], preview) -> Any:
    """Run ``worker`` on a thread while ``preview.run`` owns the main thread; return the worker's result.

    The worker should poll ``preview.closed`` and stop when it becomes true; the window stops when
    the worker finishes. Worker exceptions are re-raised here after the window closes.
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["result"] = worker()
        except BaseException as exc:  # re-raised on the main thread
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=target, name="capture-worker", daemon=True)
    thread.start()
    preview.run(until=done.is_set)
    thread.join(timeout=5.0)
    if "error" in box:
        raise box["error"]
    return box.get("result")
