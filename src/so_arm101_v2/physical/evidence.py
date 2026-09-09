"""Evidence written by the physical runner: per-step CSV, boundary images, camera video, run record."""
from __future__ import annotations

import csv
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES

from .runner import StepRecord

STEP_COLUMNS = (
    ["step", "boundary", "start_monotonic", "lateness_ms", "overrun", "reanchored"]
    + [f"measured_act_{n}" for n in JOINT_NAMES] + [f"policy_act_{n}" for n in JOINT_NAMES]
    + [f"requested_act_{n}" for n in JOINT_NAMES] + [f"executed_act_{n}" for n in JOINT_NAMES]
    + [f"sent_physical_{n}" for n in JOINT_NAMES] + [f"returned_physical_{n}" for n in JOINT_NAMES]
    + [f"raw_goal_ticks_{n}" for n in JOINT_NAMES]
    + ["hold_reason", "consecutive_holds", "frame_seq", "frame_unix_time", "frame_age_ms",
       "read_ms", "observe_ms", "infer_ms", "gate_ms", "send_ms", "step_ms"]
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StepLog:
    """Exclusive CSV of every control step; flushed per row so an abort leaves a complete record."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = self.path.open("x", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(STEP_COLUMNS)
        self._handle.flush()
        self.rows = 0

    def write(self, record: StepRecord, start_monotonic: float | None = None) -> None:
        d = record.decision
        obs = record.observation
        returned = record.send.returned_physical
        row = [record.step, int(record.boundary), start_monotonic if start_monotonic is not None else "",
               f"{record.outcome.lateness_ms:.3f}", int(record.outcome.overrun), int(record.outcome.reanchored)]
        for values in (record.current_act, d.policy_act, d.requested_act, d.executed_act, record.send.sent_physical):
            row += [f"{float(v):.6f}" for v in np.asarray(values, dtype=np.float64).ravel()[:6]] if np.asarray(values).size == 6 else [""] * 6
        row += [f"{float(v):.6f}" for v in returned] if returned is not None else [""] * 6
        row += [int(v) for v in d.raw_goal_ticks]
        row += [d.hold_reason, record.consecutive_holds,
                obs.frame_seq if obs else "", f"{obs.frame_time:.6f}" if obs else "", f"{obs.age_s * 1e3:.1f}" if obs else "",
                f"{record.read_ms:.2f}", f"{obs.observe_ms:.2f}" if obs else "", f"{record.infer_ms:.2f}", f"{record.gate_ms:.2f}",
                f"{record.send.send_ms:.2f}", f"{record.step_ms:.2f}"]
        self._writer.writerow(row)
        self._handle.flush()
        self.rows += 1

    def close(self) -> None:
        self._handle.close()


class BoundaryStore:
    """Keeps boundary frames in memory during the run and writes PNGs (with hashes) afterwards."""

    def __init__(self) -> None:
        self.items: list[tuple[int, np.ndarray | None, np.ndarray]] = []

    def add(self, record: StepRecord) -> None:
        if record.observation is not None:
            self.items.append((record.step, record.observation.raw_rgb, np.array(record.observation.image, copy=True)))

    def write(self, directory: Path) -> list[dict[str, Any]]:
        from PIL import Image

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for step, raw, obs in self.items:
            entry: dict[str, Any] = {"step": step, "observation_array_sha256": hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()}
            obs_path = directory / f"step_{step:03d}.obs.png"
            Image.fromarray(obs).save(obs_path)
            entry["observation_png"] = obs_path.name
            entry["observation_png_sha256"] = sha256_file(obs_path)
            if raw is not None:
                raw_path = directory / f"step_{step:03d}.raw.png"
                Image.fromarray(raw).save(raw_path)
                entry["raw_png"] = raw_path.name
                entry["raw_png_sha256"] = sha256_file(raw_path)
                entry["raw_array_sha256"] = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
            written.append(entry)
        (directory / "boundaries.json").write_text(json.dumps(written, indent=1) + "\n")
        return written


class VideoRecorder:
    """Records the grabber's stream to an mp4 on a side thread (libx264 ultrafast, zerolatency)."""

    def __init__(self, grabber: Any, path: Path, *, fps: int = 30) -> None:
        import av

        self.grabber = grabber
        self.path = Path(path)
        self.fps = fps
        self._av = av
        self._queue: "queue.Queue[tuple[int, float, np.ndarray] | None]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.dropped = 0
        self.frames: list[tuple[int, int, float]] = []  # (video_index, grabber_seq, unix_time)
        self._poll = threading.Thread(target=self._poll_loop, name="video-poll", daemon=True)
        self._encode = threading.Thread(target=self._encode_loop, name="video-encode", daemon=True)

    def start(self) -> None:
        self._encode.start()
        self._poll.start()

    def _poll_loop(self) -> None:
        last = -1
        period = 1.0 / self.fps
        while not self._stop.is_set():
            seq, stamp, frame = self.grabber.latest()
            if seq != last:
                last = seq
                try:
                    self._queue.put_nowait((seq, stamp, frame))
                except queue.Full:
                    self.dropped += 1
            time.sleep(period / 3)
        self._queue.put(None)

    def _encode_loop(self) -> None:
        av = self._av
        container = av.open(str(self.path), mode="w")
        stream = container.add_stream("libx264", rate=self.fps)
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "20", "tune": "zerolatency"}
        first = True
        index = 0
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                seq, stamp, frame = item
                if first:
                    stream.width, stream.height = int(frame.shape[1]), int(frame.shape[0])
                    first = False
                video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                self.frames.append((index, int(seq), float(stamp)))
                index += 1
            if not first:
                for packet in stream.encode():
                    container.mux(packet)
        finally:
            container.close()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._poll.join(timeout=2.0)
        self._encode.join(timeout=30.0)
        sidecar = self.path.with_name(self.path.stem + "_frames.csv")
        with sidecar.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_index", "grabber_seq", "unix_time"])
            writer.writerows(self.frames)
        return dict(path=self.path.name, frames=len(self.frames), dropped=self.dropped,
                    sha256=(sha256_file(self.path) if self.path.exists() else None), frames_csv=sidecar.name)


def write_run_record(path: Path, record: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(record, indent=2, default=str) + "\n")


__all__ = ["STEP_COLUMNS", "BoundaryStore", "StepLog", "VideoRecorder", "sha256_file", "write_run_record"]
