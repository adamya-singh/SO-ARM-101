"""Real-arm backend for the physical runner: LeRobot bus for joints, the wrist webcam for frames.

Duck-typed on the ``robot`` (a LeRobot ``SO101Follower``) and ``grabber``
(:class:`so_arm101_v2.physical.camera.FrameGrabber`) so it is testable with
mocks; nothing here imports lerobot.

Observation contract (what the lens-scene policies trained on): the raw
1920x1080 MJPEG frame, channel order flipped BGR -> RGB, squashed to 256x256
with the exact area filter of ``LensModel.real_operator()``. No undistortion,
no crop. In the loop the area filter is computed with ``cv2.resize`` in
INTER_AREA mode, which is bit-identical to the contract operator and about
eight times faster; ``verify_fast_resampler`` proves that on live frames at
preflight and the runner refuses to start otherwise.

Timing: absolute deadlines ``t0 + (k + 1) / hz``; an overrun of more than half
a period re-anchors the schedule instead of bursting to catch up; a step over
``max_step_ms`` or ``max_consecutive_overruns`` overruns aborts (the last
command sent is always a gated one).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.lens import LensModel
from so_arm101_v2.contracts.physical import HoldDecision
from so_arm101_v2.contracts.physical_io import read_measured_act

from .runner import CONTROL_HZ, Observation, PeriodOutcome, SendRecord


MIN_SERVO_VOLTAGE_V = 6.0   # STS3215 rated 6-12 V; the 2026-09-09 bench supply read 5.3-5.4 V (gripper voltage error, elbow sag)


class StaleFrame(RuntimeError):
    pass


def read_present_voltages(robot: Any) -> dict[str, float]:
    """Per-motor supply voltage in volts (read-only register, raw unit 0.1 V)."""
    raw = robot.bus.sync_read("Present_Voltage", normalize=False)
    return {name: float(raw[name]) / 10.0 for name in JOINT_NAMES}


def check_servo_voltage(robot: Any, *, minimum_v: float = MIN_SERVO_VOLTAGE_V) -> dict[str, Any]:
    """Preflight evidence: every motor's present voltage and whether all are at or above ``minimum_v``."""
    volts = read_present_voltages(robot)
    lowest = min(volts.values())
    return dict(volts={name: round(value, 2) for name, value in volts.items()}, minimum_v=float(minimum_v), lowest_v=round(lowest, 2), ok=bool(lowest >= minimum_v))


class DriverModifiedCommand(RuntimeError):
    pass


class TimingViolation(RuntimeError):
    pass


def observation_from_bgr(frame_bgr: np.ndarray, lens: LensModel) -> np.ndarray:
    """Contract path: BGR frame (any 16:9 size) -> RGB -> exact area filter -> (256, 256, 3) uint8."""
    rgb = np.ascontiguousarray(np.asarray(frame_bgr)[:, :, ::-1])
    return lens.real_operator(source_size=(rgb.shape[1], rgb.shape[0])).apply(rgb)


def fast_area_resampler(lens: LensModel, source_size=None) -> Callable[[np.ndarray], np.ndarray]:
    """cv2 INTER_AREA on the BGR frame, then the channel flip (per-channel filters commute with it)."""
    import cv2

    n = lens.observation_size
    w, h = lens.image_size if source_size is None else (int(source_size[0]), int(source_size[1]))
    lens.real_operator(source_size=(w, h))  # validates the aspect ratio

    def resample(frame_bgr: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_bgr)
        if frame.shape[:2] != (h, w) or frame.dtype != np.uint8:
            raise ValueError(f"expected a uint8 {h}x{w} frame, got {frame.shape} {frame.dtype}")
        small = cv2.resize(frame, (n, n), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(small[:, :, ::-1])

    return resample


def verify_fast_resampler(lens: LensModel, bgr_frames: Sequence[np.ndarray], resampler=None) -> dict[str, Any]:
    """Assert the fast path equals the contract operator on every frame; returns evidence for run.json."""
    if resampler is None:
        first = np.asarray(bgr_frames[0])
        resampler = fast_area_resampler(lens, source_size=(first.shape[1], first.shape[0]))
    checked = []
    for index, frame in enumerate(bgr_frames):
        fast = resampler(frame)
        exact = observation_from_bgr(frame, lens)
        if fast.shape != exact.shape or fast.dtype != exact.dtype or not np.array_equal(fast, exact):
            differing = int(np.count_nonzero(fast != exact)) if fast.shape == exact.shape else -1
            raise RuntimeError(f"fast area resampler differs from LensModel.real_operator on frame {index} ({differing} values)")
        checked.append(dict(index=index, shape=list(exact.shape), mean=float(exact.mean())))
    source = np.asarray(bgr_frames[0]).shape if len(bgr_frames) else None
    return dict(method="cv2.resize INTER_AREA + BGR->RGB", source_shape=(list(source) if source else None),
                frames_checked=len(checked), frames=checked, bit_identical=True)


class LeRobotBackend:
    name = "lerobot"

    def __init__(self, robot: Any, grabber: Any, *, lens: LensModel, control_hz: float = CONTROL_HZ,
                 max_frame_age_s: float = 0.100, max_step_ms: float = 100.0, max_consecutive_overruns: int = 3,
                 resampler: Callable[[np.ndarray], np.ndarray] | None = None, source_size=None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.robot = robot
        self.grabber = grabber
        self.lens = lens
        self.period = 1.0 / float(control_hz)
        self.max_frame_age_s = float(max_frame_age_s)
        self.max_step_ms = float(max_step_ms)
        self.max_consecutive_overruns = int(max_consecutive_overruns)
        self.resampler = resampler or fast_area_resampler(lens, source_size=source_size)
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.t0: float | None = None
        self.step_started: float | None = None
        self.last_boundary_seq = -1
        self.consecutive_overruns = 0
        self.overruns = 0
        self.reanchors = 0
        self.engaged = False

    # ---- torque -------------------------------------------------------------
    def engage(self, current_physical: np.ndarray) -> None:
        """Seed the goal to the present pose, then enable torque (no startup jump)."""
        self.robot.bus.sync_write("Goal_Position", {name: float(value) for name, value in zip(JOINT_NAMES, current_physical)})
        self.robot.bus.enable_torque()
        self.engaged = True

    # ---- backend protocol ---------------------------------------------------
    def read_act(self) -> np.ndarray:
        self.step_started = self.clock()
        if self.t0 is None:
            self.t0 = self.step_started
        return read_measured_act(self.robot)

    def observe(self, step: int) -> Observation:
        started = self.clock()
        seq, stamp, frame = self.grabber.latest()
        age = self.wall() - stamp
        if age > self.max_frame_age_s:
            raise StaleFrame(f"step {step}: newest camera frame is {age * 1e3:.0f} ms old")
        if seq <= self.last_boundary_seq:
            raise StaleFrame(f"step {step}: camera frame {seq} already used at the previous boundary")
        self.last_boundary_seq = seq
        image = self.resampler(frame)
        raw_rgb = np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])
        return Observation(image=image, frame_seq=int(seq), frame_time=float(stamp), age_s=float(age), raw_rgb=raw_rgb,
                           observe_ms=(self.clock() - started) * 1e3)

    def send(self, step: int, decision: HoldDecision) -> SendRecord:
        started = self.clock()
        physical = np.asarray(decision.sent_physical, dtype=np.float64)
        action = {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, physical)}
        returned = self.robot.send_action(action)
        sent_values = np.asarray([float(returned[f"{name}.pos"]) for name in JOINT_NAMES], dtype=np.float64)
        record = SendRecord(sent_physical=physical.astype(np.float32), returned_physical=sent_values.astype(np.float32),
                            send_ms=(self.clock() - started) * 1e3)
        if not np.allclose(sent_values, physical, atol=1e-5, rtol=0):
            raise DriverModifiedCommand(f"step {step}: driver modified the command {physical.tolist()} -> {sent_values.tolist()}")
        return record

    def wait_next_period(self, step: int) -> PeriodOutcome:
        now = self.clock()
        step_ms = (now - self.step_started) * 1e3 if self.step_started is not None else 0.0
        deadline = self.t0 + (step + 1) * self.period
        lateness = now - deadline
        overrun = lateness > 0
        reanchored = False
        if step_ms > self.max_step_ms:
            raise TimingViolation(f"step {step} took {step_ms:.0f} ms (limit {self.max_step_ms:.0f} ms)")
        if overrun:
            self.overruns += 1
            self.consecutive_overruns += 1
            if self.consecutive_overruns > self.max_consecutive_overruns:
                raise TimingViolation(f"{self.consecutive_overruns} consecutive overruns at step {step}")
            if lateness > 0.5 * self.period:
                self.t0 += lateness
                self.reanchors += 1
                reanchored = True
        else:
            self.consecutive_overruns = 0
            self.sleep(-lateness)
        return PeriodOutcome(done=False, success=None, invalidated=False, lateness_ms=lateness * 1e3,
                             overrun=overrun, reanchored=reanchored)

    def close(self) -> None:
        pass  # the tool owns the bus and the grabber lifecycle


__all__ = ["DriverModifiedCommand", "LeRobotBackend", "MIN_SERVO_VOLTAGE_V", "StaleFrame", "TimingViolation", "check_servo_voltage",
           "fast_area_resampler", "observation_from_bgr", "read_present_voltages", "verify_fast_resampler"]
