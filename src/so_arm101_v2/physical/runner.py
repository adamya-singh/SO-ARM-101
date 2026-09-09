"""Backend-agnostic control loop for a chunked vision policy, plus the gated approach to the reset pose.

``run_episode`` is the physical twin of ``simulation.rollout._run_pick_place_rollout``:

    for step in range(max_actions):
        current = backend.read_act()                       # fresh measured pose (ACT units)
        image   = backend.observe(step) if policy.needs_frame else None
        policy_act = policy.predict(image, current)        # network only at chunk boundaries
        decision = bench_hold_decision(current, policy_act, ...)   # the shared bench gate
        backend.send(step, decision)                       # executed or held
        outcome = backend.wait_next_period(step)           # sim: physics; real: absolute deadline

Off-boundary steps deliberately pass ``None`` as the image: a lockstep bug
would crash loudly instead of silently reusing a stale frame. The bench rule
on refusal is to hold the current pose and continue (as the simulator does);
``max_consecutive_holds`` aborts a run that is stuck holding.

``approach_to_reset`` brings the arm from wherever it is (gravity rest, with
the shoulder just below the -92 floor) to the recorded reset pose in small
gated steps, modelled on ``tools/prepare_physical_elbow.py``: fresh feedback
every step, targets ramped toward the reset, a command lead cap for gravity
tracking error, never a step away from the reset, a feedback stop and a
timeout. It never disables torque.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Protocol

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.physical import (
    HoldDecision,
    act_to_physical_normalized,
    bench_hold_decision,
    physical_normalized_to_act,
)

CONTROL_HZ = 30.0


@dataclass(frozen=True)
class Observation:
    image: np.ndarray            # (256, 256, 3) uint8 RGB, already lens-resampled
    frame_seq: int
    frame_time: float            # unix time of the raw frame
    age_s: float                 # observe time - frame time
    raw_rgb: np.ndarray | None   # full-resolution RGB copy (boundaries only, evidence)
    observe_ms: float


@dataclass(frozen=True)
class SendRecord:
    sent_physical: np.ndarray
    returned_physical: np.ndarray | None
    send_ms: float


@dataclass(frozen=True)
class PeriodOutcome:
    done: bool = False
    success: bool | None = None
    invalidated: bool = False
    lateness_ms: float = 0.0
    overrun: bool = False
    reanchored: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepRecord:
    step: int
    boundary: bool
    current_act: np.ndarray
    decision: HoldDecision
    observation: Observation | None
    send: SendRecord
    outcome: PeriodOutcome
    read_ms: float
    infer_ms: float
    gate_ms: float
    step_ms: float
    consecutive_holds: int


@dataclass
class EpisodeResult:
    steps: list[StepRecord]
    actions: int
    success: bool | None
    invalidated: bool
    aborted_reason: str | None
    hold_frames: int
    boundaries: list[int]


class EpisodeBackend(Protocol):
    name: str

    def read_act(self) -> np.ndarray: ...
    def observe(self, step: int) -> Observation: ...
    def send(self, step: int, decision: HoldDecision) -> SendRecord: ...
    def wait_next_period(self, step: int) -> PeriodOutcome: ...
    def close(self) -> None: ...


class ConsecutiveHoldAbort(RuntimeError):
    pass


def run_episode(
    backend: EpisodeBackend,
    policy: Any,
    *,
    bench: Any,
    max_actions: int,
    max_consecutive_holds: int | None = 15,
    on_step: Callable[[StepRecord], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EpisodeResult:
    """Run one episode of ``policy`` on ``backend`` under the bench gate; see the module docstring."""
    joint_map = bench.joint_map_object
    shoulder_floor = bench.shoulder_floor
    reset_method = getattr(policy, "reset", None)
    if callable(reset_method):
        reset_method()
    steps: list[StepRecord] = []
    boundaries: list[int] = []
    consecutive = 0
    holds = 0
    aborted: str | None = None
    success: bool | None = None
    invalidated = False
    for step in range(int(max_actions)):
        t_step = clock()
        current = np.asarray(backend.read_act(), dtype=np.float32)
        read_ms = (clock() - t_step) * 1e3
        boundary = bool(getattr(policy, "needs_frame", True))
        observation = backend.observe(step) if boundary else None
        if boundary:
            boundaries.append(step)
        t_infer = clock()
        policy_act = policy.predict(None if observation is None else observation.image, current)
        infer_ms = (clock() - t_infer) * 1e3
        t_gate = clock()
        decision = bench_hold_decision(current, policy_act, shoulder_floor=shoulder_floor, joint_map=joint_map)
        gate_ms = (clock() - t_gate) * 1e3
        if decision.held:
            holds += 1
            consecutive += 1
        else:
            consecutive = 0
        send = backend.send(step, decision)
        outcome = backend.wait_next_period(step)
        record = StepRecord(
            step=step, boundary=boundary, current_act=current, decision=decision, observation=observation,
            send=send, outcome=outcome, read_ms=read_ms, infer_ms=infer_ms, gate_ms=gate_ms,
            step_ms=(clock() - t_step) * 1e3, consecutive_holds=consecutive,
        )
        steps.append(record)
        if on_step is not None:
            on_step(record)
        if outcome.invalidated:
            invalidated = True
        if outcome.done:
            success = outcome.success
            break
        if max_consecutive_holds is not None and consecutive >= max_consecutive_holds:
            aborted = "consecutive_holds"
            break
    return EpisodeResult(steps=steps, actions=len(steps), success=success, invalidated=invalidated,
                         aborted_reason=aborted, hold_frames=holds, boundaries=boundaries)


# ----------------------------------------------------------------------------- approach phase

APPROACH_MAX_UNITS_PER_STEP = 0.5
APPROACH_LEAD_CAP_UNITS = 10.0
APPROACH_SETTLE_UNITS = 1.0
APPROACH_MAX_RELATIVE_TARGET = 10.01


@dataclass(frozen=True)
class ApproachStep:
    measured_physical: np.ndarray
    target_physical: np.ndarray
    decision: HoldDecision
    settled: bool
    reason: str


def next_approach_target(measured_physical, previous_target, reset_physical, *, max_step=APPROACH_MAX_UNITS_PER_STEP,
                         lead_cap=APPROACH_LEAD_CAP_UNITS) -> np.ndarray:
    """Ramp every joint's command toward the reset by at most ``max_step`` units per call.

    The ramp runs on the *previous target*, not the measurement, so a joint held
    back by gravity keeps receiving a steady command; the lead over the
    measured value is capped at ``lead_cap`` so the command never runs far
    ahead of the arm. A joint already within the step of the reset gets the
    reset value exactly. Targets never move away from the reset.
    """
    measured = np.asarray(measured_physical, dtype=np.float64)
    previous = np.asarray(previous_target, dtype=np.float64)
    reset = np.asarray(reset_physical, dtype=np.float64)
    direction = np.sign(reset - previous)
    target = previous + direction * np.minimum(np.abs(reset - previous), max_step)
    # Lead cap: do not command further than lead_cap beyond what the arm has reached, in the ramp direction.
    capped = np.where(direction > 0, np.minimum(target, measured + lead_cap), np.maximum(target, measured - lead_cap))
    # Never move away from the reset relative to the previous target.
    away = (direction > 0) & (capped < previous) | (direction < 0) & (capped > previous)
    capped = np.where(away, previous, capped)
    return capped.astype(np.float32)


def approach_step(measured_act, previous_target_physical, bench, *, calibration=None) -> ApproachStep:
    """One gated approach command; raises on any refusal other than the shoulder starting below the floor."""
    measured_physical = act_to_physical_normalized(np.asarray(measured_act, dtype=np.float32))
    reset = np.asarray(bench.reset_physical, dtype=np.float32)
    if not np.all(np.isfinite(measured_physical)):
        raise RuntimeError("nonfinite measured pose during approach")
    target = next_approach_target(measured_physical, previous_target_physical, reset)
    if np.any(target[1] < bench.shoulder_floor):
        raise RuntimeError("approach target would command the shoulder below the floor")
    decision = bench_hold_decision(
        measured_act, physical_normalized_to_act(target), shoulder_floor=bench.shoulder_floor,
        joint_map=bench.joint_map_object, max_relative_target=APPROACH_MAX_RELATIVE_TARGET, calibration=calibration,
        hold_on_any_mask=True,
    )
    if decision.held:
        # The only tolerated mask is the elbow's MuJoCo envelope while it travels inward from gravity rest.
        masks = decision.evaluation
        other = (np.any(masks.act_clip_mask) or np.any(masks.physical_clip_mask) or np.any(masks.relative_limit_mask)
                 or np.any(np.delete(masks.mujoco_clip_mask, 2)))
        if other:
            raise RuntimeError(f"approach refused: {decision.hold_reason}")
    settled = bool(np.all(np.abs(measured_physical - reset) <= APPROACH_SETTLE_UNITS))
    return ApproachStep(measured_physical=measured_physical, target_physical=target, decision=decision,
                        settled=settled, reason=decision.hold_reason)


def approach_plan(measured_act, bench) -> dict[str, Any]:
    """Human-readable plan for the approach: per-joint deltas and the estimated duration."""
    measured = act_to_physical_normalized(np.asarray(measured_act, dtype=np.float32))
    reset = np.asarray(bench.reset_physical, dtype=np.float32)
    delta = reset - measured
    steps = float(np.max(np.abs(delta)) / APPROACH_MAX_UNITS_PER_STEP)
    return dict(
        measured_physical=[round(float(v), 3) for v in measured],
        reset_physical=[round(float(v), 3) for v in reset],
        delta_physical={name: round(float(d), 3) for name, d in zip(JOINT_NAMES, delta)},
        shoulder_below_floor=bool(measured[1] < bench.shoulder_floor),
        estimated_seconds=round(steps / CONTROL_HZ, 1),
        max_units_per_step=APPROACH_MAX_UNITS_PER_STEP, lead_cap_units=APPROACH_LEAD_CAP_UNITS,
        settle_units=APPROACH_SETTLE_UNITS,
    )


__all__ = [
    "APPROACH_LEAD_CAP_UNITS", "APPROACH_MAX_UNITS_PER_STEP", "APPROACH_SETTLE_UNITS", "ApproachStep", "CONTROL_HZ",
    "EpisodeBackend", "EpisodeResult", "Observation", "PeriodOutcome", "SendRecord", "StepRecord",
    "approach_plan", "approach_step", "next_approach_target", "run_episode",
]
