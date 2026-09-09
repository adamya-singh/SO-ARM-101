"""Backend-agnostic runner loop and the gated approach phase (no hardware, no MuJoCo)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import BenchConfig, scene_bench_config
from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act
from so_arm101_v2.physical.runner import (
    APPROACH_LEAD_CAP_UNITS,
    APPROACH_MAX_UNITS_PER_STEP,
    Observation,
    PeriodOutcome,
    SendRecord,
    approach_plan,
    approach_step,
    next_approach_target,
    run_episode,
)

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def _bench() -> BenchConfig:
    return scene_bench_config(SCENE)


class FakePolicy:
    """Chunked policy stand-in: 'network' runs when the buffer is empty, emitting `horizon` commands."""

    def __init__(self, horizon: int, command_fn):
        self.horizon = horizon
        self.command_fn = command_fn
        self.forward_calls: list[int] = []
        self.images: list = []
        self.action_index = 0
        self._buffer: list[np.ndarray] = []

    def reset(self):
        self.action_index = 0
        self._buffer = []

    @property
    def needs_frame(self) -> bool:
        return not self._buffer

    def predict(self, image, current_act):
        if not self._buffer:
            self.forward_calls.append(self.action_index)
            assert image is not None, "network invoked without a frame"
            self._buffer = [self.command_fn(current_act, k) for k in range(self.horizon)]
        self.images.append(image)
        self.action_index += 1
        return self._buffer.pop(0)


class FakeBackend:
    name = "fake"

    def __init__(self, start_act, *, done_at=None):
        self.act = np.asarray(start_act, dtype=np.float32)
        self.sent: list[np.ndarray] = []
        self.decisions = []
        self.observed: list[int] = []
        self.done_at = done_at

    def read_act(self):
        return self.act.copy()

    def observe(self, step):
        self.observed.append(step)
        return Observation(image=np.zeros((256, 256, 3), np.uint8), frame_seq=step, frame_time=0.0, age_s=0.0, raw_rgb=None, observe_ms=0.0)

    def send(self, step, decision):
        self.sent.append(decision.sent_physical.copy())
        self.decisions.append(decision)
        # The arm tracks the executed command perfectly.
        self.act = decision.executed_act.copy()
        return SendRecord(sent_physical=decision.sent_physical, returned_physical=None, send_ms=0.0)

    def wait_next_period(self, step):
        return PeriodOutcome(done=(self.done_at is not None and step + 1 >= self.done_at), success=True)

    def close(self):
        pass


def test_network_runs_only_at_chunk_boundaries_and_progress_stays_in_lockstep():
    bench = _bench()
    start = physical_normalized_to_act(np.asarray(bench.reset_physical, dtype=np.float32))
    horizon = 4
    policy = FakePolicy(horizon, lambda current, k: current + np.float32(0.001) * (k + 1))
    backend = FakeBackend(start)
    result = run_episode(backend, policy, bench=bench, max_actions=2 * horizon + 1)
    assert policy.forward_calls == [0, horizon, 2 * horizon]
    assert backend.observed == [0, horizon, 2 * horizon]
    assert result.boundaries == [0, horizon, 2 * horizon]
    assert policy.action_index == result.actions == 2 * horizon + 1
    assert [image is None for image in policy.images] == [step % horizon != 0 for step in range(2 * horizon + 1)]
    assert result.hold_frames == 0 and result.aborted_reason is None and result.success is None


def test_consecutive_holds_abort_with_the_present_pose_sent():
    bench = _bench()
    start = physical_normalized_to_act(np.asarray(bench.reset_physical, dtype=np.float32))
    policy = FakePolicy(90, lambda current, k: current + np.array([1.0, 0, 0, 0, 0, 0], np.float32))  # a 1 rad pan jump every step
    backend = FakeBackend(start)
    result = run_episode(backend, policy, bench=bench, max_actions=480, max_consecutive_holds=15)
    assert result.aborted_reason == "consecutive_holds" and result.actions == 15 and result.hold_frames == 15
    assert all(d.held and d.hold_reason == "relative_limit:shoulder_pan" for d in backend.decisions)
    present = act_to_physical_normalized(start)
    assert np.allclose(backend.sent[-1], present, atol=1e-4)
    assert np.array_equal(backend.act, start)  # never moved


def test_episode_stops_when_the_backend_reports_done():
    bench = _bench()
    start = physical_normalized_to_act(np.asarray(bench.reset_physical, dtype=np.float32))
    policy = FakePolicy(90, lambda current, k: current.copy())
    backend = FakeBackend(start, done_at=7)
    result = run_episode(backend, policy, bench=bench, max_actions=480)
    assert result.actions == 7 and result.success is True


def test_next_approach_target_ramps_caps_lead_and_never_backs_away():
    reset = np.array([0.5, -90.8, 91.3, 43.6, -1.2, 13.5], np.float32)
    rest = np.array([0.0, -92.08, 100.0, 40.0, -2.0, 13.5], np.float32)
    target = next_approach_target(rest, rest, reset)
    assert np.all(np.abs(target - rest) <= APPROACH_MAX_UNITS_PER_STEP + 1e-6)
    assert target[1] > rest[1] and target[2] < rest[2] and target[5] == reset[5]
    # Ramp on the previous target while the arm lags: the lead is capped.
    previous = rest.copy()
    for _ in range(100):
        previous = next_approach_target(rest, previous, reset)
    assert previous[2] >= rest[2] - APPROACH_LEAD_CAP_UNITS - 1e-6
    assert previous[1] <= rest[1] + APPROACH_LEAD_CAP_UNITS + 1e-6
    # Converges to the reset once the arm follows.
    previous = rest.copy(); measured = rest.copy()
    for _ in range(400):
        previous = next_approach_target(measured, previous, reset); measured = previous.copy()
    assert np.allclose(previous, reset, atol=1e-5)
    # Overshoot in the measurement never produces a command away from the reset.
    beyond = reset + np.array([0, 0.3, -0.3, 0, 0, 0], np.float32)
    assert np.all(np.abs(next_approach_target(beyond, reset, reset) - reset) < 1e-6)


def test_approach_step_gates_each_command_and_reports_settling():
    bench = _bench()
    reset = np.asarray(bench.reset_physical, dtype=np.float32)
    rest = np.array([0.0, -92.08, 100.0, 40.0, -2.0, 13.5], np.float32)
    first = approach_step(physical_normalized_to_act(rest), rest, bench)
    assert not first.settled and first.target_physical[1] > bench.shoulder_floor  # the first shoulder target is already above the floor
    assert not first.decision.held or "mujoco_clip:elbow_flex" == first.decision.hold_reason
    at_reset = approach_step(physical_normalized_to_act(reset), reset, bench)
    assert at_reset.settled and not at_reset.decision.held
    plan = approach_plan(physical_normalized_to_act(rest), bench)
    assert plan["shoulder_below_floor"] and plan["estimated_seconds"] > 0
    # A previous target that would keep the shoulder below the floor is refused before any gate call.
    with pytest.raises(RuntimeError, match="below the floor"):
        approach_step(physical_normalized_to_act(rest), np.array([0.0, -93.0, 100, 40, -2, 13.5], np.float32), bench)
    huge = rest.copy(); huge[0] = 60.0
    with pytest.raises(RuntimeError, match="approach refused"):
        approach_step(physical_normalized_to_act(huge), np.array([0.0, -92.0, 100, 40, -2, 13.5], np.float32) - np.array([15, 0, 0, 0, 0, 0], np.float32), bench)
