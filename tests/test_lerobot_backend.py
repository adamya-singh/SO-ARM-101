"""Real-arm backend with mocks: colour order, resampler verification, stale frames, driver modification, timing."""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.contracts.physical import act_to_physical_normalized, bench_hold_decision, physical_normalized_to_act
from so_arm101_v2.physical.lerobot_backend import (
    DriverModifiedCommand,
    LeRobotBackend,
    StaleFrame,
    TimingViolation,
    fast_area_resampler,
    observation_from_bgr,
    verify_fast_resampler,
)

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def _lens():
    return scene_bench_config(SCENE).lens_model


class FakeGrabber:
    def __init__(self, frames, stamps):
        self.frames, self.stamps = list(frames), list(stamps)
        self.index = 0

    def latest(self):
        i = min(self.index, len(self.frames) - 1)
        return i + 1, self.stamps[i], self.frames[i]

    def advance(self):
        self.index += 1


def _robot(present_physical):
    robot = Mock()
    robot.bus.sync_read.return_value = {name: float(v) for name, v in zip(JOINT_NAMES, present_physical)}
    robot.send_action.side_effect = lambda action: dict(action)
    return robot


def test_bgr_frames_reach_the_policy_as_rgb():
    pytest.importorskip("cv2")
    lens = _lens()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, :, 0] = 200  # B
    frame[:, :, 1] = 100  # G
    frame[:, :, 2] = 50   # R
    exact = observation_from_bgr(frame, lens)
    fast = fast_area_resampler(lens)(frame)
    assert exact.shape == (256, 256, 3) and exact.dtype == np.uint8
    assert np.unique(exact.reshape(-1, 3), axis=0).tolist() == [[50, 100, 200]]
    assert np.array_equal(fast, exact)


def test_720p_source_takes_the_same_contract_path():
    pytest.importorskip("cv2")
    lens = _lens()
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    evidence = verify_fast_resampler(lens, [frame])
    assert evidence["bit_identical"] and evidence["source_shape"] == [720, 1280, 3]
    assert observation_from_bgr(frame, lens).shape == (256, 256, 3)
    with pytest.raises(ValueError, match="aspect ratio"):
        lens.real_operator(source_size=(1280, 960))


def test_fast_resampler_verification_passes_on_noise_and_fails_when_perturbed():
    pytest.importorskip("cv2")
    lens = _lens()
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(2)]
    evidence = verify_fast_resampler(lens, frames)
    assert evidence["bit_identical"] and evidence["frames_checked"] == 2
    bad = lambda frame: np.clip(fast_area_resampler(lens)(frame).astype(np.int16) + 1, 0, 255).astype(np.uint8)
    with pytest.raises(RuntimeError, match="differs"):
        verify_fast_resampler(lens, frames[:1], resampler=bad)


def _backend(lens, robot, grabber, **kw):
    clock = [0.0]
    wall = [1000.0]
    slept = []

    def advance_clock(seconds):
        clock[0] += seconds
        wall[0] += seconds

    backend = LeRobotBackend(robot, grabber, lens=lens, clock=lambda: clock[0], wall=lambda: wall[0],
                             sleep=lambda s: (slept.append(s), advance_clock(s)), resampler=lambda f: np.zeros((256, 256, 3), np.uint8), **kw)
    return backend, advance_clock, slept, wall


def test_stale_or_reused_frames_abort_a_boundary_without_sending():
    lens = _lens()
    bench = scene_bench_config(SCENE)
    present = np.asarray(bench.reset_physical, dtype=np.float32)
    robot = _robot(present)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    grabber = FakeGrabber([frame, frame], [1000.0, 1000.0])
    backend, advance, _, wall = _backend(lens, robot, grabber)
    backend.read_act()
    obs = backend.observe(0)
    assert obs.frame_seq == 1 and obs.age_s == 0.0 and obs.raw_rgb.shape == (1080, 1920, 3)
    with pytest.raises(StaleFrame, match="already used"):
        backend.observe(90)  # grabber did not advance
    grabber.advance()
    advance(0.150)
    with pytest.raises(StaleFrame, match="ms old"):
        backend.observe(90)
    robot.send_action.assert_not_called()
    robot.bus.enable_torque.assert_not_called()


def test_send_uses_the_gated_physical_command_and_detects_driver_changes():
    lens = _lens()
    bench = scene_bench_config(SCENE)
    present = np.asarray(bench.reset_physical, dtype=np.float32)
    robot = _robot(present)
    backend, _, _, _ = _backend(lens, robot, FakeGrabber([np.zeros((1080, 1920, 3), np.uint8)], [1000.0]))
    current = backend.read_act()
    assert np.allclose(current, physical_normalized_to_act(present), atol=1e-5)
    decision = bench_hold_decision(current, current + np.float32(0.01), shoulder_floor=bench.shoulder_floor, joint_map=bench.joint_map_object)
    record = backend.send(0, decision)
    sent = robot.send_action.call_args.args[0]
    assert [sent[f"{n}.pos"] for n in JOINT_NAMES] == pytest.approx(decision.sent_physical.tolist(), abs=1e-6)
    assert np.allclose(record.returned_physical, decision.sent_physical, atol=1e-5)
    # A held decision sends the measured present pose.
    held = bench_hold_decision(current, current + np.array([1.0, 0, 0, 0, 0, 0], np.float32), shoulder_floor=bench.shoulder_floor, joint_map=bench.joint_map_object)
    backend.send(1, held)
    sent = robot.send_action.call_args.args[0]
    assert np.allclose([sent[f"{n}.pos"] for n in JOINT_NAMES], act_to_physical_normalized(current), atol=1e-4)
    robot.send_action.side_effect = lambda action: {k: v + 0.5 for k, v in action.items()}
    with pytest.raises(DriverModifiedCommand):
        backend.send(2, decision)


def test_engage_seeds_the_goal_before_enabling_torque_and_never_disables_it():
    lens = _lens()
    bench = scene_bench_config(SCENE)
    present = np.asarray(bench.reset_physical, dtype=np.float32)
    robot = _robot(present)
    events = []
    robot.bus.sync_write.side_effect = lambda *a, **k: events.append("goal")
    robot.bus.enable_torque.side_effect = lambda *a, **k: events.append("torque")
    backend, _, _, _ = _backend(lens, robot, FakeGrabber([np.zeros((1080, 1920, 3), np.uint8)], [1000.0]))
    backend.engage(present)
    assert events == ["goal", "torque"] and backend.engaged
    robot.bus.disable_torque.assert_not_called()


def test_timing_schedule_sleeps_to_absolute_deadlines_and_aborts_on_violations():
    lens = _lens()
    bench = scene_bench_config(SCENE)
    present = np.asarray(bench.reset_physical, dtype=np.float32)
    robot = _robot(present)
    backend, advance, slept, _ = _backend(lens, robot, FakeGrabber([np.zeros((1080, 1920, 3), np.uint8)], [1000.0]), max_consecutive_overruns=2)
    backend.read_act(); advance(0.010)
    outcome = backend.wait_next_period(0)
    assert not outcome.overrun and slept[-1] == pytest.approx(1 / 30 - 0.010)
    # A 20 ms overrun (less than half a period) is logged, not re-anchored.
    backend.read_act(); advance(1 / 30 + 0.010)
    outcome = backend.wait_next_period(1)
    assert outcome.overrun and not outcome.reanchored and outcome.lateness_ms == pytest.approx(10.0, abs=0.5)
    # A 25 ms lateness (more than half a period) re-anchors.
    backend.read_act(); advance(1 / 30 + 0.015)
    outcome = backend.wait_next_period(2)
    assert outcome.overrun and outcome.reanchored and backend.reanchors == 1
    # Third consecutive overrun exceeds the limit of two.
    backend.read_act(); advance(1 / 30 + 0.005)
    with pytest.raises(TimingViolation, match="consecutive overruns"):
        backend.wait_next_period(3)
    # Any single step over 100 ms aborts.
    backend2, advance2, _, _ = _backend(lens, robot, FakeGrabber([np.zeros((1080, 1920, 3), np.uint8)], [1000.0]))
    backend2.read_act(); advance2(0.120)
    with pytest.raises(TimingViolation, match="took"):
        backend2.wait_next_period(0)
