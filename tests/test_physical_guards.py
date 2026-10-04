"""Episode guards of the physical runner (no hardware): workspace, survey, stall, brightness, and the runner hook."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act
from so_arm101_v2.physical.guards import SURVEY_ONLY, EpisodeGuards, GuardLimits, frame_brightness
from so_arm101_v2.physical.runner import run_episode
from test_physical_runner import FakeBackend, FakePolicy

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def _bench():
    return scene_bench_config(SCENE)


def _decision(physical, held=False):
    return SimpleNamespace(sent_physical=np.asarray(physical, np.float32), held=held)


def _lit(value=120):
    return SimpleNamespace(image=np.full((256, 256, 3), value, np.uint8))


def test_brightness_floor_separates_lights_off_from_the_lit_bench():
    assert not frame_brightness(np.full((256, 256, 3), 2, np.uint8))["ok"]          # episode_16: mean 1.4, p99 3
    dim = np.full((256, 256, 3), 30, np.uint8); dim[:40] = 200                       # a dim bench with the towel in view
    assert frame_brightness(dim)["ok"]


def test_workspace_floor_and_box_use_the_commanded_pose():
    bench = _bench()
    guards = EpisodeGuards(SCENE, bench)
    reset = np.asarray(bench.reset_physical, np.float32)
    measured = physical_normalized_to_act(reset)
    assert guards.before_send(0, measured, _decision(reset), _lit()) is None
    # The episode_20 command at step 240: jaws 13 mm above the table beside the base.
    low = np.array([60.4, -24.1, 76.1, 55.1, 35.4, 33.0], np.float32)
    assert guards.before_send(1, measured, _decision(low), None) == "workspace_height"
    assert guards.trip["step"] == 1 and guards.trip["detail"]["tip_height_m"] < 0.022
    # A held step sends the measured pose: never a workspace trip.
    guards.reset()
    assert guards.before_send(0, measured, _decision(low, held=True), None) is None
    # The jaw midpoint must stay inside the box: the reset pose sits at y = 0.184 m, outside a box that ends at 0.17.
    guards.reset()
    assert guards.box == pytest.approx((-0.2378, 0.2378, 0.0804, 0.3694), abs=1e-3)
    guards.box = (guards.box[0], guards.box[1], guards.box[2], 0.17)
    assert guards.before_send(0, measured, _decision(reset), None) == "workspace_xy"


def test_survey_checks_pose_then_target_and_survey_only_stops_cleanly():
    bench = _bench()
    viewing = bench.joint_map_object.mujoco_to_act(np.asarray(bench.viewing_qpos, np.float32))
    sent = act_to_physical_normalized(viewing)
    towel = np.full((256, 256, 3), 20, np.uint8)
    towel[60:140, 90:170] = 230           # the towel
    towel[85:115, 115:145] = 15           # the cube on it
    # At the viewing pose with the towel and cube in view: passes; survey-only stops there.
    guards = EpisodeGuards(SCENE, bench)
    assert guards.before_send(bench.observation_steps, viewing, _decision(sent), SimpleNamespace(image=towel)) is None
    assert guards.survey["pose_ok"] and guards.survey["target_ok"]
    staged = EpisodeGuards(SCENE, bench, stop_after_survey=True)
    assert staged.before_send(bench.observation_steps, viewing, _decision(sent), SimpleNamespace(image=towel)) == SURVEY_ONLY
    # Elbow 13 units short (episode_20): refused on the pose before the frame is even considered.
    short = act_to_physical_normalized(viewing).copy(); short[2] += 12.7
    guards.reset()
    assert guards.before_send(bench.observation_steps, physical_normalized_to_act(short), _decision(short), SimpleNamespace(image=towel)) == "survey_pose_not_reached"
    assert guards.survey["joint"] == "elbow_flex"
    # At the pose but the frame shows only the mat.
    mat = np.full((256, 256, 3), 60, np.uint8)
    guards.reset()
    assert guards.before_send(bench.observation_steps, viewing, _decision(sent), SimpleNamespace(image=mat)) == "survey_target_not_visible"
    advisory = EpisodeGuards(SCENE, bench, survey_gate=False)
    assert advisory.before_send(bench.observation_steps, viewing, _decision(sent), SimpleNamespace(image=mat)) is None
    assert advisory.survey["target_ok"] is False
    # Record-only also covers the pose (a fixed-view policy stays at the reset pose).
    advisory.reset()
    assert advisory.before_send(bench.observation_steps, physical_normalized_to_act(short), _decision(short), SimpleNamespace(image=mat)) is None
    assert advisory.survey["pose_ok"] is False


def test_stall_needs_a_large_error_that_is_not_closing():
    bench = _bench()
    reset = np.asarray(bench.reset_physical, np.float32)
    measured = physical_normalized_to_act(reset)
    command = reset.copy(); command[2] -= 14.0                      # elbow commanded 14 units away, arm never moves
    guards = EpisodeGuards(SCENE, bench)
    reasons = [guards.before_send(step, measured, _decision(command), None) for step in range(12)]
    assert reasons[:10] == [None] * 10 and reasons[10] == "stall:elbow_flex"
    # The same error while the joint is moving toward the command is servo lag, not a stall.
    guards.reset()
    for step in range(30):
        pose = reset.copy(); pose[2] -= 1.0 * step
        command = pose.copy(); command[2] -= 14.0
        assert guards.before_send(step, physical_normalized_to_act(pose), _decision(command), None) is None
    # A resting error under the limit (a compensated command resting 10 units past a loaded elbow) never trips.
    guards.reset()
    sag = reset.copy(); sag[2] -= 10.0
    assert all(guards.before_send(step, measured, _decision(sag), None) is None for step in range(40))


def test_runner_sends_the_measured_pose_and_stops_when_a_guard_trips():
    bench = _bench()
    start = physical_normalized_to_act(np.asarray(bench.reset_physical, dtype=np.float32))
    policy = FakePolicy(4, lambda current, k: current + np.float32(0.001) * (k + 1))

    class TripAt:
        def __init__(self, step):
            self.step, self.resets = step, 0

        def reset(self):
            self.resets += 1

        def before_send(self, step, current, decision, observation):
            return "workspace_height" if step == self.step else None

    backend = FakeBackend(start)
    guard = TripAt(5)
    result = run_episode(backend, policy, bench=bench, max_actions=40, guard=guard)
    assert result.aborted_reason == "guard:workspace_height" and result.actions == 6 and guard.resets == 1
    assert np.allclose(backend.sent[-1], act_to_physical_normalized(backend.decisions[-1].executed_act), atol=1e-4)
    assert np.allclose(backend.decisions[-1].executed_act, backend.decisions[-2].executed_act, atol=1e-6)   # held where it was
    # No guard: the loop is unchanged.
    assert run_episode(FakeBackend(start), FakePolicy(4, lambda c, k: c), bench=bench, max_actions=8).aborted_reason is None


def test_relax_reseeds_the_goal_at_the_present_pose_without_touching_torque():
    from unittest.mock import Mock
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    from so_arm101_v2.physical.lerobot_backend import LeRobotBackend

    bench = _bench()
    robot = Mock()
    present = {name: float(v) for name, v in zip(JOINT_NAMES, [1.0, -30.0, 80.0, 60.0, 2.0, 10.0])}
    robot.bus.sync_read.return_value = present
    backend = LeRobotBackend(robot, Mock(), lens=bench.lens_model, resampler=lambda f: f)
    out = backend.relax()
    robot.bus.sync_write.assert_called_once_with("Goal_Position", present)
    robot.bus.disable_torque.assert_not_called()
    assert out.tolist() == pytest.approx(list(present.values()))


def test_runner_passes_every_policy_target_through_the_assist_before_the_gate():
    bench = _bench()
    start = physical_normalized_to_act(np.asarray(bench.reset_physical, dtype=np.float32))

    class Offset:
        def __init__(self):
            self.resets, self.seen = 0, []

        def reset(self):
            self.resets += 1

        def command(self, target, measured):
            self.seen.append(np.asarray(target).copy())
            out = np.asarray(target, np.float32).copy()
            out[0] += np.float32(0.01)
            return out

    assist = Offset()
    backend = FakeBackend(start)
    result = run_episode(backend, FakePolicy(4, lambda current, k: current.copy()), bench=bench, max_actions=6, assist=assist)
    assert assist.resets == 1 and len(assist.seen) == 6 and result.aborted_reason is None
    assert all(d.policy_act[0] == pytest.approx(s[0] + 0.01, abs=1e-6) for d, s in zip(backend.decisions, assist.seen))
