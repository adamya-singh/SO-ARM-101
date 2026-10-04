"""Measured servo model: the bench_config servo block, the scene generator, the compensated teacher, the probe and replay policies."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import BenchConfig, scene_bench_config, scene_dependency_hash
from so_arm101_v2.contracts.physical import act_to_physical_normalized, bench_clip_decision, physical_normalized_to_act

mujoco = pytest.importorskip("mujoco")
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

LIVE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1"
SCENE = LIVE / "scene_bench_pick_replace_v1.xml"
SERVO = dict(name="bench_servo", version=1, kp=[24.0, 41.0, 8.0, 60.0, 37.0, 50.0], kv=[2.0, 3.7, 0.83, 6.0, 4.1, 4.473],
             frictionloss=[0.48, 0.6, 0.4, 0.45, 0.26, 0.1], source="test")


def _prepare(tmp_path: Path, servo):
    import prepare_bench_scene
    config = replace(BenchConfig.load(LIVE / "bench_config.json"), servo=servo)
    destination = REPOSITORY_ROOT / "simulation_code/model" / f"_test_{tmp_path.name}"      # mesh paths are relative to the model tree
    return prepare_bench_scene.prepare(destination, config), destination


def test_servo_block_is_validated():
    base = BenchConfig.load(LIVE / "bench_config.json")
    assert replace(base, servo=SERVO).servo["kp"][2] == 8.0
    with pytest.raises(ValueError):
        replace(base, servo=dict(SERVO, kp=[1.0] * 5))
    with pytest.raises(ValueError):
        replace(base, servo=dict(SERVO, kp=[0.0] + SERVO["kp"][1:]))
    with pytest.raises(ValueError):
        replace(base, servo={k: v for k, v in SERVO.items() if k != "source"})


def test_scene_without_a_servo_block_is_regenerated_bit_for_bit_and_with_one_carries_the_gains(tmp_path):
    import shutil
    from so_arm101_v2.simulation.servo_model import ServoModel, apply_servo_model, model_servo
    live_servo = BenchConfig.load(LIVE / "bench_config.json").servo
    if live_servo is None:
        scene, destination = _prepare(tmp_path, None)
        try:
            assert (destination / "bench_config.json").read_bytes() == (LIVE / "bench_config.json").read_bytes()
            assert scene.read_text().replace(destination.name, LIVE.name) == SCENE.read_text()
        finally:
            shutil.rmtree(destination)
    scene, destination = _prepare(tmp_path, SERVO)
    try:
        assert scene_dependency_hash(scene) != scene_dependency_hash(SCENE) or live_servo is not None
        model = mujoco.MjModel.from_xml_path(str(scene))
        servo = model_servo(model)
        assert servo.kp == pytest.approx(SERVO["kp"]) and servo.kv == pytest.approx(SERVO["kv"]) and servo.frictionloss == pytest.approx(SERVO["frictionloss"])
        assert scene_bench_config(scene).servo["source"] == "test"
        # apply_servo_model writes the same numbers onto a loaded model
        plain = mujoco.MjModel.from_xml_path(str(SCENE))
        apply_servo_model(plain, ServoModel(tuple(SERVO["kp"]), tuple(SERVO["kv"]), tuple(SERVO["frictionloss"])))
        assert model_servo(plain) == servo
        with pytest.raises(ValueError):
            ServoModel((1.0,) * 6, (1.0,) * 6, (-0.1,) * 6)
    finally:
        shutil.rmtree(destination)


def test_teacher_needs_and_gets_compensation_on_a_measured_servo_scene(tmp_path):
    import shutil
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController

    scene, destination = _prepare(tmp_path, SERVO)
    try:
        adapter = MujocoTaskAdapter(scene)
        try:
            contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
            scenario = bench_suite(adapter.bench, [(0, 0)], label="servo", repeats=1).scenarios[0]

            def run(compensate: bool):
                adapter.reset(scenario)
                controller = PrivilegedStagedController()
                controller.reset(adapter)
                if not compensate:
                    controller._comp_kp = None
                state, result, lead = PickPlaceEvaluationState(), None, 0.0
                for step in range(contract.max_actions):
                    current = adapter.current_act()
                    command = adapter.apply_policy_command(controller.predict(None, current, adapter))
                    adapter.advance_control_period()
                    if step == adapter.bench.observation_steps - 1:      # resting at the viewing pose
                        lead = float(act_to_physical_normalized(command.executed_act)[2] - act_to_physical_normalized(adapter.current_act())[2])
                        short = float(np.max(np.abs(adapter.mujoco_qpos()[:5] - np.asarray(adapter.bench.viewing_qpos)[:5])))
                    measurement, _ = adapter.pick_place_measurement(command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m)
                    state, result = evaluate_pick_place_step(contract, measurement, state)
                    if result.terminated or result.truncated:
                        break
                return bool(result.success and not result.invalidated), lead, short

            ok, lead, short = run(True)
            assert ok and short < 0.02 and abs(lead) > 2.0          # reaches the viewing pose by commanding the elbow past it
            ok, lead, short = run(False)
            assert not ok and short > 0.05                            # the uncompensated plan stops short, as the real arm did
        finally:
            adapter.close()
    finally:
        shutil.rmtree(destination)


def test_servo_probe_stays_in_the_air_and_inside_the_gate():
    from so_arm101_v2.physical.guards import EpisodeGuards
    from so_arm101_v2.physical.servo_probe import ServoProbePolicy

    bench = scene_bench_config(SCENE)
    policy = ServoProbePolicy(bench)
    guards = EpisodeGuards(SCENE, bench, survey_gate=False)
    assert not policy.needs_frame and [s["name"] for s in policy.segments][:2] == ["survey_move", "wiggle_shoulder_pan"]
    current = physical_normalized_to_act(np.asarray(bench.reset_physical, np.float32))
    for step in range(policy.teacher_horizon):
        decision = bench_clip_decision(current, policy.predict(None, current), shoulder_floor=bench.shoulder_floor, joint_map=bench.joint_map_object)
        assert not decision.held and guards.before_send(step, current, decision, None) is None
        current = decision.executed_act
    assert guards.min_tip_height_m > 0.06 and np.allclose(policy.offsets[-1][:5], policy.offsets[89][:5])       # ends back at the viewing pose


def test_replay_policy_plays_the_recorded_commands_and_reads_two_frames(tmp_path):
    from so_arm101_v2.physical.replay_policy import ReplayPolicy

    bench = scene_bench_config(SCENE)
    rows = [dict(executed_act=[0.01 * k, -2.8, 2.8, 1.4, 0.0, 0.2]) for k in range(120)]
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(dict(rows=rows)))
    policy = ReplayPolicy(path, bench)
    reads = []
    for step in range(130):
        if policy.needs_frame:
            reads.append(step)
        command = policy.predict(None, np.zeros(6, np.float32))
        assert command[0] == pytest.approx(0.01 * min(step, 119))
    assert reads == [0, bench.observation_steps] and policy.teacher_horizon == 120
    path.write_text(json.dumps(dict(rows=[dict(executed_act=[0.0] * 5)])))
    with pytest.raises(ValueError):
        ReplayPolicy(path, bench)


def test_servo_assist_bounds_and_bookkeeping():
    from so_arm101_v2.physical.servo_assist import AssistTuning, ServoAssist

    bench = scene_bench_config(SCENE)
    assist = ServoAssist(SCENE, bench)
    assert assist.source == "servo_model_20261004" and assist.kp[2] < 10 < assist.kp[1]        # the elbow is the soft joint
    viewing = bench.joint_map_object.mujoco_to_act(np.asarray(bench.viewing_qpos, np.float32))
    command = assist.command(viewing, viewing)
    offset = act_to_physical_normalized(command) - act_to_physical_normalized(viewing)
    assert command[5] == viewing[5] and 2.0 < abs(offset[2]) < 12.0 and abs(offset[0]) < 1.0    # gravity term: elbow pushed, pan not
    # At rest with a standing error the integral grows, and stops at its cap.
    short = viewing.copy(); short[2] += 0.2
    first = assist.command(viewing, short)
    for _ in range(200):
        last = assist.command(viewing, short)
    assert abs(last[2] - viewing[2]) > abs(first[2] - viewing[2])
    assert np.max(np.abs(assist._integral)) == pytest.approx(AssistTuning().integral_cap_rad)
    # While the target moves, the error is not integrated (no wind-up through a move).
    assist.reset()
    for k in range(40):
        target = viewing.copy(); target[2] -= 0.01 * k
        assist.command(target, viewing)
    assert np.allclose(assist._integral, 0.0)
    # The command never leads the measured joint by more than the cap, and a nonfinite target passes through to the gate.
    far = viewing.copy(); far[2] -= 1.5
    lead = np.abs(act_to_physical_normalized(assist.command(far, viewing)) - act_to_physical_normalized(viewing))
    assert lead[2] <= AssistTuning().max_lead_units + 1e-3
    assert np.isnan(assist.command(np.full(6, np.nan, np.float32), viewing)).all()
    assert len(assist.record()["max_offset_units"]) == 6
    with pytest.raises(ValueError):
        ServoAssist(SCENE, bench, servo=dict(kp=[1.0] * 5, frictionloss=[0.1] * 6))


def test_servo_assist_makes_a_plain_plan_work_on_the_measured_servo_arm(tmp_path):
    """The measured-servo scene stands in for the arm: the teacher's plain plan fails on it and succeeds through the assist."""
    import shutil
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.physical.servo_assist import AssistedPolicy, ServoAssist
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController

    class Plain(PrivilegedStagedController):
        policy_id = "plain"

        def reset(self, adapter):
            super().reset(adapter)
            self._comp_kp = None            # the plan itself, as a policy trained on stiff actuators would command it

    scene, destination = _prepare(tmp_path, SERVO)
    try:
        adapter = MujocoTaskAdapter(scene)
        try:
            contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
            scenario = bench_suite(adapter.bench, [(0, 0)], label="assist", repeats=1).scenarios[0]

            def run(policy):
                adapter.reset(scenario)
                policy.reset(adapter)
                state, result = PickPlaceEvaluationState(), None
                for _ in range(contract.max_actions):
                    command = adapter.apply_policy_command(policy.predict(None, adapter.current_act(), adapter))
                    adapter.advance_control_period()
                    measurement, _ = adapter.pick_place_measurement(command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m)
                    state, result = evaluate_pick_place_step(contract, measurement, state)
                    if result.terminated or result.truncated:
                        break
                return bool(result.success and not result.invalidated)

            servo = dict(kp=SERVO["kp"], frictionloss=SERVO["frictionloss"], name="test")
            assert not run(Plain())
            assisted = AssistedPolicy(Plain(), ServoAssist(SCENE, adapter.bench, servo=servo))
            assert run(assisted) and assisted.policy_id == "plain.assist"
        finally:
            adapter.close()
    finally:
        shutil.rmtree(destination)
