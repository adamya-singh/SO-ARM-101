"""The shared bench gate (`bench_hold_decision`) equals the simulator adapter's rule and pins the pixel-free rollout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import BenchConfig, scene_bench_config
from so_arm101_v2.contracts.physical import act_to_physical_normalized, bench_hold_decision, physical_normalized_to_act

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
LENS_RUN = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909"
CHECKPOINT = LENS_RUN / "models/vision_h90/5e96018881d4140f/model.pt"
BLACK_TELEMETRY = LENS_RUN / "evaluations/nominal/policies/bench_pick_replace_v1_nominal_eval_3de9de823621/telemetry/vision_black.nominal.repeat0.json"


def _bench() -> BenchConfig:
    return scene_bench_config(SCENE)


def _reset_act(bench: BenchConfig) -> np.ndarray:
    return physical_normalized_to_act(np.asarray(bench.reset_physical, dtype=np.float32))


def test_hold_decision_cases_match_the_bench_rule():
    bench = _bench()
    current = _reset_act(bench)
    kwargs = dict(shoulder_floor=bench.shoulder_floor, joint_map=bench.joint_map_object)
    # In range: executed is the float32 round trip of the request, not held.
    small = current + np.float32(0.01)
    d = bench_hold_decision(current, small, **kwargs)
    assert not d.held and d.hold_reason == "" and not d.nonfinite
    assert np.array_equal(d.executed_act, physical_normalized_to_act(d.evaluation.relative_limited_physical))
    assert np.array_equal(d.sent_physical, d.evaluation.relative_limited_physical)
    # Relative limit: held, executed is the current pose, sent is the measured present pose.
    jump = current.copy(); jump[0] += 1.0
    d = bench_hold_decision(current, jump, **kwargs)
    assert d.held and d.hold_reason == "relative_limit:shoulder_pan" and d.delta_limiter_activated
    assert np.array_equal(d.executed_act, current) and np.array_equal(d.sent_physical, d.evaluation.current_physical)
    # Shoulder floor (physical clip): held with the joint named.
    floor = current.copy(); floor[1] = physical_normalized_to_act(np.array([0, -93.0, 0, 0, 0, 10], dtype=np.float32))[1]
    d = bench_hold_decision(current, floor, **kwargs)
    assert d.held and "physical_clip:shoulder_lift" in d.hold_reason and d.command_bound_violation
    # Nonfinite: request replaced by current, executed current, reason says so.
    d = bench_hold_decision(current, np.array([np.nan] * 6), **kwargs)
    assert d.nonfinite and d.hold_reason == "nonfinite" and np.array_equal(d.requested_act, current)
    assert np.array_equal(d.executed_act, physical_normalized_to_act(d.evaluation.relative_limited_physical))
    # Legacy lane (no hold): the executed command is the relative-limited one.
    d = bench_hold_decision(current, jump, shoulder_floor=None, joint_map=None, hold_on_any_mask=False)
    assert not d.held and d.hold_reason == "relative_limit:shoulder_pan"
    assert abs(act_to_physical_normalized(d.executed_act)[0] - act_to_physical_normalized(current)[0] - 20.0) < 1e-3


def test_hold_decision_equals_the_adapter_on_the_live_scene():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    adapter = MujocoTaskAdapter(SCENE)
    try:
        suite = bench_suite(adapter.bench, [(0, 0)], label="hold_parity", repeats=1)
        adapter.reset(suite.scenarios[0])
        rng = np.random.default_rng(3)
        for _ in range(40):
            current = adapter.current_act()
            request = current + rng.normal(0, 0.15, 6).astype(np.float32)
            if rng.random() < 0.2:
                request[rng.integers(0, 6)] = np.nan
            decision = bench_hold_decision(current, request, shoulder_floor=adapter.bench.shoulder_floor, joint_map=adapter.joint_map)
            command = adapter.apply_policy_command(request)
            assert np.array_equal(command.executed_act, decision.executed_act)
            assert np.array_equal(command.requested_act, decision.requested_act)
            assert command.nonfinite_command == decision.nonfinite
            assert (command.command_bound_violation or command.delta_limiter_activated) == decision.held
            adapter.advance_control_period()
    finally:
        adapter.close()


@pytest.mark.skipif(not (CHECKPOINT.exists() and BLACK_TELEMETRY.exists()), reason="lens-run artifacts not present")
def test_black_image_rollout_reproduces_the_stored_telemetry_rows():
    """Pixel-free rollout (no rendering, deterministic): the adapter refactor must not change a single row."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    from so_arm101_v2.contracts import load_pick_place_contract
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step
    torch.set_num_threads(1)
    stored = json.loads(BLACK_TELEMETRY.read_text())["rows"]
    adapter = MujocoTaskAdapter(SCENE)
    try:
        suite = bench_suite(adapter.bench, [(0, 0)], label="nominal_eval", repeats=3)
        contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
        adapter.reset(suite.scenarios[0])
        policy = VisionChunkedPolicy(CHECKPOINT, black_image=True, clamp_channels=(5,))
        policy.reset(adapter)
        state = PickPlaceEvaluationState()
        for action in range(contract.max_actions):
            raw, _, current = adapter.observation(render_pixels=False)
            requested = policy.predict(raw, current, adapter)
            command = adapter.apply_policy_command(requested)
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m)
            state, evaluation = evaluate_pick_place_step(contract, measurement, state)
            row = stored[action]
            assert row["current_act"] == current.tolist()
            assert row["requested_act"] == command.requested_act.tolist()
            assert row["executed_act"] == command.executed_act.tolist()
            assert row["robot_qpos"] == adapter.mujoco_qpos().tolist()
            if evaluation.terminated or evaluation.truncated:
                assert action + 1 == len(stored)
                break
    finally:
        adapter.close()
