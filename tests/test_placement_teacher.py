"""The privileged teacher on yawed and moved squares: strict grasps and complete episodes."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.contracts.placement import PlacementRegime

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def _run_teacher(adapter):
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    controller = PrivilegedStagedController()
    controller.reset(adapter)
    contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
    state = PickPlaceEvaluationState()
    strict = safety = 0
    for _ in range(contract.max_actions):
        command = adapter.apply_policy_command(controller.predict(None, adapter.current_act(), adapter))
        adapter.advance_control_period()
        measurement, _ = adapter.pick_place_measurement(command)
        state, result = evaluate_pick_place_step(contract, measurement, state)
        strict += measurement.pickup.strict_bilateral_grasp
        safety += (measurement.pickup.unsafe_contact + measurement.pickup.command_bound_violation
                   + measurement.pickup.delta_limiter_activated + measurement.pickup.nonfinite_command)
        if result.terminated or result.truncated or result.invalidated:
            break
    return result, strict, safety, controller


@pytest.mark.parametrize("square,yaw_deg", [((0.0, 0.2805353), 45.0), ((0.0, 0.2805353), -30.0), ((0.12, 0.20), 20.0), ((-0.10, 0.24), -45.0)])
def test_teacher_completes_on_yawed_and_moved_squares(tmp_path, square, yaw_deg):
    pytest.importorskip("mujoco")
    from tools.prepare_bench_scene import prepare
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    config = replace(scene_bench_config(LIVE_SCENE), placement=PlacementRegime().identity())
    scene = prepare(tmp_path / "scene", config)
    adapter = MujocoTaskAdapter(scene)
    try:
        scenario = bench_suite(adapter.bench, [(0, 0)], label="t", repeats=1, square_centers=[square], square_yaws=[np.deg2rad(yaw_deg)]).scenarios[0]
        adapter.reset(scenario)
        result, strict, safety, controller = _run_teacher(adapter)
        # The cube is 4-fold symmetric: the grasp yaw equals the square yaw modulo 90 degrees and lies within +-45.
        assert abs(controller.grasp_yaw_rad) <= np.pi / 4 + 1e-9
        assert (controller.grasp_yaw_rad - np.deg2rad(yaw_deg)) % (np.pi / 2) == pytest.approx(0.0, abs=1e-6) or \
               (controller.grasp_yaw_rad - np.deg2rad(yaw_deg)) % (np.pi / 2) == pytest.approx(np.pi / 2, abs=1e-6)
        assert result.success and not result.invalidated, (square, yaw_deg)
        assert strict >= 100 and safety == 0
    finally:
        adapter.close()


def test_legacy_nominal_teacher_has_zero_grasp_yaw():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    adapter = MujocoTaskAdapter(LIVE_SCENE)
    try:
        adapter.reset(bench_suite(adapter.bench, [(0, 0)], label="t", repeats=1).scenarios[0])
        controller = PrivilegedStagedController()
        controller.reset(adapter)
        assert controller.grasp_yaw_rad == 0.0
        normal, depth = controller._targets()
        assert np.array_equal(normal, np.array([1.0, 0.0, 0.0]))
    finally:
        adapter.close()
