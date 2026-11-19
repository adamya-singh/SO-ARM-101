"""Counterfactual feature probes (2026-11-18, notes/feature-probes-20261118.md): loop parity, the two teacher controls, each change."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.simulation import probes as P

MODELS = REPOSITORY_ROOT / "simulation_code/model"
NOMINAL = MODELS / "bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml"
SERVO = MODELS / "bench_pick_replace_v1_so101_servo/scene_bench_pick_replace_v1.xml"
EXPERIMENTS = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
HELDOUT = EXPERIMENTS / "so101_benchmark_20261106/so101/cap4800/heldout_suite.path"
STIFF = EXPERIMENTS / "next_round_20261108/scenes/so101_servo_stiff_elbow/scene_bench_pick_replace_v1.xml"
pytestmark = pytest.mark.skipif(not HELDOUT.exists(), reason="needs the SO-101 held-out suite under artifacts/")


@pytest.fixture(scope="module")
def suite():
    from so_arm101_v2.simulation.suites import load_suite_from_path
    return load_suite_from_path(Path(HELDOUT.read_text().strip()))


@pytest.fixture(scope="module")
def nominal():
    from so_arm101_v2.contracts.pick_place import load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    adapter = MujocoTaskAdapter(NOMINAL)
    yield adapter, load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
    adapter.close()


def run(nominal, scenario, kind, name):
    adapter, contract = nominal
    spec = P.ProbeSpec(name)
    return spec, *P.run_probe_rollout(adapter, contract, scenario, P.policy_factory(kind, None, 90, spec), spec)


def paired(nominal, scenarios, kind, name):
    control = [run(nominal, s, kind, "control") for s in scenarios]
    probed = [run(nominal, s, kind, name) for s in scenarios]
    stack = lambda items: {k: np.stack([t[k] for _, _, t in items]) for k in items[0][2]}
    return P.paired_metrics(probed[0][0], [r for _, r, _ in control], stack(control), [r for _, r, _ in probed], stack(probed))


def test_direction_is_a_pure_unit_vector():
    a = P.probe_direction(0, "pose_003")
    assert np.allclose(a, P.probe_direction(0, "pose_003")) and abs(np.linalg.norm(a) - 1.0) < 1e-12
    assert not np.allclose(a, P.probe_direction(1, "pose_003")) and not np.allclose(a, P.probe_direction(0, "pose_004"))
    with pytest.raises(ValueError):
        P.ProbeSpec("nudge")


def test_unprobed_loop_matches_the_certified_rollout(nominal, suite, tmp_path):
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    from so_arm101_v2.simulation.rollout import _run_pick_place_rollout
    scenario = suite.scenarios[1]
    certified = _run_pick_place_rollout(NOMINAL, suite, scenario, 0, "teacher", PrivilegedStagedController, tmp_path, record_video=False)
    _, row, trace = run(nominal, scenario, "teacher", "control")
    assert (row["success"], row["pickup_completed"], row["invalidated"], row["contract_actions"]) == (
        certified.success, certified.pickup_completed, certified.invalidated, certified.actions)
    assert row["event_action"] is None and row["shift_xy_m"] is None
    assert np.allclose(trace["cube"][certified.actions - 1], certified.final_cube_position, atol=1e-6)
    assert trace["jaw"].shape == (480, 3)      # the probe loop keeps stepping after the contract's terminal step


def test_teacher_controls_read_zero_and_one(nominal, suite):
    scenarios = suite.scenarios[1:3]
    blind = paired(nominal, scenarios, "teacher", "shift_placement")
    informed = paired(nominal, scenarios, "teacher_informed", "shift_placement")
    assert abs(blind["follow_gain_mean"]) < 0.2 and blind["reaction_act_median"] == 0.0     # open loop: not one command changes
    assert 0.9 < informed["follow_gain_mean"] < 1.1 and informed["probe_successes"] == len(scenarios)
    assert informed["residual_mm_median"] < 2.0


def test_square_move_is_followed_only_by_the_informed_teacher(nominal, suite):
    scenarios = suite.scenarios[1:2]
    blind = paired(nominal, scenarios, "teacher", "move_square")
    informed = paired(nominal, scenarios, "teacher_informed", "move_square")
    assert blind["fired"] == 1 and abs(blind["follow_gain_mean"]) < 0.1 and blind["on_moved_square"] == 0
    assert 0.9 < informed["follow_gain_mean"] < 1.1 and informed["on_moved_square"] == 1


def test_cube_shift_stays_on_the_square_and_drop_puts_the_cube_back(nominal, suite):
    adapter, _ = nominal
    for scenario in suite.scenarios[:4]:
        adapter.reset(scenario)
        delta = P.cube_shift_on_square(adapter, P.ProbeSpec("shift_cube"), scenario.scenario_id)
        assert abs(np.linalg.norm(delta) - 0.010) < 1e-9
        P.shift_cube(adapter, delta)
        measurement, _ = adapter.pick_place_measurement()
        assert measurement.cube_footprint_inside
    _, row, trace = run(nominal, suite.scenarios[1], "teacher", "drop_cube")
    event = row["event_action"]
    assert event is not None and trace["height"][event - 1] >= P.LIFT_TRIGGER_M
    assert np.linalg.norm(trace["cube"][event] - trace["cube"][0]) < 0.002     # back on its square
    assert row["max_height_gain_after_event_m"] < P.REGRASP_HEIGHT_M           # the open-loop teacher never retries


def test_square_shift_does_not_leak_into_the_next_episode(nominal, suite):
    adapter, _ = nominal
    scenario = suite.scenarios[1]
    adapter.reset(scenario)
    before = adapter.model.geom_pos[adapter._napkin_geom].copy()
    with P.shifted_world(adapter, (0.01, 0.0), (0.01, 0.0)):
        assert np.allclose(adapter.model.geom_pos[adapter._napkin_geom, :2], before[:2] + (0.01, 0.0))
    assert np.array_equal(adapter.model.geom_pos[adapter._napkin_geom], before)
    P.shift_square(adapter, (0.02, 0.0))
    adapter.reset(scenario)
    assert np.array_equal(adapter.model.geom_pos[adapter._napkin_geom], before) and adapter.square_center_xy == tuple(scenario.square_center_xy)


@pytest.mark.skipif(not STIFF.exists(), reason="needs the stiff-elbow diagnostic scene under artifacts/")
def test_swapped_elbow_is_the_other_scene_elbow_and_reset_restores_it(suite, nominal):
    import mujoco
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    adapter = MujocoTaskAdapter(SERVO)
    try:
        stiff = mujoco.MjModel.from_xml_path(str(STIFF))
        actuator, dof = adapter._actuator_ids[P.ELBOW], adapter._joint_dof[P.ELBOW]
        soft = adapter.model.actuator_gainprm[actuator].copy()
        adapter.reset(suite.scenarios[1])
        assert P.swap_elbow(adapter) == "stiff"
        assert not np.allclose(soft, adapter.model.actuator_gainprm[actuator])
        for name in ("actuator_gainprm", "actuator_biasprm"):
            assert np.allclose(getattr(adapter.model, name)[actuator], getattr(stiff, name)[actuator], rtol=1e-4), name
        assert np.isclose(adapter.model.dof_frictionloss[dof], stiff.dof_frictionloss[dof], rtol=1e-4)
        adapter.reset(suite.scenarios[1])
        assert np.array_equal(adapter.model.actuator_gainprm[actuator], soft)
        stiff_adapter = nominal[0]                       # the nominal scene's stiff elbow becomes the measured soft one
        stiff_adapter.reset(suite.scenarios[1])
        assert P.swap_elbow(stiff_adapter) == "soft"
        for name in ("actuator_gainprm", "actuator_biasprm"):
            assert np.allclose(getattr(stiff_adapter.model, name)[actuator], getattr(adapter.model, name)[actuator], rtol=1e-4), name
        assert np.isclose(stiff_adapter.model.dof_frictionloss[dof], adapter.model.dof_frictionloss[dof], rtol=1e-4)
        stiff_adapter.reset(suite.scenarios[1])
        assert np.allclose(stiff_adapter.model.actuator_gainprm[actuator, 0], P.STIFF_ELBOW[0])
    finally:
        adapter.close()


def test_report_verdicts_follow_the_registered_thresholds():
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    from feature_probes import verdict
    assert verdict("shift_placement", dict(follow_gain_mean=0.7, follow_gain_ci95=[0.6, 0.8])) == "present"
    assert verdict("shift_placement", dict(follow_gain_mean=0.55, follow_gain_ci95=[0.1, 0.9])) == "partial"   # interval reaches too low
    assert verdict("shift_placement", dict(follow_gain_mean=0.1, follow_gain_ci95=[0.0, 0.2])) == "absent"
    assert verdict("move_square", dict(follow_gain_mean=None)) == "n/a"
    assert verdict("drop_cube", dict(regrasp_rate=0.02)) == "absent" and verdict("drop_cube", dict(regrasp_rate=0.6)) == "present"
    assert verdict("swap_elbow", dict(pickup_retention=0.9)) == "present" and verdict("swap_elbow", dict(pickup_retention=0.3)) == "absent"
    both = dict(follow_gain_mean=0.7, follow_gain_ci95=[0.6, 0.8], pickup_retention=0.9)
    assert verdict("shift_and_swap", both) == "present" and verdict("shift_and_swap", dict(both, pickup_retention=0.3)) == "absent"
    assert verdict("shift_and_swap", dict(both, follow_gain_mean=0.3)) == "partial"     # the weaker reading decides


def test_shift_plan_admits_only_displacements_the_teacher_completes(nominal, suite):
    import dataclasses
    one = dataclasses.replace(suite, scenarios=suite.scenarios[1:2], repeats=1)
    plan = P.plan_shifts(NOMINAL, one, max_draws=4)
    entry = plan[one.scenarios[0].scenario_id]
    assert entry["placement"] is not None and abs(np.hypot(*entry["placement"]) - 0.020) < 1e-9
    assert entry["cube"] is not None and abs(np.hypot(*entry["cube"]) - 0.010) < 1e-9
    rows, traces = P.run_probe(NOMINAL, one, "teacher_informed", None, 90, P.ProbeSpec("move_square"), plan=plan)
    assert rows[0]["shift_xy_m"] == entry["placement"] and rows[0]["success"]       # move_square reuses the placement vector
    none = {one.scenarios[0].scenario_id: dict(placement=None, cube=None)}
    with pytest.raises(RuntimeError):
        P.run_probe(NOMINAL, one, "teacher", None, 90, P.ProbeSpec("shift_cube"), plan=none)
    rows, _ = P.run_probe(NOMINAL, one, "teacher", None, 90, P.ProbeSpec("drop_cube"), plan=none)     # needs no displacement
    assert len(rows) == 1


def test_combined_probe_is_the_shift_on_an_arm_whose_elbow_changed(nominal, suite):
    scenario = suite.scenarios[1]
    out = {}
    for kind in ("teacher", "teacher_informed"):
        runs = {name: run(nominal, scenario, kind, name) for name in ("control", "swap_elbow", "shift_and_swap")}
        spec, row, trace = runs["shift_and_swap"]
        assert row["event_action"] == spec.at_action and abs(np.hypot(*row["shift_xy_m"]) - 0.020) < 1e-9
        stack = lambda item: {k: v[None] for k, v in item[2].items()}
        with pytest.raises(ValueError):
            P.paired_metrics(spec, [runs["control"][1]], stack(runs["control"]), [row], stack(runs["shift_and_swap"]))
        out[kind] = P.paired_metrics(spec, [runs["control"][1]], stack(runs["control"]), [row], stack(runs["shift_and_swap"]),
                                     reference=([runs["swap_elbow"][1]], stack(runs["swap_elbow"])))
        assert out[kind]["gain_reference"] == "swap_elbow" and out[kind]["pickup_retention"] is not None
    assert abs(out["teacher"]["follow_gain_mean"]) < 0.2 and 0.8 < out["teacher_informed"]["follow_gain_mean"] < 1.2
    adapter = nominal[0]                          # the swapped elbow does not leak into the next episode
    adapter.reset(scenario)
    assert np.allclose(adapter.model.actuator_gainprm[adapter._actuator_ids[P.ELBOW], 0], P.STIFF_ELBOW[0])
