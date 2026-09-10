"""Placement regime, per-scenario square placement in the adapter, and identity neutrality for fixed suites."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import BenchConfig, scene_bench_config
from so_arm101_v2.contracts.placement import INCH, PlacementRegime, fold_cube_yaw, placement_draws, yaw_quaternion
from so_arm101_v2.simulation.suites import SimulationScenario, SimulationSuite, _suite_from_payload, scenario_record, suite_payload

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def test_regime_defaults_match_the_user_rectangle_and_validate():
    regime = PlacementRegime()
    assert regime.x_range_m == pytest.approx((-7 * INCH, 7 * INCH))
    assert regime.y_range_m[0] == pytest.approx(0.0646353 + 2 * INCH) and regime.depth_m == pytest.approx(10 * INCH)
    assert regime.contains((0.0, 0.2805353)) and not regime.contains((0.0, 0.40)) and not regime.contains((0.2, 0.2))
    assert regime.distance_to_edge_mm((0.0, 0.2805353)) > 0 and regime.distance_to_edge_mm((0.0, 0.40)) < 0
    regime.validate_against(0.0646353, (0.0, 0.2805353))
    with pytest.raises(ValueError, match="near edge"):
        regime.validate_against(0.10, (0.0, 0.2805353))
    with pytest.raises(ValueError, match="nominal"):
        regime.validate_against(0.0646353, (0.0, 0.5))
    with pytest.raises(ValueError):
        PlacementRegime(y_range_m=(0.1154, 0.30))     # depth mismatch
    with pytest.raises(ValueError):
        PlacementRegime(yaw_range_deg=120)
    assert PlacementRegime.from_mapping(json.loads(json.dumps(regime.identity()))) == regime
    draws = placement_draws(12, 500, regime)
    assert draws == placement_draws(12, 500, regime) and draws != placement_draws(13, 500, regime)
    assert all(regime.contains(d[:2]) and abs(d[2]) <= np.deg2rad(45) for d in draws)
    assert len({round(d[0], 6) for d in draws}) > 400 and placement_draws(12, 5, regime, stream=3) != draws[:5]
    q = yaw_quaternion(0.3)
    assert q == pytest.approx((np.cos(0.15), 0, 0, np.sin(0.15)))
    assert fold_cube_yaw(np.deg2rad(50)) == pytest.approx(np.deg2rad(-40)) and fold_cube_yaw(np.deg2rad(-100)) == pytest.approx(np.deg2rad(-10))


def test_bench_config_carries_the_regime_and_the_geometry_guard_keeps_the_size():
    live = scene_bench_config(LIVE_SCENE)
    with_regime = replace(live, placement=PlacementRegime().identity())
    assert with_regime.placement_regime == PlacementRegime() and with_regime.square_center_xy == (0.0, 0.2805353)
    assert with_regime.cube_center_at((0.1, 0.2)) == (0.1, 0.2, 0.011)
    with pytest.raises(ValueError, match="geometry"):
        replace(live, cube_edge_m=0.025)
    with pytest.raises(ValueError, match="nominal"):
        replace(live, placement=PlacementRegime().identity(), square_center_xy=(0.0, 0.45))
    reloaded = BenchConfig(**json.loads(json.dumps({**live.__dict__, "placement": PlacementRegime().identity()}, default=list)))
    assert reloaded.placement_regime == PlacementRegime()


def test_scenario_placement_fields_round_trip_and_fixed_suites_hash_as_before():
    base = SimulationScenario("s", (0.0, 0.28, 0.011), (0.0,) * 6, (1.0, 0.0, 0.0, 0.0))
    assert "square_center_xy" not in scenario_record(base) and "square_yaw_rad" not in scenario_record(base)
    placed = replace(base, scenario_id="p", square_center_xy=(0.05, 0.2), square_yaw_rad=0.4)
    record = scenario_record(placed)
    assert record["square_center_xy"] == (0.05, 0.2) and record["square_yaw_rad"] == 0.4
    with pytest.raises(ValueError):
        replace(base, square_center_xy=(0.05,))
    with pytest.raises(ValueError):
        replace(base, square_yaw_rad=float("nan"))
    suite = SimulationSuite("suite", 1, "bench_pick_replace_v1", 1, False, (base, placed))
    again = _suite_from_payload(json.loads(json.dumps(suite_payload(suite))))
    assert again.scenarios == suite.scenarios
    legacy = json.loads(json.dumps(suite_payload(suite)))["scenarios"][0]
    assert "square_center_xy" not in legacy and _suite_from_payload({**suite_payload(suite), "scenarios": [legacy]}).scenarios[0] == base


def _renderer_available() -> bool:
    try:
        import mujoco
        mujoco.Renderer(mujoco.MjModel.from_xml_path(str(LIVE_SCENE)), height=256, width=256).close()
        return True
    except Exception:
        return False


def test_adapter_moves_square_towel_and_cube_together_and_restores(tmp_path):
    pytest.importorskip("mujoco")
    from tools.prepare_bench_scene import prepare
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    config = replace(scene_bench_config(LIVE_SCENE), placement=PlacementRegime().identity())
    scene = prepare(tmp_path / "scene", config)
    adapter = MujocoTaskAdapter(scene)
    try:
        bench = adapter.bench
        nominal = bench_suite(bench, [(0, 0)], label="t", repeats=1).scenarios[0]
        adapter.reset(nominal)
        assert adapter.placement_record is None
        napkin, towel = adapter.model.geom("napkin").id, adapter.model.body("towel_visual").id
        nominal_pos = adapter.data.geom_xpos[napkin].copy()
        # A moved and yawed square: napkin, towel and cube go together; the footprint metric follows.
        placed = bench_suite(bench, [(0.004, -0.003)], label="t", repeats=1, square_centers=[(0.12, 0.20)], square_yaws=[0.5]).scenarios[0]
        assert placed.square_center_xy == (0.12, 0.20) and placed.square_yaw_rad == 0.5 and placed.cube_quaternion_wxyz == yaw_quaternion(0.5)
        c, s = np.cos(0.5), np.sin(0.5)
        assert placed.cube_position_m[0] == pytest.approx(0.12 + c * 0.004 - s * (-0.003)) and placed.cube_position_m[1] == pytest.approx(0.20 + s * 0.004 + c * (-0.003))
        adapter.reset(placed)
        assert adapter.placement_record == dict(square_center_xy=[0.12, 0.20], square_yaw_rad=0.5, resolver="bench_placement_v1")
        assert np.allclose(adapter.data.geom_xpos[napkin][:2], (0.12, 0.20)) and np.allclose(adapter.data.xpos[towel][:2], (0.12, 0.20))
        yaw = np.arctan2(adapter.data.geom_xmat[napkin].reshape(3, 3)[1, 0], adapter.data.geom_xmat[napkin].reshape(3, 3)[0, 0])
        assert yaw == pytest.approx(0.5, abs=1e-6)
        measurement, _ = adapter.pick_place_measurement()
        assert measurement.cube_footprint_inside and abs(measurement.cube_support_error_m) < 0.001
        # The cube at the nominal spot is now outside the moved square's footprint.
        outside = replace(placed, scenario_id="o", cube_position_m=nominal.cube_position_m)
        adapter.reset(outside)
        assert not adapter.pick_place_measurement()[0].cube_footprint_inside
        # Back to the nominal scenario: square restored exactly, record None.
        adapter.reset(nominal)
        assert adapter.placement_record is None and np.allclose(adapter.data.geom_xpos[napkin], nominal_pos)
        assert np.allclose(adapter.model.body_pos[towel][:2], bench.square_center_xy) and np.allclose(adapter.model.body_quat[towel], (1, 0, 0, 0))
        assert adapter._appearance_pristine.matches(adapter.model)
    finally:
        adapter.close()


def test_placement_and_appearance_draw_coexist_and_render(tmp_path):
    pytest.importorskip("mujoco")
    if not _renderer_available():
        pytest.skip("offscreen renderer unavailable")
    from tools.prepare_bench_scene import prepare
    from so_arm101_v2.contracts.appearance import AppearanceRegime, resolve_appearance
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    config = replace(scene_bench_config(LIVE_SCENE), placement=PlacementRegime().identity(), appearance=AppearanceRegime().identity())
    scene = prepare(tmp_path / "scene", config)
    adapter = MujocoTaskAdapter(scene)
    try:
        bench = adapter.bench
        towel = adapter.model.body("towel_visual").id
        seed = next(s for s in range(40) if resolve_appearance(AppearanceRegime(), s).towel is not None)
        scenario = bench_suite(bench, [(0, 0)], label="t", repeats=1, appearance_seeds=[seed], square_centers=[(-0.08, 0.24)], square_yaws=[-0.3]).scenarios[0]
        adapter.reset(scenario)
        draw_yaw = resolve_appearance(AppearanceRegime(), seed).towel[2]
        quat = adapter.model.body_quat[towel]
        assert 2 * np.arctan2(quat[3], quat[0]) == pytest.approx(-0.3 + draw_yaw, abs=1e-6)   # placement yaw composed on the draw's yaw
        image = adapter.render_wrist_observation()
        assert image.shape == (256, 256, 3)
        adapter.reset(bench_suite(bench, [(0, 0)], label="t", repeats=1).scenarios[0])
        assert adapter._appearance_pristine.matches(adapter.model)
    finally:
        adapter.close()


def test_scenario_without_regime_cannot_move_the_square():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    adapter = MujocoTaskAdapter(LIVE_SCENE)
    try:
        if adapter.bench.placement is not None:
            pytest.skip("live scene already carries a placement regime")
        nominal = bench_suite(adapter.bench, [(0, 0)], label="t", repeats=1).scenarios[0]
        with pytest.raises(ValueError, match="placement"):
            bench_suite(adapter.bench, [(0, 0)], label="t", repeats=1, square_centers=[(0.1, 0.2)])
        with pytest.raises(ValueError, match="placement regime"):
            adapter.reset(replace(nominal, square_center_xy=(0.1, 0.2)))
    finally:
        adapter.close()
