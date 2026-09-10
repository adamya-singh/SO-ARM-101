"""Placement-randomized suite generation: deterministic draws, screening with rejections recorded, certification set."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.contracts.placement import PlacementRegime
from so_arm101_v2.physical.placement import cube_visible_at
from so_arm101_v2.simulation.bench import CERTIFICATION_PLACEMENTS, bench_suite, certification_suite, generate_bench_suite, screen_scenario
from so_arm101_v2.simulation.suites import load_suite_from_path

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
SURVEY_DELTA = (0.0, 0.40, -0.60, 0.40, 0.0, 0.0)


def _placement_scene(tmp_path) -> Path:
    pytest.importorskip("mujoco")
    from tools.prepare_bench_scene import prepare
    live = scene_bench_config(LIVE_SCENE)
    survey = tuple(float(a + b) for a, b in zip(live.reset_qpos, SURVEY_DELTA))
    return prepare(tmp_path / "scene", replace(live, placement=PlacementRegime().identity(), viewing_qpos=survey))


def test_certification_placements_are_inside_the_rectangle_and_visible_from_the_survey_pose(tmp_path):
    scene = _placement_scene(tmp_path)
    bench = scene_bench_config(scene)
    regime = bench.placement_regime
    suite = certification_suite(bench)
    assert len(suite.scenarios) == len(CERTIFICATION_PLACEMENTS) == 10 and suite.repeats == 3
    for scenario, (x, y, yaw) in zip(suite.scenarios, CERTIFICATION_PLACEMENTS):
        assert regime.contains((x, y)) and scenario.square_center_xy == (x, y) and scenario.square_yaw_rad == pytest.approx(np.deg2rad(yaw))
        assert cube_visible_at(scene, bench, bench.viewing_qpos, scenario.cube_position_m, yaw_rad=scenario.square_yaw_rad,
                               margin_px=regime.survey_visibility_margin_px)["visible"], (x, y, yaw)
    # A config without a regime keeps the legacy five-offset certification.
    legacy = certification_suite(scene_bench_config(LIVE_SCENE) if scene_bench_config(LIVE_SCENE).placement is None else replace(bench, placement=None))
    assert len(legacy.scenarios) == 5 and legacy.scenarios[0].scenario_id == "nominal"


def test_visibility_screen_rejects_placements_the_survey_pose_cannot_see(tmp_path):
    scene = _placement_scene(tmp_path)
    bench = scene_bench_config(scene)
    far = bench_suite(bench, [(0, 0)], label="t", repeats=1, square_centers=[(0.1778, 0.3694)], square_yaws=[0.0]).scenarios[0]
    assert screen_scenario(scene, far) == (False, "not_visible_from_survey")


def test_generated_placement_suite_is_deterministic_and_records_rejections(tmp_path):
    scene = _placement_scene(tmp_path)
    bench = scene_bench_config(scene)
    first, path = generate_bench_suite(scene, bench, seed=5, count=2, repeats=1, output_dir=tmp_path / "a", randomize_placement=True)
    second, _ = generate_bench_suite(scene, bench, seed=5, count=2, repeats=1, output_dir=tmp_path / "b", randomize_placement=True)
    assert first.suite_id == second.suite_id and first.scenarios == second.scenarios
    assert first.suite_id.startswith("bench_pick_replace_v1_seed5_n2_placement_")
    assert all(s.square_center_xy is not None and bench.placement_regime.contains(s.square_center_xy) for s in first.scenarios)
    assert all(abs(s.square_yaw_rad) <= np.deg2rad(45) + 1e-9 for s in first.scenarios)
    payload = json.loads(Path(path).read_text())
    block = payload["generator"]["placement"]
    assert block["resolver"] == "bench_placement_v1" and block["seed_stream"] == dict(seed=5, stream=2)
    assert "accepted_bbox_m" in block and isinstance(block["rejections_by_reason"], dict)
    assert load_suite_from_path(path) == first
    with pytest.raises(ValueError, match="placement regime"):
        generate_bench_suite(scene, replace(bench, placement=None), seed=5, count=1, repeats=1, output_dir=tmp_path / "c", randomize_placement=True)
