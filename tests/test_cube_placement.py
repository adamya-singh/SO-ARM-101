"""Cube placement from the wrist observation: exact on the simulated reset frame, ~40 mm off on the 2026-09-10 real frame."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.physical.placement import camera_pose_at, check_cube_placement, locate_cube, observation_to_bench

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
EPISODE_04 = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/physical/episode_04_20260910"
SIM_REF = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/sim_reset_observation_fcead5c7.png"


def _point_bench():
    """The pre-placement configuration (single task pose): point-mode placement checks."""
    from dataclasses import replace
    return replace(scene_bench_config(SCENE), placement=None)


def _renderer_available() -> bool:
    try:
        import mujoco
        mujoco.Renderer(mujoco.MjModel.from_xml_path(str(SCENE)), height=256, width=256).close()
        return True
    except Exception:
        return False


def test_simulated_reset_frame_locates_the_nominal_cube():
    pytest.importorskip("mujoco")
    if not _renderer_available():
        pytest.skip("offscreen renderer unavailable")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    adapter = MujocoTaskAdapter(SCENE)
    try:
        nominal = bench_suite(adapter.bench, [(0, 0)], label="t", repeats=1).scenarios[0]
        adapter.reset(nominal)
        image = adapter.render_wrist_observation()
        current = adapter.current_act()
        bench = _point_bench()
    finally:
        adapter.close()
    result = check_cube_placement(image, bench, SCENE, current, tolerance_mm=15.0)
    assert result["found"] and result["ok"], result
    assert result["distance_mm"] <= 12.0                      # the top face's dark blob is biased a little toward the camera
    # The square's near edge back-projects to its true y (0.2551 m) within 1 mm.
    cam_pos, cam_mat = camera_pose_at(SCENE, bench, current)
    g = image.astype(float).mean(-1)
    ys, xs = np.nonzero(g > 150)
    near = observation_to_bench(bench.lens_model, cam_pos, cam_mat, (xs.min() + xs.max()) / 2, ys.max(), 0.0)
    assert abs(near[1] - (bench.square_center_xy[1] - bench.square_edge_m / 2)) < 0.001
    # Offset poses are recovered too.
    adapter = MujocoTaskAdapter(SCENE)
    try:
        shifted = bench_suite(adapter.bench, [(0.008, -0.009)], label="t", repeats=1).scenarios[0]
        adapter.reset(shifted)
        image = adapter.render_wrist_observation()
        current = adapter.current_act()
    finally:
        adapter.close()
    result = check_cube_placement(image, bench, SCENE, current, tolerance_mm=15.0)
    assert result["found"] and abs(result["offset_mm"]["dx"] - 8) < 4 and abs(result["offset_mm"]["dy"] - (-9)) < 5, result


@pytest.mark.skipif(not EPISODE_04.exists(), reason="episode_04 evidence unavailable")
def test_real_reset_frame_of_episode_04_shows_the_cube_40mm_beyond_the_task_pose():
    pytest.importorskip("mujoco")
    from so_arm101_v2.physical.dry_pass import load_boundary_frame
    bench = _point_bench()
    image, anchor, _ = load_boundary_frame(EPISODE_04, step=0)
    result = check_cube_placement(image, bench, SCENE, anchor, tolerance_mm=15.0)
    assert result["found"] and not result["ok"]
    assert result["offset_mm"]["dy"] > 30 and result["clipped_at_frame_edge"]
    assert "toward the base" in result["advice"]


def test_no_towel_is_reported_not_guessed():
    bench = scene_bench_config(SCENE)
    black = np.zeros((256, 256, 3), np.uint8)
    result = locate_cube(black, bench.lens_model, np.array([0, 0.19, 0.19]), np.eye(3))
    assert not result["found"] and "towel" in result["reason"]


def test_placement_regime_makes_the_check_region_aware():
    from dataclasses import replace
    from so_arm101_v2.contracts.placement import PlacementRegime
    bench = replace(scene_bench_config(SCENE), placement=PlacementRegime().identity())
    anchor = np.asarray([0.0017, -2.8565, 2.8529, 1.3711, -0.0345, 0.2324], dtype=np.float32)
    black = np.zeros((256, 256, 3), np.uint8)
    unseen = check_cube_placement(black, bench, SCENE, anchor)
    assert unseen["mode"] == "rectangle" and unseen["ok"] is True and unseen["inside_rectangle"] is None
    if EPISODE_04.exists():
        from so_arm101_v2.physical.dry_pass import load_boundary_frame
        image, act, _ = load_boundary_frame(EPISODE_04, step=0)
        seen = check_cube_placement(image, bench, SCENE, act)
        assert seen["found"] and seen["inside_rectangle"] is True and seen["ok"] and "inside the placement rectangle" in seen["advice"]
        assert seen["distance_to_edge_mm"] > 0
