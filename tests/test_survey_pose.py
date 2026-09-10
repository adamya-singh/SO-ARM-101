"""Projection round trip, cube visibility from the reset and survey poses, and the survey pose in the config."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.contracts.placement import PlacementRegime
from so_arm101_v2.physical.placement import bench_to_observation, camera_pose_at_qpos, cube_visible_at, observation_to_bench

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
SURVEY_DELTA = (0.0, 0.40, -0.60, 0.40, 0.0, 0.0)


def _survey_qpos(bench):
    return tuple(float(a + b) for a, b in zip(bench.reset_qpos, SURVEY_DELTA))


def test_projection_round_trips_across_the_frame():
    pytest.importorskip("mujoco")
    bench = scene_bench_config(LIVE_SCENE)
    cam_pos, cam_mat = camera_pose_at_qpos(LIVE_SCENE, bench.reset_qpos)
    for u in (10, 60, 127.5, 200, 245):
        for v in (10, 90, 127.5, 180, 245):
            point = observation_to_bench(bench.lens_model, cam_pos, cam_mat, u, v, 0.02)
            back = bench_to_observation(bench.lens_model, cam_pos, cam_mat, point)
            assert back is not None and abs(back[0] - u) < 0.05 and abs(back[1] - v) < 0.05
    assert bench_to_observation(bench.lens_model, cam_pos, cam_mat, cam_pos + cam_mat @ np.array([0, 0, 0.1])) is None   # behind the camera


def test_survey_pose_sees_what_the_reset_pose_cannot():
    pytest.importorskip("mujoco")
    bench = scene_bench_config(LIVE_SCENE)
    regime = PlacementRegime()
    survey = _survey_qpos(bench)
    bench.validate_qpos(survey)
    nominal = bench.cube_center
    assert cube_visible_at(LIVE_SCENE, bench, bench.reset_qpos, nominal)["visible"]
    assert cube_visible_at(LIVE_SCENE, bench, survey, nominal)["visible"]
    # Near-lateral and mid-far points of the reachable region: invisible from the reset pose, visible from the survey pose.
    for xy in ((0.15, 0.13), (-0.15, 0.13), (-0.17, 0.18), (0.13, 0.14)):   # all teacher-reachable (reach scan 2026-09-10)
        cube = bench.cube_center_at(xy)
        assert not cube_visible_at(LIVE_SCENE, bench, bench.reset_qpos, cube, margin_px=regime.survey_visibility_margin_px)["visible"], xy
        assert cube_visible_at(LIVE_SCENE, bench, survey, cube, yaw_rad=np.deg2rad(45), margin_px=regime.survey_visibility_margin_px)["visible"], xy
    # The far strip beyond the arm's reach is outside both views' interest; a corner at 12 in is not visible from the reset pose.
    assert not cube_visible_at(LIVE_SCENE, bench, bench.reset_qpos, bench.cube_center_at((0.1778, 0.3694)))["visible"]


def test_config_accepts_the_survey_pose_as_viewing_pose():
    bench = scene_bench_config(LIVE_SCENE)
    survey = _survey_qpos(bench)
    with_survey = replace(bench, viewing_qpos=survey)
    assert np.max(np.abs(np.asarray(with_survey.viewing_qpos) - np.asarray(with_survey.reset_qpos))) > 0.3
    with pytest.raises(ValueError):
        replace(bench, viewing_qpos=tuple(float(a + b) for a, b in zip(bench.reset_qpos, (0.0, -0.5, 0.0, 0.0, 0.0, 0.0))))   # below the floor
