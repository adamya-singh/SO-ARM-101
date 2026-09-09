"""Physical lens model: inversion, validity domain, and the sim/real observation operators."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from so_arm101_v2.contracts.lens import LensModel, Resampler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INTRINSICS = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/camera_intrinsics.json"

# A strongly barrel-distorted 5-coefficient lens (the 2026-09-08 centre fit) with a small render
# so the tests stay fast. It is NOT invertible out to the frame corners, so tests that need a
# full-frame operator use the calibrated 2026-09-09 model (`_calibrated`) or `IDENTITY`.
CENTRE = LensModel(model="opencv_pinhole_5", image_size=(1920, 1080), fx=1334.651, fy=1336.99, cx=960.0, cy=540.0,
                   dist=(-0.567035, 0.46358, -0.004185, -0.010255, -0.205684), render_fovy_deg=60.0, render_size=(320, 180))
IDENTITY = LensModel(model="opencv_pinhole_5", image_size=(1920, 1080), fx=1335.0, fy=1335.0, cx=959.5, cy=539.5,
                     dist=(0.0, 0.0, 0.0, 0.0, 0.0), render_fovy_deg=2 * np.degrees(np.arctan(540 / 1335.0)), render_size=(1920, 1080))


def _calibrated():
    if not INTRINSICS.exists():
        pytest.skip("calibrated intrinsics not present")
    probe = LensModel.from_intrinsics_file(INTRINSICS, render_fovy_deg=90.0, render_size=(1600, 900))
    return LensModel.from_intrinsics_file(INTRINSICS, render_fovy_deg=round(probe.required_render_fovy_deg(), 2), render_size=(1600, 900))


def test_distort_undistort_round_trip_inside_valid_radius():
    rng = np.random.default_rng(0)
    limit = CENTRE.valid_radius
    r = rng.uniform(0, 0.9 * limit, 2000)
    theta = rng.uniform(0, 2 * np.pi, 2000)
    xd, yd = r * np.cos(theta), r * np.sin(theta)
    x, y = CENTRE.undistort(xd, yd)
    bx, by = CENTRE.distort(x, y)
    assert np.max(np.hypot(bx - xd, by - yd)) < 1e-12


def test_undistort_refuses_points_beyond_the_valid_radius():
    limit = CENTRE.valid_radius
    assert 0.65 < limit < 0.75  # the 2026-09-08 centre model peaks at 929 px / 1335 px per unit
    with pytest.raises(ValueError, match="invertible radius"):
        CENTRE.undistort(np.array([limit * 1.01]), np.array([0.0]))


def test_valid_radius_matches_closed_form_peak():
    r = np.linspace(0, 1.2, 200001)
    k1, k2, _p1, _p2, k3 = CENTRE.dist
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    assert abs(CENTRE.valid_radius - rd.max()) < 1e-4


def test_barrel_lens_pulls_edges_inward_by_the_closed_form_amount():
    # A raw-frame point at the side edge maps to a larger undistorted radius (barrel); with the
    # tangential terms zeroed the ratio equals the radial factor at the undistorted radius.
    radial_only = replace(CENTRE, dist=(CENTRE.dist[0], CENTRE.dist[1], 0.0, 0.0, CENTRE.dist[4]))
    xd, yd = radial_only.raw_to_normalised(np.array([1700.0]), np.array([540.0]))
    x, y = radial_only.undistort(xd, yd)
    k1, k2, _p1, _p2, k3 = radial_only.dist
    r2 = x * x + y * y
    factor = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    assert x[0] > xd[0] > 0
    assert abs(xd[0] / x[0] - factor[0]) < 1e-9


def test_sim_operator_coverage_guard_rejects_a_too_narrow_render():
    lens = _calibrated()
    with pytest.raises(ValueError, match="outside the source image"):
        replace(lens, render_fovy_deg=30.0).sim_operator()


def test_sim_operator_refuses_a_lens_that_cannot_describe_the_frame_corners():
    # The 2026-09-08 centre-only model is not invertible out to the frame corners, so no render
    # can make a full-frame observation from it.
    with pytest.raises(ValueError, match="invertible radius"):
        replace(CENTRE, render_fovy_deg=120.0).sim_operator()


def test_sim_operator_preserves_a_constant_image_and_is_bitwise_repeatable():
    lens = _calibrated()
    op = lens.sim_operator()
    rh, rw = op.source_shape
    constant = np.full((rh, rw, 3), 137, dtype=np.uint8)
    assert np.unique(op.apply(constant)).tolist() == [137]
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (rh, rw, 3), dtype=np.uint8)
    a, b = op.apply(image), op.apply(image)
    assert a.shape == (256, 256, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    weights_per_row = np.add.reduceat(op.weight.astype(np.float64), op.row_ptr[:-1])
    assert np.allclose(weights_per_row, 1.0, atol=1e-5)


def test_identity_lens_sim_operator_matches_the_exact_box_filter():
    # With zero distortion and a render identical to the raw frame, the supersampled sim path must
    # agree with the exact separable area filter to within one grey level on a smooth image.
    yy, xx = np.mgrid[0:1080, 0:1920]
    smooth = np.stack([(xx / 1920 * 255), (yy / 1080 * 255), ((xx + yy) / 3000 * 255)], axis=-1).astype(np.uint8)
    via_sim = IDENTITY.sim_operator().apply(smooth)
    via_real = IDENTITY.real_operator().apply(smooth)
    assert np.abs(via_sim.astype(int) - via_real.astype(int)).max() <= 1


def test_real_operator_is_the_exact_area_filter():
    lens = IDENTITY
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    out = lens.real_operator().apply(image)
    # Observation pixel (0, 0) averages raw x in [0, 7.5) and y in [0, 4.21875) exactly.
    sub = image[:5, :8, 0].astype(np.float64)
    wy = np.array([1, 1, 1, 1, 0.21875]); wx = np.array([1, 1, 1, 1, 1, 1, 1, 0.5])
    expected = (wy[:, None] * wx[None, :] * sub).sum() / (4.21875 * 7.5)
    assert out[0, 0, 0] == int(np.clip(np.rint(expected), 0, 255))
    assert out.shape == (256, 256, 3)


def test_render_density_exceeds_observation_density_everywhere():
    lens = _calibrated()
    op = lens.sim_operator()
    # Every observation pixel must draw on more than one distinct render pixel; a starved row means
    # the render is coarser than the observation somewhere.
    counts = np.diff(op.row_ptr)
    assert counts.min() >= 4
    assert lens.render_focal_px > lens.fy * 256 / 1080 * 1.2


def test_lens_model_round_trips_through_its_identity_mapping():
    lens = _calibrated()
    again = LensModel.from_mapping(json.loads(json.dumps(lens.identity())))
    assert again == lens
    assert lens.intrinsics_sha256 is not None and lens.calibration_rms_px is not None


def test_lens_model_validation():
    with pytest.raises(ValueError, match="distortion coefficients"):
        replace(CENTRE, dist=(0.1, 0.2))
    with pytest.raises(ValueError, match="unsupported lens model"):
        replace(CENTRE, model="fisheye")
    with pytest.raises(ValueError, match="principal point"):
        replace(CENTRE, cx=5000.0)
    with pytest.raises(ValueError, match="fixed at 256"):
        replace(CENTRE, observation_size=128)


def test_bench_config_with_lens_requires_matching_camera_fovy_and_hashes_the_block(tmp_path):
    from so_arm101_v2.contracts.bench import BenchConfig
    lens = _calibrated()
    config = BenchConfig(reset_physical=(0, -90, 90, 43, -1, 13), camera_pos=(0.0, 0.0, 0.07), camera_quat_wxyz=(1, 0, 0, 0),
                         camera_fovy_deg=round(lens.fovy_deg, 2), lens=lens.identity())
    assert config.lens_model == lens
    with pytest.raises(ValueError, match="vertical field of view"):
        replace(config, camera_fovy_deg=43.99)
    with pytest.raises(ValueError, match="render fovy"):
        replace(config, lens=replace(lens, render_fovy_deg=30.0).identity())
    with pytest.raises(ValueError, match="16:9"):
        replace(config, lens=replace(lens, render_size=(900, 900)).identity())
    # The block round-trips through the JSON the scene generator writes and changes the scene hash.
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.bench import scene_dependency_hash
    from tools.prepare_bench_scene import prepare
    plain = prepare(tmp_path / "plain", replace(config, lens=None, camera_fovy_deg=43.99))
    with_lens = prepare(tmp_path / "lens", config)
    assert BenchConfig.load(tmp_path / "lens" / "bench_config.json").lens_model == lens
    assert scene_dependency_hash(plain) != scene_dependency_hash(with_lens)
    xml = (tmp_path / "lens" / "scene_bench_pick_replace_v1.xml").read_text()
    assert f'fovy="{lens.render_fovy_deg:.2f}"' in xml and 'offwidth="1600"' in xml and 'offheight="900"' in xml
