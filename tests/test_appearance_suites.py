"""Appearance seeds through the suite helpers, capture parity and evidence, evaluation telemetry, runner parity."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.appearance import AppearanceRegime, appearance_seeds, photometric_lut, resolve_appearance
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.simulation.bench import appearance_product, bench_suite, generate_bench_suite
from so_arm101_v2.simulation.suites import load_suite_from_path

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
LENS_RUN = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909"
CHECKPOINT = LENS_RUN / "models/vision_h90/5e96018881d4140f/model.pt"


def _renderer_available() -> bool:
    try:
        import mujoco
        mujoco.Renderer(mujoco.MjModel.from_xml_path(str(LIVE_SCENE)), height=256, width=256).close()
        return True
    except Exception:
        return False


def _regime_scene(tmp_path) -> Path:
    pytest.importorskip("mujoco")
    from tools.prepare_bench_scene import prepare
    config = replace(scene_bench_config(LIVE_SCENE), appearance=AppearanceRegime().identity())
    return prepare(tmp_path / "scene", config)


def test_bench_suite_and_product_carry_seeds_and_change_ids():
    live = scene_bench_config(LIVE_SCENE)
    config = replace(live, appearance=AppearanceRegime().identity())
    offsets = [(0, 0), (0.004, -0.003)]
    fixed = bench_suite(config, offsets, label="t", repeats=2)
    seeded = bench_suite(config, offsets, label="t", repeats=2, appearance_seeds=[5, 6])
    assert fixed.suite_id != seeded.suite_id
    assert [s.appearance_seed for s in seeded.scenarios] == [5, 6] and all(not s.fixed_appearance for s in seeded.scenarios)
    assert [s.scenario_id for s in seeded.scenarios] == ["nominal", "pose_001"]
    assert [s.cube_position_m for s in seeded.scenarios] == [s.cube_position_m for s in fixed.scenarios]
    with pytest.raises(ValueError, match="one appearance seed per offset"):
        bench_suite(config, offsets, label="t", repeats=1, appearance_seeds=[5])
    with pytest.raises(ValueError, match="regime"):
        bench_suite(live, offsets, label="t", repeats=1, appearance_seeds=[5, 6])
    product = appearance_product(config, fixed, per_scenario=3, seed=8, label="heldout_appearance")
    assert len(product.scenarios) == 6 and product.repeats == 1
    assert [s.scenario_id for s in product.scenarios] == ["nominal_a0", "nominal_a1", "nominal_a2", "pose_001_a0", "pose_001_a1", "pose_001_a2"]
    assert len({s.appearance_seed for s in product.scenarios}) == 6
    assert tuple(s.appearance_seed for s in product.scenarios) == appearance_seeds(8, 6, stream=1)
    assert not set(appearance_seeds(8, 6, stream=1)) & set(appearance_seeds(8, 400, stream=0))
    assert product.scenarios[3].cube_position_m == fixed.scenarios[1].cube_position_m
    assert appearance_product(config, fixed, per_scenario=3, seed=8, label="x").suite_id != appearance_product(config, fixed, per_scenario=3, seed=9, label="x").suite_id
    with pytest.raises(ValueError, match="regime"):
        appearance_product(live, fixed, per_scenario=1, seed=1, label="x")


@pytest.mark.skipif(not _renderer_available(), reason="offscreen renderer unavailable")
def test_generated_suite_keeps_the_pose_stream_and_round_trips_seeds(tmp_path):
    scene = _regime_scene(tmp_path)
    config = scene_bench_config(scene)
    fixed, fixed_path = generate_bench_suite(scene, config, seed=3, count=2, repeats=1, output_dir=tmp_path / "fixed")
    seeded, seeded_path = generate_bench_suite(scene, config, seed=3, count=2, repeats=1, output_dir=tmp_path / "seeded", randomize_appearance=True)
    assert [s.cube_position_m for s in seeded.scenarios] == [s.cube_position_m for s in fixed.scenarios]
    assert tuple(s.appearance_seed for s in seeded.scenarios) == appearance_seeds(3, 2, stream=0)
    assert seeded.suite_id.startswith("bench_pick_replace_v1_seed3_n2_appearance_") and fixed.suite_id.startswith("bench_pick_replace_v1_seed3_n2_")
    payload = json.loads(Path(seeded_path).read_text())
    assert payload["generator"]["appearance"]["resolver"] == "bench_appearance_v1" and payload["generator"]["appearance"]["seeds"] == list(appearance_seeds(3, 2, stream=0))
    assert "appearance" not in json.loads(Path(fixed_path).read_text())["generator"]
    assert load_suite_from_path(seeded_path) == seeded and load_suite_from_path(fixed_path) == fixed
    with pytest.raises(ValueError, match="regime"):
        generate_bench_suite(LIVE_SCENE, scene_bench_config(LIVE_SCENE), seed=3, count=1, repeats=1, output_dir=tmp_path / "bad", randomize_appearance=True)


def _json(value):
    return json.loads(json.dumps(value))


def _amplified_tolerance(suite) -> int:
    regime = AppearanceRegime()
    return max(max(1, int(np.diff(photometric_lut(resolve_appearance(regime, s.appearance_seed).photometric).astype(int), axis=1).max()))
               for s in suite.scenarios)


@pytest.mark.skipif(not _renderer_available(), reason="offscreen renderer unavailable")
def test_capture_with_appearance_is_parallel_safe_and_recorded(tmp_path):
    from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations
    from so_arm101_v2.simulation.rollout import run_simulation_preflight
    scene = _regime_scene(tmp_path)
    config = scene_bench_config(scene)
    suite = bench_suite(config, [(0, 0), (0.003, 0.002)], label="capture", repeats=1, appearance_seeds=[41, 42])
    preflight = Path(run_simulation_preflight(scene, tmp_path / "sim", suite=suite, record_video=False, workers=1).report_json)
    sequential = capture_oracle_demonstrations(scene, suite, preflight, tmp_path / "seq", scenario="all", record_video=False,
                                               teacher_horizon=450, store_frames=True, workers=1)
    parallel = capture_oracle_demonstrations(scene, suite, preflight, tmp_path / "par", scenario="all", record_video=False,
                                             teacher_horizon=450, store_frames=True, workers=2)
    seq_manifest = json.loads(Path(sequential.manifest).read_text())
    par_manifest = json.loads(Path(parallel.manifest).read_text())
    assert (Path(sequential.directory) / "demonstrations.npz").read_bytes() == (Path(parallel.directory) / "demonstrations.npz").read_bytes()
    skip = ("collection_digest", "content_sha256", "frames")
    assert {k: v for k, v in seq_manifest.items() if k not in skip} == {k: v for k, v in par_manifest.items() if k not in skip}
    assert seq_manifest["appearance"] == _json(dict(regime=AppearanceRegime().identity(), resolver="bench_appearance_v1"))  # identity keys are top-level
    assert seq_manifest["frames"]["convention"] == "raw_wrist_hwc_uint8_lens_opencv_rational_8_full_frame_squash_appearance_bench_appearance_v1_preprocess_with_preprocess_wrist_image"
    for episode, scenario in zip(seq_manifest["episodes"], suite.scenarios):
        assert episode["appearance"]["seed"] == scenario.appearance_seed
        assert episode["appearance"]["params"] == _json(resolve_appearance(AppearanceRegime(), scenario.appearance_seed).as_record())
        assert episode["final_evaluation"]["success"]
    fa = np.load(Path(sequential.directory) / "images.npy", mmap_mode="r")
    fb = np.load(Path(parallel.directory) / "images.npy", mmap_mode="r")
    delta = np.abs(fa[:].astype(np.int16) - fb[:].astype(np.int16))
    assert int(delta.max()) <= _amplified_tolerance(suite) and float((delta > 0).mean()) < 0.05  # blur spreads raster jitter to neighbours
    # The two scenarios look different from each other (different draws), and neither is the pristine look.
    assert np.abs(fa[0].astype(int) - fa[450].astype(int)).mean() > 3


@pytest.mark.skipif(not (CHECKPOINT.exists() and _renderer_available()), reason="lens-run checkpoint or EGL renderer unavailable")
def test_runner_reproduces_the_evaluation_under_a_random_appearance(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from so_arm101_v2.contracts import load_pick_place_contract
    from so_arm101_v2.physical.runner import run_episode
    from so_arm101_v2.physical.sim_backend import SimBackend
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.policy_specs import PolicySpec
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    torch.set_num_threads(1)
    scene = _regime_scene(tmp_path)
    config = scene_bench_config(scene)
    suite = bench_suite(config, [(0, 0)], label="runner_parity_appearance", repeats=1, appearance_seeds=[7])
    recorded: dict[int, np.ndarray] = {}
    original = MujocoTaskAdapter.render_wrist_observation

    def recording(self):
        image = original(self)
        recorded[int(self.control_actions)] = np.array(image, copy=True)
        return image

    monkeypatch.setattr(MujocoTaskAdapter, "render_wrist_observation", recording)
    spec = PolicySpec(kind="vision_chunked", checkpoint=str(CHECKPOINT), options=(("clamp_channels", (5,)), ("black_image", False)))
    result = evaluate_closed_loop(scene, suite, {"vision": spec}, tmp_path / "reference", environment_proven=True, record_video=False, workers=1)
    monkeypatch.setattr(MujocoTaskAdapter, "render_wrist_observation", original)
    report = json.loads(Path(result.report_json).read_text())
    rollout = report["rollouts"][0]
    telemetry = json.loads(Path(rollout["telemetry_path"]).read_text())
    assert telemetry["appearance"]["seed"] == 7 and telemetry["appearance"]["resolver"] == "bench_appearance_v1"
    assert report["suite"]["scenarios"][0]["appearance_seed"] == 7
    reference_rows = telemetry["rows"]
    adapter = MujocoTaskAdapter(scene)
    contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
    backend = SimBackend(adapter, suite.scenarios[0], contract=contract, frame_source=lambda step: recorded[step])
    policy = VisionChunkedPolicy(CHECKPOINT, black_image=False, clamp_channels=(5,))
    try:
        episode = run_episode(backend, policy, bench=adapter.bench, max_actions=contract.max_actions)
    finally:
        backend.close()
    assert adapter.appearance_record["seed"] == 7
    assert episode.success == rollout["success"] and episode.actions == len(reference_rows)
    for ours, theirs in zip(backend.rows, reference_rows):
        assert ours["current_act"] == theirs["current_act"]
        assert ours["requested_act"] == theirs["requested_act"]
        assert ours["executed_act"] == theirs["executed_act"]
        assert ours["robot_qpos"] == theirs["robot_qpos"]
