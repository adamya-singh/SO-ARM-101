"""Process-parallel oracle capture: parity with the sequential path, video-guard neutrality, skip compaction.

Physics, labels, episodes and the manifest are byte-identical between the two paths. Rendered
frames are not reproducible even sequentially: EGL rasterization jitters by one grey level on a
handful of pixels run to run (measured 2026-09-09: ~4-8 % of 256x256 frames differ, max |delta| 1),
so the frames sidecar is compared with that tolerance and its digest is a record, not a claim.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.simulation import load_simulation_suite
from so_arm101_v2.simulation import oracle as oracle_module
from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations
from so_arm101_v2.simulation.rollout import run_simulation_preflight

SCENE = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"


def _renderer_available() -> bool:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(SCENE))
        renderer = mujoco.Renderer(model, height=256, width=256)
        renderer.close()
        return True
    except Exception:
        return False


def _narrow_suite():
    suite = load_simulation_suite("fixed_pick_place_v3")
    return replace(suite, scenarios=suite.scenarios[:2], repeats=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FRAME_KEYS = ("collection_digest", "content_sha256", "frames")


def _assert_frames_within_raster_jitter(a: Path, b: Path) -> None:
    fa = np.load(a, mmap_mode="r")
    fb = np.load(b, mmap_mode="r")
    assert fa.shape == fb.shape
    delta = np.abs(fa[:].astype(np.int16) - fb[:].astype(np.int16))
    assert int(delta.max()) <= 1
    assert float((delta > 0).mean()) < 1e-3


@pytest.fixture(scope="module")
def preflight(tmp_path_factory):
    if not _renderer_available():
        pytest.skip("offscreen renderer unavailable")
    root = tmp_path_factory.mktemp("preflight")
    result = run_simulation_preflight(SCENE, root / "sim", suite=_narrow_suite(), record_video=False, workers=1)
    return Path(result.report_json)


@pytest.mark.skipif(not _renderer_available(), reason="offscreen renderer unavailable")
def test_parallel_capture_matches_sequential_bytes(tmp_path: Path, preflight: Path) -> None:
    suite = _narrow_suite()
    sequential = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "seq", scenario="all", record_video=False,
                                               teacher_horizon=450, store_frames=True, workers=1)
    parallel = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "par", scenario="all", record_video=False,
                                             teacher_horizon=450, store_frames=True, workers=2)
    seq_manifest = json.loads(Path(sequential.manifest).read_text())
    par_manifest = json.loads(Path(parallel.manifest).read_text())
    assert (Path(sequential.directory) / "demonstrations.npz").read_bytes() == (Path(parallel.directory) / "demonstrations.npz").read_bytes()
    assert {k: v for k, v in seq_manifest.items() if k not in FRAME_KEYS} == {k: v for k, v in par_manifest.items() if k not in FRAME_KEYS}
    assert seq_manifest["frames"]["rows"] == par_manifest["frames"]["rows"] == 900
    _assert_frames_within_raster_jitter(Path(sequential.directory) / "images.npy", Path(parallel.directory) / "images.npy")


@pytest.mark.skipif(not _renderer_available(), reason="offscreen renderer unavailable")
def test_video_toggle_is_digest_neutral(tmp_path: Path, preflight: Path) -> None:
    pytest.importorskip("av")
    suite = _narrow_suite()
    first = suite.scenarios[0].scenario_id
    with_video = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "video", scenario=first, record_video=True,
                                               teacher_horizon=450, store_frames=True, workers=1)
    without = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "novideo", scenario=first, record_video=False,
                                            teacher_horizon=450, store_frames=True, workers=1)
    a = json.loads(Path(with_video.manifest).read_text())
    b = json.loads(Path(without.manifest).read_text())
    assert a["arrays"]["sha256"] == b["arrays"]["sha256"]
    assert a["episodes"] == b["episodes"]
    assert len(a["videos"]) == 2 and b["videos"] == []
    _assert_frames_within_raster_jitter(Path(with_video.directory) / "images.npy", Path(without.directory) / "images.npy")


def _forged_preflight(path: Path, suite) -> Path:
    report = {
        "environment_proven": True, "deterministic": True, "suite": {"suite_id": suite.suite_id},
        "rollouts": [{"scenario_id": s.scenario_id, "success": True} for s in suite.scenarios],
    }
    report["content_sha256"] = content_sha256(report)
    path.write_text(json.dumps(report))
    return path


def test_skip_compaction_keeps_frames_and_columns_in_scenario_order(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("mujoco")
    suite = replace(load_simulation_suite("fixed_pick_place_v3"), repeats=1)
    suite = replace(suite, scenarios=suite.scenarios[:3])
    horizon = 4

    def stub(task):
        frames = np.load(task.frames_path, mmap_mode="r+")
        base = task.scenario_index * horizon
        frames[base:base + horizon] = np.full((horizon, 256, 256, 3), task.scenario_index + 1, dtype=np.uint8)
        frames.flush()
        if task.scenario_index == 1:
            if not task.skip_failed_scenarios:
                raise RuntimeError("stubbed failure")
            return oracle_module._OracleScenarioResult(task.scenario.scenario_id, "stubbed failure", {n: [] for n in oracle_module._ORACLE_FIELDS},
                                                       [], (), [], {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}, None, ())
        columns = {name: [] for name in oracle_module._ORACLE_FIELDS}
        for action_index in range(horizon):
            for name in oracle_module._ORACLE_FIELDS:
                if name == "scenario_index":
                    columns[name].append(task.scenario_index)
                elif name == "action_index":
                    columns[name].append(action_index)
                elif name == "progress":
                    columns[name].append(action_index / (horizon - 1))
                elif name in ("current_act", "robot_qvel", "requested_act", "executed_act", "executed_delta_act"):
                    columns[name].append(np.zeros(6, dtype=np.float32))
                elif name in ("cube_position", "cube_linear_velocity", "cube_angular_velocity", "post_cube_position"):
                    columns[name].append(np.zeros(3, dtype=np.float64))
                elif name == "cube_quaternion_wxyz":
                    columns[name].append(np.array([1.0, 0, 0, 0]))
                else:
                    columns[name].append(0.0)
        events = [{"action_index": i, "pickup_events": [], "placement_events": []} for i in range(horizon)]
        return oracle_module._OracleScenarioResult(task.scenario.scenario_id, None, columns, events, (0, horizon), [],
                                                   {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}, {"success": True}, ())

    monkeypatch.setattr(oracle_module, "_capture_scenario", stub)
    preflight = _forged_preflight(tmp_path / "preflight.json", suite)
    result = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "out", scenario="all", record_video=False,
                                           teacher_horizon=horizon, store_frames=True, skip_failed_scenarios=True, workers=1)
    manifest = json.loads(Path(result.manifest).read_text())
    frames = np.load(Path(result.directory) / "images.npy", mmap_mode="r")
    assert manifest["frames"]["rows"] == 2 * horizon and frames.shape[0] == 2 * horizon
    assert np.unique(frames[:horizon]).tolist() == [1] and np.unique(frames[horizon:]).tolist() == [3]
    arrays = np.load(Path(result.directory) / "demonstrations.npz")
    assert arrays["scenario_index"].tolist() == [0] * horizon + [2] * horizon
    assert [s["scenario_id"] for s in manifest["skipped_scenarios"]] == [suite.scenarios[1].scenario_id]
    with pytest.raises(RuntimeError):
        capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "strict", scenario="all", record_video=False,
                                      teacher_horizon=horizon, store_frames=True, skip_failed_scenarios=False, workers=1)


@pytest.mark.skipif(not _renderer_available(), reason="offscreen renderer unavailable")
def test_frame_row_stride_stores_every_nth_frame(tmp_path: Path, preflight: Path) -> None:
    suite = _narrow_suite()
    full = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "full", scenario="all", record_video=False,
                                         teacher_horizon=450, store_frames=True, workers=2)
    strided = capture_oracle_demonstrations(SCENE, suite, preflight, tmp_path / "strided", scenario="all", record_video=False,
                                            teacher_horizon=450, store_frames=True, workers=2, frame_row_stride=5)
    a = json.loads(Path(full.manifest).read_text()); b = json.loads(Path(strided.manifest).read_text())
    assert a["arrays"]["sha256"] == b["arrays"]["sha256"] and a["episodes"] == b["episodes"]
    assert "row_stride" not in a["frames"] and "row_stride" not in a["frame_store"]
    assert b["frames"]["row_stride"] == 5 and b["frames"]["rows_per_episode"] == 90 and b["frames"]["rows"] == 90 * len(a["episodes"])
    assert b["frame_store"]["row_stride"] == 5
    fa = np.load(Path(full.directory) / "images.npy", mmap_mode="r"); fb = np.load(Path(strided.directory) / "images.npy", mmap_mode="r")
    for episode in range(len(a["episodes"])):
        expected = fa[episode * 450:(episode + 1) * 450:5]; got = fb[episode * 90:(episode + 1) * 90]
        delta = np.abs(expected.astype(np.int16) - got.astype(np.int16))
        assert int(delta.max()) <= 1 and float((delta > 0).mean()) < 1e-3
    from so_arm101_v2.learning.vision import load_vision_frames
    _, frames, arrays = load_vision_frames(strided.manifest)
    assert frames.shape[0] == arrays["action_index"].shape[0] and frames.stored_mask.sum() == b["frames"]["rows"]
