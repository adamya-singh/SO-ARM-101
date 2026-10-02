"""pi0.5 lane (2026-10-02): targets and state are the CNN lane's, closure matches the 09-18 analysis, the prompt fits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
LADDER = EXPERIMENTS / "scaling_ladder_20260910"
MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
CLOSURE_REF = EXPERIMENTS / "tranche_analysis_20260918/closure_error_vs_reference.json"
TELEMETRY_4800 = (EXPERIMENTS / "augmentation_shift12_4800_20260912/evaluations/shift12/policies/"
                  "bench_pick_replace_v1_seed8_n10_appearance_placement_0a75289f941d/telemetry")


def _heldout_manifest() -> Path:
    pointer = LADDER / "heldout_manifest.path"
    if not pointer.exists():
        pytest.skip("held-out capture not on disk")
    return Path(pointer.read_text().strip())


def test_capture_source_targets_and_state_match_the_cnn_lane():
    from so_arm101_v2.learning.chunked import build_chunked_targets
    from so_arm101_v2.learning.pi05 import CaptureSource
    from so_arm101_v2.learning.tiny_model import normalize_act
    from so_arm101_v2.learning.vision import load_vision_frames

    manifest_path = _heldout_manifest()
    source = CaptureSource.load(manifest_path, name="heldout")
    manifest, _, arrays = load_vision_frames(manifest_path)
    lengths = [int(e["rows"]) for e in manifest["episodes"]]
    reference = build_chunked_targets(arrays, 90, episode_lengths=lengths)
    rows = np.array([0, 90, 470, 479, 480, 569, 4799])   # episode starts, ends and the boundary between episodes
    np.testing.assert_array_equal(source.targets(rows, 90), reference[rows])
    state = np.concatenate([normalize_act(np.asarray(arrays["current_act"], np.float32)),
                            np.asarray(arrays["progress"], np.float32)[:, None]], axis=1)
    np.testing.assert_array_equal(source.state(rows), state[rows])
    assert source.rows_at(90).shape[0] == len(lengths)


def test_quantile_round_trip_and_prompt_fits():
    torch = pytest.importorskip("torch")
    from so_arm101_v2.learning.pi05 import DEFAULT_TOKENIZER, PromptTokenizer, QuantileStats

    stats = QuantileStats(np.full(7, -0.5, np.float32), np.full(7, 0.5, np.float32), np.full(6, -0.2, np.float32), np.full(6, 0.3, np.float32))
    action = torch.linspace(-0.2, 0.3, 12).reshape(2, 1, 6)
    torch.testing.assert_close(stats.unnorm_action(stats.norm_action(action)), action)
    assert np.all(np.abs(stats.norm_state(np.full((1, 7), 9.0, np.float32))) <= 1.0)
    if not DEFAULT_TOKENIZER.exists():
        pytest.skip("tokenizer not downloaded")
    prompt = PromptTokenizer(str(DEFAULT_TOKENIZER), 64)
    tokens, masks = prompt(np.array([[-1.0, 1.0, 0.0, 0.5, -0.5, 0.99, -0.99]], np.float32), "cpu")
    assert tokens.shape == (1, 64) and int(masks.sum()) < 64
    assert prompt.prompts(np.zeros((1, 7), np.float32))[0].endswith(";\nAction: ")


def test_closure_matches_the_tranche_analysis():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.closure import ClosureKinematics, closure_row

    if not CLOSURE_REF.exists() or not TELEMETRY_4800.exists():
        pytest.skip("tranche analysis artifacts not on disk")
    kin = ClosureKinematics(MODEL)
    rows = [r for r in json.loads(CLOSURE_REF.read_text())["rows"] if r["run"] == "4800/shift12/s202"]
    assert len(rows) == 10
    for ref in rows:
        got = closure_row(kin, json.loads((TELEMETRY_4800 / f"vision.{ref['pose']}.repeat0.json").read_text()))
        for key in ("along", "perp", "height", "lat", "dz", "yaw_err"):
            assert got[key] == pytest.approx(ref[key], abs=1e-9)
