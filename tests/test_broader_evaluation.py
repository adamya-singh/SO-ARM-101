from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.chunked import (
    ChunkedCloneConfig,
    build_chunked_targets,
    train_chunked_clone,
)
from so_arm101_v2.simulation.broader import (
    _require_promoted_policy,
    memorization_signature,
    resolve_broader_evaluation_status,
)
from so_arm101_v2.simulation.margins import analyze_pick_place_margins, rollout_margins
from so_arm101_v2.simulation.recovery import evaluate_policy_anchor_starts

from conftest import REPOSITORY_ROOT
from test_chunked_promotion import _write_tiny_oracle_manifest


def _summary(scenario_id: str, success: bool, *, nonfinite: int = 0, clipped: int = 0) -> dict:
    return {
        "scenario_id": scenario_id,
        "success": success,
        "safety_counts": {
            "clipping_frames": clipped,
            "limiting_frames": 0,
            "nonfinite_frames": nonfinite,
            "unsafe_contact_frames": 0,
        },
    }


def test_memorization_signature_requires_nominal_pass_shifted_failure_and_finite_frames() -> None:
    nominal = [_summary("nominal", True)] * 3
    shifted_fail = [_summary("cube_x_plus_1p5mm", False)] * 3
    shifted_pass = [_summary("cube_x_plus_1p5mm", True)] * 3
    assert memorization_signature(nominal + shifted_fail) is True
    assert memorization_signature(nominal + shifted_pass) is False
    dirty_nominal = [_summary("nominal", True, clipped=2)] * 3
    assert memorization_signature(dirty_nominal + shifted_fail) is False
    nonfinite = [_summary("cube_x_plus_1p5mm", False, nonfinite=1)] * 3
    assert memorization_signature(nominal + nonfinite) is False
    with pytest.raises(ValueError, match="nominal and shifted"):
        memorization_signature(nominal)


def test_broader_status_resolver_enumerates_all_terminal_statuses() -> None:
    def record(**overrides) -> dict:
        base = {
            "promoted": {"stage_a_passed": False, "stage_b_passed": False},
            "memorization_signature": False,
            "branch_taken": False,
            "retrained": None,
            "nonfinite_frames_observed": False,
        }
        base.update(overrides)
        return base

    assert resolve_broader_evaluation_status(
        record(promoted={"stage_a_passed": True, "stage_b_passed": True})
    ) == "promoted_policy_robust"
    assert resolve_broader_evaluation_status(
        record(promoted={"stage_a_passed": True, "stage_b_passed": False})
    ) == "promoted_policy_passes_starts_fails_handoffs"
    assert resolve_broader_evaluation_status(record()) == "starts_not_resolved"
    assert resolve_broader_evaluation_status(
        record(nonfinite_frames_observed=True)
    ) == "starts_not_resolved"
    retrained_pass = {
        "capture_ok": True, "stage_a_passed": True, "stage_b_passed": True,
    }
    assert resolve_broader_evaluation_status(record(
        memorization_signature=True, branch_taken=True, retrained=retrained_pass,
    )) == "retrained_policy_robust"
    assert resolve_broader_evaluation_status(record(
        memorization_signature=True, branch_taken=True,
        retrained={**retrained_pass, "stage_b_passed": False},
    )) == "retrained_policy_passes_starts_fails_handoffs"
    assert resolve_broader_evaluation_status(record(
        memorization_signature=True, branch_taken=True,
        retrained={"capture_ok": False},
    )) == "starts_not_resolved"
    assert resolve_broader_evaluation_status(record(
        memorization_signature=True, branch_taken=True,
        retrained={**retrained_pass, "stage_a_passed": False},
    )) == "starts_not_resolved"
    with pytest.raises(ValueError, match="despite a promoted Stage A pass"):
        resolve_broader_evaluation_status(record(
            promoted={"stage_a_passed": True, "stage_b_passed": True},
            branch_taken=True,
        ))
    with pytest.raises(ValueError, match="without the memorization signature"):
        resolve_broader_evaluation_status(record(branch_taken=True, retrained={}))
    with pytest.raises(ValueError, match="signature present"):
        resolve_broader_evaluation_status(record(memorization_signature=True))


def test_broader_eligibility_rejects_unpromoted_gate_and_checkpoint_mismatch() -> None:
    gate = {
        "status": "promoted_noise_penalty_only",
        "candidates": [{
            "candidate_id": "noise_penalty_only",
            "state": "nominal_passed",
            "checkpoint_sha256": "a" * 64,
        }],
    }
    assert _require_promoted_policy(gate, "a" * 64)["candidate_id"] == "noise_penalty_only"
    with pytest.raises(ValueError, match="promoted gate status"):
        _require_promoted_policy({**gate, "status": "closed_loop_not_resolved"}, "a" * 64)
    with pytest.raises(ValueError, match="promoted candidate"):
        _require_promoted_policy(gate, "b" * 64)
    failed_state = {
        **gate,
        "candidates": [{**gate["candidates"][0], "state": "nominal_failed"}],
    }
    with pytest.raises(ValueError, match="promoted candidate"):
        _require_promoted_policy(failed_state, "a" * 64)


def test_chunked_targets_single_episode_lengths_match_none_path() -> None:
    rows = 7
    arrays = {
        "current_act": np.random.RandomState(0).rand(rows, 6).astype(np.float32) * 0.1,
        "executed_act": np.random.RandomState(1).rand(rows, 6).astype(np.float32) * 0.1,
    }
    via_none = build_chunked_targets(arrays, chunk_horizon=3, episode_lengths=None)
    via_lengths = build_chunked_targets(arrays, chunk_horizon=3, episode_lengths=[rows])
    np.testing.assert_array_equal(via_none, via_lengths)


def test_train_chunked_clone_derived_episode_lengths_preserve_legacy_digest(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "config", "target", "chunk_padding", "offline_role",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]
    with pytest.raises(ValueError, match="episode lengths disagree"):
        # A manifest whose episode records disagree with its row count is
        # rejected before any training.
        broken_dir = tmp_path / "broken"
        broken_dir.mkdir()
        broken_path, broken = _write_tiny_oracle_manifest(broken_dir)
        payload = json.loads(broken_path.read_text(encoding="utf-8"))
        payload["episodes"][0]["rows"] = 3
        payload.pop("content_sha256")
        payload["content_sha256"] = content_sha256(payload)
        broken_path.write_text(json.dumps(payload), encoding="utf-8")
        train_chunked_clone(
            broken_path, tmp_path / "broken-output",
            config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
        )


def test_margin_analyzer_formulas_on_synthetic_telemetry() -> None:
    low = np.full(6, -1.0)
    high = np.full(6, 1.0)
    caps = np.full(6, 0.5)
    napkin_half = np.array([0.0254, 0.0254, 0.0005])
    corners = [[0.01, -0.01, 0.0]] * 8
    rows = [
        {
            "requested_act": [0.5] * 6,
            "current_act": [0.2] * 6,
            "pickup_measurement": {"strict_bilateral_grasp": False},
            "placement_measurement": {
                "cube_footprint_inside": False, "cube_support_error_m": 0.01,
            },
            "contact": {"napkin_local_cube_corners": corners},
        },
        {
            "requested_act": [0.9] * 6,
            "current_act": [0.5] * 6,
            "pickup_measurement": {"strict_bilateral_grasp": True},
            "placement_measurement": {
                "cube_footprint_inside": False, "cube_support_error_m": 0.01,
            },
            "contact": {"napkin_local_cube_corners": corners},
        },
        {
            "requested_act": [0.9] * 6,
            "current_act": [0.9] * 6,
            "pickup_measurement": {"strict_bilateral_grasp": True},
            "placement_measurement": {
                "cube_footprint_inside": True, "cube_support_error_m": 0.0004,
            },
            "contact": {"napkin_local_cube_corners": corners},
        },
    ]
    margins = rollout_margins(
        rows, act_low=low, act_high=high, delta_caps=caps, napkin_half=napkin_half,
    )
    assert margins["envelope_headroom_act"]["global_min"] == pytest.approx(0.1)
    assert margins["delta_usage"]["max_fraction_of_cap"] == pytest.approx(0.8)
    assert margins["delta_usage"]["steps_above_0p9"] == 0
    assert margins["strict_grasp"]["total_frames"] == 2
    assert margins["strict_grasp"]["longest_streak"] == 2
    assert margins["strict_grasp"]["hold_slack_frames"] == -28
    assert margins["placement"]["final_footprint_inside"] is True
    assert margins["placement"]["min_footprint_margin_m"] == pytest.approx(0.0134)
    assert margins["placement"]["final_support_margin_m"] == pytest.approx(0.0006)


def test_margin_analyzer_idempotent_immutable(tmp_path: Path) -> None:
    telemetry = {
        "rows": [{
            "requested_act": [0.1] * 6,
            "current_act": [0.1] * 6,
            "pickup_measurement": {"strict_bilateral_grasp": False},
            "placement_measurement": {
                "cube_footprint_inside": False, "cube_support_error_m": 0.01,
            },
            "contact": {"napkin_local_cube_corners": [[0.0, 0.0, 0.0]] * 8},
        }],
    }
    telemetry["content_sha256"] = content_sha256(telemetry)
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry), encoding="utf-8")
    evaluation = {
        "rollouts": [{
            "scenario_id": "nominal", "repeat": 0, "success": False,
            "telemetry_path": str(telemetry_path),
        }],
    }
    evaluation["content_sha256"] = content_sha256(evaluation)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    model = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    first = analyze_pick_place_margins(
        evaluation_path, tmp_path / "output", mujoco_model_path=model,
    )
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["rollouts"][0]["margins"]["strict_grasp"]["total_frames"] == 0
    assert report["napkin_half_size_m"][0] == pytest.approx(0.0254)
    again = analyze_pick_place_margins(
        evaluation_path, tmp_path / "output", mujoco_model_path=model,
    )
    assert again == first
    telemetry_path.write_text(json.dumps({**telemetry, "rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry content hash mismatch"):
        analyze_pick_place_margins(
            evaluation_path, tmp_path / "other", mujoco_model_path=model,
        )


def test_policy_anchor_starts_requires_full_nominal_episode(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
    )
    recovery = {
        "schema_version": 1,
        "source_manifest_content_sha256": json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["content_sha256"],
        "arrays": {"path": "recovery_examples.npz", "sha256": "0" * 64, "rows": 8},
    }
    recovery["content_sha256"] = content_sha256(recovery)
    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    stream_path = tmp_path / "recovery_examples.npz"
    np.savez_compressed(stream_path, action_index=np.arange(8))
    raw = stream_path.read_bytes()
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload.pop("content_sha256")
    payload["arrays"]["sha256"] = hashlib.sha256(raw).hexdigest()
    payload["content_sha256"] = content_sha256(payload)
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    model = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    with pytest.raises(ValueError, match="450-row nominal episode"):
        evaluate_policy_anchor_starts(
            model, manifest_path, recovery_path, result.checkpoint, tmp_path / "anchors",
        )
