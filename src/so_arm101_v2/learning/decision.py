"""Machine-readable evidence gate that keeps ACT absent until justified."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

from .full_dataset import Lead3TrainingCache, SmallModelComparison, SmallModelKind


TEMPORAL_FAILURES = {
    "bad_closure_timing", "grasp_loss", "grasp_no_lift", "lift_not_sustained", "oscillation",
}


def _episode_reversal_rates(values: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    rates = []
    for episode in sorted(set(episodes.tolist())):
        sequence = values[episodes == episode].astype(np.float64)
        velocity = np.diff(sequence, axis=0)
        if len(velocity) < 2:
            rates.append(0.0)
            continue
        reversal = (np.sign(velocity[1:]) != np.sign(velocity[:-1])) & (np.abs(velocity[1:]) > 1e-4) & (np.abs(velocity[:-1]) > 1e-4)
        rates.append(float(np.mean(reversal)))
    return np.asarray(rates, dtype=np.float64)


def _rollout_reversal_rate(telemetry_path: str) -> float:
    payload = json.loads(Path(telemetry_path).read_text(encoding="utf-8"))
    commands = np.asarray([row["requested_act"] for row in payload["rows"]], dtype=np.float64)
    if len(commands) < 3:
        return 0.0
    velocity = np.diff(commands, axis=0)
    reversal = (np.sign(velocity[1:]) != np.sign(velocity[:-1])) & (np.abs(velocity[1:]) > 1e-4) & (np.abs(velocity[:-1]) > 1e-4)
    return float(np.mean(reversal))


def build_act_decision_report(
    cache: Lead3TrainingCache,
    comparison: SmallModelComparison | str | Path,
    image_ablation_report: str | Path,
    preflight_report: str | Path,
    simulation_report: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Combine offline and rollout evidence without importing or implementing ACT."""
    comparison_path = comparison.report_json if isinstance(comparison, SmallModelComparison) else Path(comparison)
    offline = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    ablation = json.loads(Path(image_ablation_report).read_text(encoding="utf-8"))
    preflight = json.loads(Path(preflight_report).read_text(encoding="utf-8"))
    simulation = json.loads(Path(simulation_report).read_text(encoding="utf-8"))
    digests = {offline.get("dataset_digest"), ablation.get("dataset_digest"), cache.dataset_digest}
    if len(digests) != 1:
        raise ValueError("ACT decision inputs use different dataset digests")

    baseline_mse = float(offline["development_baselines"]["splits"]["validation"]["current_pose"]["normalized_mse"])
    by_kind: dict[str, list[dict[str, Any]]] = {kind.value: [] for kind in SmallModelKind}
    for run in offline["runs"]:
        by_kind[run["model_kind"]].append(run)
    kind_means = {
        kind: float(np.mean([float(run["validation_normalized_mse"]) for run in runs]))
        for kind, runs in by_kind.items()
    }
    selected_kind = min(kind_means, key=kind_means.get)
    selected_runs = by_kind[selected_kind]
    improvements = [
        (baseline_mse - float(run["validation_normalized_mse"])) / baseline_mse
        for run in selected_runs
    ]
    offline_gate = bool(np.mean(improvements) >= 0.05 and all(value > 0 for value in improvements))

    environment_gate = bool(preflight.get("environment_proven"))
    rollout_rows = [
        row for row in simulation.get("rollouts", [])
        if str(row["policy_id"]).startswith(selected_kind + ".seed") and ".black" not in str(row["policy_id"])
    ]
    scenario_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rollout_rows:
        scenario_rows.setdefault((str(row["policy_id"]), str(row["scenario_id"])), row)
    matched = list(scenario_rows.values())
    total = len(matched)
    reach_rate = float(np.mean([row["reached"] for row in matched])) if matched else 0.0
    contact_rate = float(np.mean([row["contacted"] for row in matched])) if matched else 0.0
    success_rate = float(np.mean([row["success"] for row in matched])) if matched else 0.0
    total_actions = sum(int(row["actions"]) for row in matched)
    intervention_rate = (
        sum(int(row["clipping_frames"]) + int(row["limiting_frames"]) for row in matched) / total_actions
        if total_actions else 1.0
    )
    failed = [row for row in matched if not row["success"]]
    temporal_fraction = (
        float(np.mean([row["failure_category"] in TEMPORAL_FAILURES for row in failed])) if failed else 0.0
    )
    validation = cache.indices("validation")
    demo_rates = _episode_reversal_rates(cache.targets[validation], cache.episode_ids[validation])
    reference_p90 = float(np.quantile(demo_rates, 0.9))
    rollout_reversal = float(np.mean([_rollout_reversal_rate(row["telemetry_path"]) for row in matched])) if matched else 0.0
    reversal_ratio = rollout_reversal / max(reference_p90, 1e-12)
    temporal_gate = bool(
        reach_rate >= 0.80 and contact_rate >= 0.50 and success_rate < 0.80
        and intervention_rate < 0.01 and temporal_fraction >= 0.60 and reversal_ratio >= 2.0
    )

    if not environment_gate:
        status, recommendation = "blocked_environment", "repair or prove the simulator and strict evaluator"
    elif not offline_gate:
        status, recommendation = "blocked_offline", "investigate target construction, capacity, or demonstration consistency"
    elif success_rate >= 0.80:
        status, recommendation = "not_needed", "retain the simpler deterministic policy"
    elif temporal_gate:
        status, recommendation = "candidate_temporal", "review matched rollout videos before proposing an ACT tranche"
    elif reach_rate < 0.80 or contact_rate < 0.50:
        status = "blocked_observation_or_domain"
        recommendation = "investigate vision usefulness and physical-to-simulation observation shift"
    elif intervention_rate >= 0.01:
        status, recommendation = "blocked_safety", "fix output bounds and execution semantics"
    else:
        status, recommendation = "blocked_unspecific_failure", "categorize the remaining failure or collect targeted recovery data"

    report: dict[str, Any] = {
        "schema_version": 1, "act_implemented": False, "status": status,
        "recommendation": recommendation, "dataset_digest": cache.dataset_digest,
        "selected_simple_model": selected_kind,
        "gates": {
            "environment_proven": environment_gate,
            "offline_five_percent_every_seed": offline_gate,
            "specific_temporal_failure": temporal_gate,
        },
        "offline": {
            "current_pose_validation_normalized_mse": baseline_mse,
            "selected_model_validation_normalized_mse_by_seed": [float(row["validation_normalized_mse"]) for row in selected_runs],
            "relative_improvement_by_seed": improvements,
            "image_signal_status": ablation["status"],
        },
        "closed_loop": {
            "matched_scenario_checkpoint_pairs": total, "reach_rate": reach_rate,
            "contact_rate": contact_rate, "success_rate": success_rate,
            "clip_or_limit_frame_rate": intervention_rate,
            "temporal_failure_fraction": temporal_fraction,
            "validation_demo_reversal_rate_p90": reference_p90,
            "rollout_reversal_rate": rollout_reversal, "reversal_ratio": reversal_ratio,
        },
        "test_evaluated": False,
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(Path(output_path), report)
    return report


__all__ = ["build_act_decision_report"]
