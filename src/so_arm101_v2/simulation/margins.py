"""Immutable margin analysis for pick-place evaluations.

Margins are telemetry, never a gate: they quantify how much slack a policy
has against the strict-grasp hold requirement, the command envelope, the
per-step delta limiter, and the napkin placement tolerances.  Thin margins on
a passing scenario predict where perturbed scenarios will fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from so_arm101_v2.contracts import (
    JOINT_NAMES,
    effective_safe_act_bounds,
    load_task_contract,
)
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

from .observability import _load_hashed_json

ANALYZER_VERSION = "pick_place_margin_analyzer_v1"
EDGE_MARGIN_M = 0.002
SUPPORT_CAP_M = 0.001
HOLD_FRAMES = 30
HIGH_DELTA_USAGE = 0.9


def _napkin_half_size(model_path: Path) -> list[float]:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("margin analysis requires the 'sim' extra") from exc
    model = mujoco.MjModel.from_xml_path(str(model_path))
    napkin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "napkin")
    if napkin < 0:
        raise ValueError("MuJoCo model is missing napkin geom")
    return [float(value) for value in model.geom_size[napkin]]


def rollout_margins(
    rows: list[Mapping[str, Any]],
    *,
    act_low: np.ndarray,
    act_high: np.ndarray,
    delta_caps: np.ndarray,
    napkin_half: np.ndarray,
) -> dict[str, Any]:
    """Pure per-rollout margin metrics from telemetry rows."""
    if not rows:
        raise ValueError("margin analysis requires at least one telemetry row")
    requested = np.asarray([row["requested_act"] for row in rows], dtype=np.float64)
    current = np.asarray([row["current_act"] for row in rows], dtype=np.float64)
    envelope = np.minimum(requested - act_low, act_high - requested)
    per_joint_envelope = envelope.min(axis=0)
    delta_usage = np.abs(requested - current) / delta_caps
    per_step_usage = delta_usage.max(axis=1)

    strict = [bool(row["pickup_measurement"]["strict_bilateral_grasp"]) for row in rows]
    longest = 0
    streak = 0
    for value in strict:
        streak = streak + 1 if value else 0
        longest = max(longest, streak)

    footprint_limit = napkin_half[:2] - EDGE_MARGIN_M
    footprint_margins: list[float] = []
    for row in rows:
        if not row["placement_measurement"]["cube_footprint_inside"]:
            continue
        corners = np.asarray(row["contact"]["napkin_local_cube_corners"], dtype=np.float64)
        footprint_margins.append(float(
            np.min(footprint_limit[None, :] - np.abs(corners[:, :2]))
        ))
    final = rows[-1]
    return {
        "actions": len(rows),
        "envelope_headroom_act": {
            "per_joint_min": per_joint_envelope.tolist(),
            "global_min": float(per_joint_envelope.min()),
            "worst_joint": JOINT_NAMES[int(np.argmin(per_joint_envelope))],
        },
        "delta_usage": {
            "max_fraction_of_cap": float(per_step_usage.max()),
            "steps_above_0p9": int(np.count_nonzero(per_step_usage > HIGH_DELTA_USAGE)),
        },
        "strict_grasp": {
            "total_frames": int(sum(strict)),
            "longest_streak": int(longest),
            "hold_slack_frames": int(longest - HOLD_FRAMES),
        },
        "placement": {
            "final_footprint_inside": bool(
                final["placement_measurement"]["cube_footprint_inside"]
            ),
            "min_footprint_margin_m": (
                min(footprint_margins) if footprint_margins else None
            ),
            "final_footprint_margin_m": (
                footprint_margins[-1] if footprint_margins else None
            ),
            "final_support_margin_m": float(
                SUPPORT_CAP_M - abs(final["placement_measurement"]["cube_support_error_m"])
            ),
        },
    }


def analyze_pick_place_margins(
    evaluation_json_path: str | Path,
    output_dir: str | Path,
    *,
    mujoco_model_path: str | Path,
) -> Path:
    """Write an immutable margins report for every rollout of an evaluation."""
    evaluation_path = Path(evaluation_json_path).resolve()
    model_path = Path(mujoco_model_path).resolve()
    evaluation = _load_hashed_json(evaluation_path, label="evaluation report")
    napkin_half = np.asarray(_napkin_half_size(model_path), dtype=np.float64)
    act_low, act_high = effective_safe_act_bounds()
    delta_caps = np.asarray(
        load_task_contract("fixed_cube_pickup_v1").safety.maximum_act_delta_per_step,
        dtype=np.float64,
    )
    identity = {
        "schema_version": 1,
        "experiment": ANALYZER_VERSION,
        "evaluation_content_sha256": evaluation["content_sha256"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "napkin_half_size_m": napkin_half.tolist(),
        "constants": {
            "edge_margin_m": EDGE_MARGIN_M,
            "support_cap_m": SUPPORT_CAP_M,
            "hold_frames": HOLD_FRAMES,
            "high_delta_usage_fraction": HIGH_DELTA_USAGE,
        },
    }
    digest = content_sha256(identity)
    directory = Path(output_dir) / "margin_analyses" / digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="margin analysis report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable margin analysis differs: {directory}")
        return report_path

    records: list[dict[str, Any]] = []
    for item in evaluation["rollouts"]:
        telemetry = json.loads(Path(item["telemetry_path"]).read_text(encoding="utf-8"))
        stated = telemetry.get("content_sha256")
        body = dict(telemetry)
        body.pop("content_sha256", None)
        if stated != content_sha256(body):
            raise ValueError(f"telemetry content hash mismatch: {item['telemetry_path']}")
        records.append({
            "scenario_id": item["scenario_id"],
            "repeat": int(item["repeat"]),
            "success": bool(item["success"]),
            "telemetry_content_sha256": stated,
            "margins": rollout_margins(
                telemetry["rows"],
                act_low=act_low, act_high=act_high,
                delta_caps=delta_caps, napkin_half=napkin_half,
            ),
        })

    def _scenario_minima(scenario_records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "envelope_headroom_min": min(
                item["margins"]["envelope_headroom_act"]["global_min"]
                for item in scenario_records
            ),
            "delta_usage_max": max(
                item["margins"]["delta_usage"]["max_fraction_of_cap"]
                for item in scenario_records
            ),
            "hold_slack_min": min(
                item["margins"]["strict_grasp"]["hold_slack_frames"]
                for item in scenario_records
            ),
        }

    scenarios = sorted({item["scenario_id"] for item in records})
    report = {
        **identity,
        "analysis_digest": digest,
        "rollouts": records,
        "per_scenario_minima": {
            scenario: _scenario_minima([
                item for item in records if item["scenario_id"] == scenario
            ])
            for scenario in scenarios
        },
        "claim": "margin_telemetry_only_never_a_gate",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return report_path


__all__ = [
    "ANALYZER_VERSION",
    "analyze_pick_place_margins",
    "rollout_margins",
]
