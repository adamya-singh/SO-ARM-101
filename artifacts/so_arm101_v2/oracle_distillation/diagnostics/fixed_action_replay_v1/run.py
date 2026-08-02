"""Predict on stored expert states, then replay the fixed commands in MuJoCo."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import (
    PickPlaceEvaluationState,
    evaluate_pick_place_step,
    load_pick_place_contract,
)
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.learning.oracle_distillation import build_oracle_features
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.clone_policy import OracleCloneCheckpointPolicy
from so_arm101_v2.simulation.rollout import _VideoWriter
from so_arm101_v2.simulation.suites import load_simulation_suite


REPOSITORY = Path(__file__).resolve().parents[5]
ARTIFACTS = REPOSITORY / "artifacts/so_arm101_v2/oracle_distillation"
MANIFEST = ARTIFACTS / "oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json"
CHECKPOINT = ARTIFACTS / "models/phase_state/d5f96d397bd9b915/model.pt"
OFFLINE_REPORT = CHECKPOINT.with_name("report.json")
AUTONOMOUS_EVALUATION = ARTIFACTS / (
    "clone_evaluations/943cf536710e3d84/policies/fixed_pick_place_v3.nominal/evaluation.json"
)
AUTONOMOUS_TELEMETRY = AUTONOMOUS_EVALUATION.parent / (
    "telemetry/phase_state.seed101.nominal.repeat0.json"
)
DIVERGENCE_REPORT = ARTIFACTS / "diagnostics/first_divergence_v1/report.json"
MODEL = REPOSITORY / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"
OUTPUT = Path(__file__).resolve().parent
PREDICTIONS = OUTPUT / "teacher_state_predictions.npz"
TELEMETRY = OUTPUT / "fixed_action_replay_telemetry.json"
WRIST_VIDEO = OUTPUT / "fixed_action_replay.wrist.mp4"
OVERVIEW_VIDEO = OUTPUT / "fixed_action_replay.overview.mp4"
REPORT = OUTPUT / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_true(values: list[bool] | np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if not len(indices) else int(indices[0])


def last_true(values: list[bool] | np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if not len(indices) else int(indices[-1])


def event_index(rows: list[dict[str, Any]], name: str) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(rows)
            if name in row["pickup_events"] or name in row["placement_events"]
        ),
        None,
    )


def write_immutable_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    path.write_bytes(value)


def state_dict(snapshot: Any) -> dict[str, list[float]]:
    return {
        name: getattr(snapshot, name).tolist()
        for name in snapshot.__dataclass_fields__
    }


def main() -> None:
    outputs = (PREDICTIONS, TELEMETRY, WRIST_VIDEO, OVERVIEW_VIDEO, REPORT)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"diagnostic outputs already exist; refusing to overwrite: {existing}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    offline_report = json.loads(OFFLINE_REPORT.read_text(encoding="utf-8"))
    autonomous_evaluation = json.loads(AUTONOMOUS_EVALUATION.read_text(encoding="utf-8"))
    autonomous_telemetry = json.loads(AUTONOMOUS_TELEMETRY.read_text(encoding="utf-8"))
    divergence_report = json.loads(DIVERGENCE_REPORT.read_text(encoding="utf-8"))
    arrays_path = MANIFEST.parent / manifest["arrays"]["path"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        oracle = {name: archive[name].copy() for name in archive.files}

    policy = OracleCloneCheckpointPolicy(CHECKPOINT)
    features, mean, std = build_oracle_features(
        policy.kind,
        oracle,
        extras_mean=policy.extras_mean,
        extras_std=policy.extras_std,
    )
    with policy.torch.inference_mode():
        predicted_normalized_delta = policy.model(policy.torch.from_numpy(features)).numpy().astype(np.float32)
    predicted_requested_act = (
        np.asarray(oracle["current_act"], dtype=np.float32)
        + predicted_normalized_delta * policy.maximum_delta
    ).astype(np.float32)
    target_normalized_delta = (
        np.asarray(oracle["executed_delta_act"], dtype=np.float32) / policy.maximum_delta
    )
    normalized_mse = float(np.mean(np.square(predicted_normalized_delta - target_normalized_delta)))
    maximum_act_error = float(
        np.max(np.abs(predicted_requested_act - np.asarray(oracle["executed_act"], dtype=np.float32)))
    )
    if normalized_mse != offline_report["normalized_mse"] or maximum_act_error != offline_report["max_act_error"]:
        raise RuntimeError(
            "teacher-state inference did not exactly reproduce the checkpoint report: "
            f"mse={normalized_mse}, max={maximum_act_error}"
        )
    prediction_buffer = io.BytesIO()
    np.savez_compressed(
        prediction_buffer,
        action_index=np.asarray(oracle["action_index"], dtype=np.int64),
        predicted_normalized_delta=predicted_normalized_delta,
        predicted_requested_act=predicted_requested_act,
        teacher_current_act=np.asarray(oracle["current_act"], dtype=np.float32),
        teacher_executed_act=np.asarray(oracle["executed_act"], dtype=np.float32),
    )
    write_immutable_bytes(PREDICTIONS, prediction_buffer.getvalue())

    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")
    contract = load_pick_place_contract(suite.task_contract)
    adapter = MujocoTaskAdapter(MODEL)
    wrist = _VideoWriter(WRIST_VIDEO)
    overview = _VideoWriter(OVERVIEW_VIDEO)
    state = PickPlaceEvaluationState()
    rows: list[dict[str, Any]] = []
    all_pickup_events: set[str] = set()
    all_placement_events: set[str] = set()
    safety_counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    maximum_height = float("-inf")
    terminal_evaluation = None
    try:
        adapter.reset(scenario)
        for action_index, fixed_requested in enumerate(predicted_requested_act):
            pre = adapter.privileged_state()
            wrist.add(adapter.render("wrist_camera"))
            overview.add(adapter.render("camera_side"))
            command = adapter.apply_policy_command(fixed_requested)
            substeps = adapter.advance_control_period()
            post = adapter.privileged_state()
            measurement, diagnostics = adapter.pick_place_measurement(
                command,
                footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
            )
            if not state.completed:
                state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                terminal_evaluation = evaluation
                pickup_events = [event.value for event in evaluation.pickup_events]
                placement_events = [event.value for event in evaluation.events]
            else:
                pickup_events = []
                placement_events = []
            all_pickup_events.update(pickup_events)
            all_placement_events.update(placement_events)
            safety_counts["clip"] += int(measurement.pickup.command_bound_violation)
            safety_counts["limit"] += int(measurement.pickup.delta_limiter_activated)
            safety_counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
            safety_counts["unsafe"] += int(measurement.pickup.unsafe_contact)
            maximum_height = max(maximum_height, measurement.pickup.cube_height_gain_m)
            rows.append({
                "action_index_zero_based": action_index,
                "action_number_one_based": action_index + 1,
                "simulation_time_s": float(adapter.data.time),
                "physics_substeps": substeps,
                "pre": state_dict(pre),
                "teacher_state_predicted_requested_act": fixed_requested.tolist(),
                "teacher_requested_act": oracle["requested_act"][action_index].tolist(),
                "teacher_executed_act": oracle["executed_act"][action_index].tolist(),
                "requested_act": command.requested_act.tolist(),
                "executed_act": command.executed_act.tolist(),
                "safety": {
                    "command_bound_violation": command.command_bound_violation,
                    "delta_limiter_activated": command.delta_limiter_activated,
                    "nonfinite_command": command.nonfinite_command,
                    "act_clip_mask": command.act_clip_mask.tolist(),
                    "mujoco_clip_mask": command.mujoco_clip_mask.tolist(),
                    "relative_limit_mask": command.relative_limit_mask.tolist(),
                },
                "post": state_dict(post),
                "pickup_measurement": asdict(measurement.pickup),
                "placement_measurement": {
                    "cube_footprint_inside": measurement.cube_footprint_inside,
                    "cube_support_error_m": measurement.cube_support_error_m,
                    "cube_linear_speed_m_s": measurement.cube_linear_speed_m_s,
                    "cube_angular_speed_rad_s": measurement.cube_angular_speed_rad_s,
                    "gripper_act": measurement.gripper_act,
                },
                "contact": diagnostics,
                "pickup_events": pickup_events,
                "placement_events": placement_events,
                "evaluator_already_terminal": bool(state.completed and not pickup_events and not placement_events),
            })
    finally:
        wrist.close()
        overview.close()
        adapter.close()
    if terminal_evaluation is None or len(rows) != 450:
        raise RuntimeError("fixed-action replay did not complete all 450 commands")

    telemetry_payload = {
        "schema_version": 1,
        "claim": "fixed_action_replay_from_teacher_state_predictions_no_feedback_inference",
        "policy_id": policy.policy_id,
        "suite_id": suite.suite_id,
        "scenario_id": scenario.scenario_id,
        "reward_used": False,
        "prediction_source": PREDICTIONS.name,
        "rows": rows,
    }
    telemetry_payload["content_sha256"] = content_sha256(telemetry_payload)
    write_immutable_json(TELEMETRY, telemetry_payload)

    fixed_any = [bool(row["pickup_measurement"]["any_contact"]) for row in rows]
    fixed_bilateral = [bool(row["pickup_measurement"]["bilateral_interior_contact"]) for row in rows]
    fixed_strict = [bool(row["pickup_measurement"]["strict_bilateral_grasp"]) for row in rows]
    autonomous_rows = autonomous_telemetry["rows"]
    autonomous_rollout = autonomous_evaluation["rollouts"][0]
    teacher_events = manifest["episodes"][0]["events"]
    teacher_event_rows = [
        {"pickup_events": row["pickup_events"], "placement_events": row["placement_events"]}
        for row in teacher_events
    ]
    report = {
        "schema_version": 1,
        "claim": "decisive_fixed_action_replay_only_no_training_or_policy_change",
        "setup": {
            "inference": "one checkpoint prediction per stored expert pre-action state; no simulator-produced state was used for inference",
            "execution": "fresh nominal reset followed by the 450 fixed absolute ACT commands in action-index order",
            "safety": "each fixed command passed through MujocoTaskAdapter.apply_policy_command before execution",
            "reward_used": False,
            "actions_predicted": int(predicted_requested_act.shape[0]),
            "actions_replayed": len(rows),
        },
        "inputs": {
            "manifest": {"path": str(MANIFEST.relative_to(REPOSITORY)), "sha256": sha256(MANIFEST), "content_sha256": manifest["content_sha256"]},
            "oracle_arrays": {"path": str(arrays_path.relative_to(REPOSITORY)), "sha256": sha256(arrays_path)},
            "checkpoint": {"path": str(CHECKPOINT.relative_to(REPOSITORY)), "sha256": sha256(CHECKPOINT)},
            "offline_report": {"path": str(OFFLINE_REPORT.relative_to(REPOSITORY)), "sha256": sha256(OFFLINE_REPORT), "content_sha256": offline_report["content_sha256"]},
            "autonomous_evaluation": {"path": str(AUTONOMOUS_EVALUATION.relative_to(REPOSITORY)), "sha256": sha256(AUTONOMOUS_EVALUATION), "content_sha256": autonomous_evaluation["content_sha256"]},
            "autonomous_telemetry": {"path": str(AUTONOMOUS_TELEMETRY.relative_to(REPOSITORY)), "sha256": sha256(AUTONOMOUS_TELEMETRY), "content_sha256": autonomous_telemetry["content_sha256"]},
            "first_divergence_report": {"path": str(DIVERGENCE_REPORT.relative_to(REPOSITORY)), "sha256": sha256(DIVERGENCE_REPORT)},
            "mujoco_model": {"path": str(MODEL.relative_to(REPOSITORY)), "sha256": sha256(MODEL)},
        },
        "teacher_state_prediction_reproduction": {
            "normalized_delta_mse": normalized_mse,
            "maximum_act_error": maximum_act_error,
            "exactly_matches_offline_report": True,
            "feature_mean_matches_checkpoint": bool(np.array_equal(mean, policy.extras_mean)),
            "feature_std_matches_checkpoint": bool(np.array_equal(std, policy.extras_std)),
        },
        "fixed_action_replay": {
            "success": terminal_evaluation.success,
            "outcome": terminal_evaluation.outcome.value,
            "terminal_action_count": terminal_evaluation.actions_evaluated,
            "pickup_completed": terminal_evaluation.pickup_completed,
            "settled_frames": terminal_evaluation.settled_frames,
            "maximum_cube_height_gain_m": maximum_height,
            "safety_counts": safety_counts,
            "event_indices_zero_based": {
                name: event_index(rows, name)
                for name in (
                    "reach", "first_contact", "bilateral_interior_contact", "strict_grasp_acquired",
                    "lift_5mm", "entered_placement_region", "released", "settled", "retreated", "success",
                )
            },
            "contact_spans_zero_based": {
                "any_contact": {"first": first_true(fixed_any), "last": last_true(fixed_any), "frames": int(sum(fixed_any))},
                "bilateral_interior_contact": {"first": first_true(fixed_bilateral), "last": last_true(fixed_bilateral), "frames": int(sum(fixed_bilateral))},
                "strict_bilateral_grasp": {"first": first_true(fixed_strict), "last": last_true(fixed_strict), "frames": int(sum(fixed_strict))},
            },
            "all_pickup_events": sorted(all_pickup_events),
            "all_placement_events": sorted(all_placement_events),
            "final_robot_qpos": rows[-1]["post"]["current_act"],
            "final_cube_position": rows[-1]["post"]["cube_position"],
        },
        "comparison": {
            "expert": {
                "success": manifest["episodes"][0]["final_evaluation"]["success"],
                "terminal_action_count": manifest["episodes"][0]["final_evaluation"]["actions_evaluated"],
                "maximum_cube_height_gain_m": float(np.max(oracle["post_cube_height_gain_m"])),
                "safety_counts": manifest["episodes"][0]["safety_counts"],
                "event_indices_zero_based": {
                    name: event_index(teacher_event_rows, name)
                    for name in (
                        "reach", "first_contact", "bilateral_interior_contact", "strict_grasp_acquired",
                        "lift_5mm", "entered_placement_region", "released", "settled", "retreated", "success",
                    )
                },
            },
            "autonomous_checkpoint_rollout": {
                "success": autonomous_rollout["success"],
                "failure_category": autonomous_rollout["failure_category"],
                "actions": autonomous_rollout["actions"],
                "maximum_cube_height_gain_m": autonomous_rollout["maximum_height_gain_m"],
                "safety_counts": {
                    "clip": autonomous_rollout["clipping_frames"],
                    "limit": autonomous_rollout["limiting_frames"],
                    "nonfinite": autonomous_rollout["nonfinite_frames"],
                    "unsafe": autonomous_rollout["unsafe_contact_frames"],
                },
                "event_indices_zero_based": {
                    name: event_index(autonomous_rows, name)
                    for name in (
                        "reach", "first_contact", "bilateral_interior_contact", "strict_grasp_acquired",
                        "lift_5mm", "entered_placement_region", "released", "settled", "retreated", "success",
                    )
                },
            },
        },
        "interpretation": {
            "fixed_sequence_succeeds": terminal_evaluation.success,
            "covariate_shift_confirmed_if_fixed_succeeds": bool(terminal_evaluation.success and not autonomous_rollout["success"]),
            "offline_gate_task_sensitivity_disproved_if_fixed_fails": bool(not terminal_evaluation.success),
        },
        "outputs": {
            "predictions": {"path": PREDICTIONS.name, "sha256": sha256(PREDICTIONS)},
            "telemetry": {"path": TELEMETRY.name, "sha256": sha256(TELEMETRY), "content_sha256": telemetry_payload["content_sha256"]},
            "wrist_video": {"path": WRIST_VIDEO.name, "sha256": sha256(WRIST_VIDEO)},
            "overview_video": {"path": OVERVIEW_VIDEO.name, "sha256": sha256(OVERVIEW_VIDEO)},
        },
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
