"""Reproduce and align the nominal oracle-clone failure without changing source."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from so_arm101_v2.contracts import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
from so_arm101_v2.learning.tiny_model import normalize_act
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.clone_policy import OracleCloneCheckpointPolicy
from so_arm101_v2.simulation.suites import load_simulation_suite


REPOSITORY = Path(__file__).resolve().parents[5]
ARTIFACTS = REPOSITORY / "artifacts/so_arm101_v2/oracle_distillation"
MANIFEST = ARTIFACTS / "oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json"
CHECKPOINT = ARTIFACTS / "models/phase_state/d5f96d397bd9b915/model.pt"
STORED_TELEMETRY = ARTIFACTS / (
    "clone_evaluations/943cf536710e3d84/policies/fixed_pick_place_v3.nominal/"
    "telemetry/phase_state.seed101.nominal.repeat0.json"
)
MODEL = REPOSITORY / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"
OUTPUT = Path(__file__).resolve().parent
ALIGNED = OUTPUT / "aligned_timesteps.jsonl"
REPORT = OUTPUT / "report.json"

STAGES = (
    "start_to_above", "above_to_lowered", "lowered_to_descended",
    "descended_to_engaged", "engaged_to_seated", "seated_to_half_closed",
    "half_closed_to_closed", "closed_hold", "closed_to_lift", "lift_hold",
    "lift_to_traverse", "traverse_to_set_down", "set_down_to_released",
    "released_hold", "released_to_retreat",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_over(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values > threshold)
    return None if not len(indices) else int(indices[0])


def norm_quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    # q and -q encode the same rotation.
    return float(min(np.linalg.norm(left - right), np.linalg.norm(left + right)))


def main() -> None:
    if ALIGNED.exists() or REPORT.exists():
        raise FileExistsError("diagnostic outputs already exist; refusing to overwrite")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stored = json.loads(STORED_TELEMETRY.read_text(encoding="utf-8"))
    with np.load(MANIFEST.parent / manifest["arrays"]["path"], allow_pickle=False) as archive:
        oracle = {name: archive[name].copy() for name in archive.files}

    boundaries = tuple(manifest["episodes"][0]["waypoint_boundaries"])
    teacher_events = manifest["episodes"][0]["events"]
    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")
    contract = load_pick_place_contract(suite.task_contract)
    policy = OracleCloneCheckpointPolicy(CHECKPOINT)
    adapter = MujocoTaskAdapter(MODEL)
    state = PickPlaceEvaluationState()
    fresh: list[dict[str, object]] = []
    try:
        adapter.reset(scenario)
        policy.reset(adapter)
        for action_index in range(450):
            pre = adapter.privileged_state()
            raw = adapter.render("wrist_camera")
            requested = policy.predict(raw, pre.current_act, adapter)
            command = adapter.apply_policy_command(requested)
            substeps = adapter.advance_control_period()
            post = adapter.privileged_state()
            measurement, diagnostics = adapter.pick_place_measurement(
                command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m
            )
            state, evaluation = evaluate_pick_place_step(contract, measurement, state)
            fresh.append({
                "pre": {name: getattr(pre, name).tolist() for name in pre.__dataclass_fields__},
                "requested_act": command.requested_act.tolist(),
                "executed_act": command.executed_act.tolist(),
                "post": {name: getattr(post, name).tolist() for name in post.__dataclass_fields__},
                "physics_substeps": substeps,
                "pickup_measurement": asdict(measurement.pickup),
                "placement_measurement": {
                    "cube_footprint_inside": measurement.cube_footprint_inside,
                    "cube_support_error_m": measurement.cube_support_error_m,
                    "cube_linear_speed_m_s": measurement.cube_linear_speed_m_s,
                    "cube_angular_speed_rad_s": measurement.cube_angular_speed_rad_s,
                    "gripper_act": measurement.gripper_act,
                },
                "contact": diagnostics,
                "pickup_events": [event.value for event in evaluation.pickup_events],
                "placement_events": [event.value for event in evaluation.events],
            })
    finally:
        adapter.close()

    fresh_current = np.asarray([row["pre"]["current_act"] for row in fresh], dtype=np.float32)
    fresh_requested = np.asarray([row["requested_act"] for row in fresh], dtype=np.float32)
    fresh_executed = np.asarray([row["executed_act"] for row in fresh], dtype=np.float32)
    fresh_cube = np.asarray([row["pre"]["cube_position"] for row in fresh], dtype=np.float32)
    fresh_quaternion = np.asarray([row["pre"]["cube_quaternion_wxyz"] for row in fresh], dtype=np.float32)
    fresh_qvel = np.asarray([row["pre"]["robot_qvel"] for row in fresh], dtype=np.float32)
    fresh_post_cube = np.asarray([row["post"]["cube_position"] for row in fresh], dtype=np.float32)

    stored_current = np.asarray([row["current_act"] for row in stored["rows"]], dtype=np.float32)
    stored_requested = np.asarray([row["requested_act"] for row in stored["rows"]], dtype=np.float32)
    stored_executed = np.asarray([row["executed_act"] for row in stored["rows"]], dtype=np.float32)
    stored_post_cube = np.asarray([row["cube_position"] for row in stored["rows"]], dtype=np.float32)
    reproduction = {
        "maximum_current_act_difference": float(np.max(np.abs(fresh_current - stored_current))),
        "maximum_requested_act_difference": float(np.max(np.abs(fresh_requested - stored_requested))),
        "maximum_executed_act_difference": float(np.max(np.abs(fresh_executed - stored_executed))),
        "maximum_post_cube_position_difference_m": float(np.max(np.abs(fresh_post_cube - stored_post_cube))),
        "events_identical": all(
            new["pickup_events"] == old["pickup_events"]
            and new["placement_events"] == old["placement_events"]
            for new, old in zip(fresh, stored["rows"], strict=True)
        ),
    }
    if any(reproduction[key] > 1e-7 for key in reproduction if key.startswith("maximum_")) or not reproduction["events_identical"]:
        raise RuntimeError(f"fresh rollout did not reproduce stored failure: {reproduction}")

    checkpoint = policy
    oracle_features = np.concatenate((
        normalize_act(oracle["current_act"]),
        (oracle["cube_position"] - checkpoint.extras_mean) / checkpoint.extras_std,
        oracle["progress"].reshape(-1, 1),
    ), axis=1).astype(np.float32)
    clone_features = np.concatenate((
        normalize_act(fresh_current),
        (fresh_cube - checkpoint.extras_mean) / checkpoint.extras_std,
        oracle["progress"].reshape(-1, 1),
    ), axis=1).astype(np.float32)
    feature_distances = np.linalg.norm(clone_features[:, None] - oracle_features[None, :], axis=2)
    nearest_index = np.argmin(feature_distances, axis=1)
    nearest_distance = feature_distances[np.arange(450), nearest_index]

    action_max = np.max(np.abs(fresh_requested - oracle["requested_act"]), axis=1)
    current_max = np.max(np.abs(fresh_current - oracle["current_act"]), axis=1)
    qvel_l2 = np.linalg.norm(fresh_qvel - oracle["robot_qvel"], axis=1)
    cube_position_l2 = np.linalg.norm(fresh_cube - oracle["cube_position"], axis=1)
    cube_quaternion = np.asarray([
        norm_quaternion_distance(left, right)
        for left, right in zip(fresh_quaternion, oracle["cube_quaternion_wxyz"], strict=True)
    ])
    post_cube_position_l2 = np.linalg.norm(fresh_post_cube - oracle["post_cube_position"], axis=1)

    aligned_rows: list[dict[str, object]] = []
    for index, row in enumerate(fresh):
        stage_index = next(
            stage for stage, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            if start <= index < stop
        )
        aligned_rows.append({
            "action_index_zero_based": index,
            "action_number_one_based": index + 1,
            "controller_stage_index": stage_index,
            "controller_stage": STAGES[stage_index],
            "teacher": {
                "current_act": oracle["current_act"][index].tolist(),
                "robot_qvel": oracle["robot_qvel"][index].tolist(),
                "cube_position": oracle["cube_position"][index].tolist(),
                "cube_quaternion_wxyz": oracle["cube_quaternion_wxyz"][index].tolist(),
                "cube_linear_velocity": oracle["cube_linear_velocity"][index].tolist(),
                "cube_angular_velocity": oracle["cube_angular_velocity"][index].tolist(),
                "requested_act": oracle["requested_act"][index].tolist(),
                "executed_act": oracle["executed_act"][index].tolist(),
                "post_cube_position": oracle["post_cube_position"][index].tolist(),
                "post_any_contact": bool(oracle["post_any_contact"][index]),
                "post_bilateral_interior_contact": bool(oracle["post_bilateral_interior_contact"][index]),
                "post_strict_bilateral_grasp": bool(oracle["post_strict_bilateral_grasp"][index]),
                "post_jaw_cube_distance_m": float(oracle["post_jaw_cube_distance_m"][index]),
                "post_cube_height_gain_m": float(oracle["post_cube_height_gain_m"][index]),
                "pickup_events": teacher_events[index]["pickup_events"],
                "placement_events": teacher_events[index]["placement_events"],
            },
            "clone": row,
            "differences": {
                "requested_act_max_abs": float(action_max[index]),
                "current_act_max_abs": float(current_max[index]),
                "robot_qvel_l2": float(qvel_l2[index]),
                "cube_position_l2_m": float(cube_position_l2[index]),
                "cube_quaternion_chord_distance": float(cube_quaternion[index]),
                "post_cube_position_l2_m": float(post_cube_position_l2[index]),
                "same_index_feature_l2": float(np.linalg.norm(clone_features[index] - oracle_features[index])),
                "nearest_training_feature_l2": float(nearest_distance[index]),
                "nearest_training_action_index": int(nearest_index[index]),
            },
        })

    ALIGNED.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in aligned_rows),
        encoding="utf-8",
    )

    teacher_contact = np.flatnonzero(oracle["post_any_contact"])
    clone_contact = np.flatnonzero([row["pickup_measurement"]["any_contact"] for row in fresh])
    teacher_bilateral = np.flatnonzero(oracle["post_bilateral_interior_contact"])
    clone_bilateral = np.flatnonzero([row["pickup_measurement"]["bilateral_interior_contact"] for row in fresh])
    report = {
        "schema_version": 1,
        "claim": "diagnostic_alignment_only_no_training_or_policy_change",
        "inputs": {
            "manifest": {"path": str(MANIFEST.relative_to(REPOSITORY)), "sha256": sha256(MANIFEST)},
            "oracle_arrays": {"path": str((MANIFEST.parent / manifest["arrays"]["path"]).relative_to(REPOSITORY)), "sha256": sha256(MANIFEST.parent / manifest["arrays"]["path"])},
            "checkpoint": {"path": str(CHECKPOINT.relative_to(REPOSITORY)), "sha256": sha256(CHECKPOINT)},
            "stored_failed_telemetry": {"path": str(STORED_TELEMETRY.relative_to(REPOSITORY)), "sha256": sha256(STORED_TELEMETRY)},
            "mujoco_model": {"path": str(MODEL.relative_to(REPOSITORY)), "sha256": sha256(MODEL)},
        },
        "alignment": "zero-based oracle action_index i equals stored one-based telemetry action i+1; all states are pre-action unless prefixed post_",
        "fresh_rollout_reproduction": reproduction,
        "threshold_crossings_zero_based": {
            "requested_action_max_abs_gt_0.005": first_over(action_max, 0.005),
            "requested_action_max_abs_gt_offline_max_error_gate_0.01": first_over(action_max, 0.01),
            "current_act_max_abs_gt_0.005": first_over(current_max, 0.005),
            "current_act_max_abs_gt_0.01": first_over(current_max, 0.01),
            "cube_position_l2_gt_0.0001_m": first_over(cube_position_l2, 0.0001),
            "nearest_training_feature_l2_gt_0.01": first_over(nearest_distance, 0.01),
        },
        "earliest_supported_divergences": {
            "first_nonzero_policy_action": {
                "action_index": 0,
                "requested_action_max_abs": float(action_max[0]),
            },
            "first_discrete_task_event_mismatch": {
                "action_index": 111,
                "teacher_pickup_events": teacher_events[111]["pickup_events"],
                "clone_pickup_events": fresh[111]["pickup_events"],
                "note": "clone crosses the reach threshold six actions before the teacher (teacher reach is action 117)",
            },
            "first_cube_interaction_split": {
                "action_index": 194,
                "stage": "engaged_to_seated",
                "both_report_any_contact": True,
                "post_cube_position_l2_m": float(post_cube_position_l2[194]),
                "next_pre_cube_position_l2_m": float(cube_position_l2[195]),
                "note": "the cube trajectories are identical before contact; they split at the shared first-contact action",
            },
        },
        "contact_outcome": {
            "teacher_any_contact": {"first": int(teacher_contact[0]), "last": int(teacher_contact[-1]), "frames": int(len(teacher_contact))},
            "clone_any_contact": {"first": int(clone_contact[0]), "last": int(clone_contact[-1]), "frames": int(len(clone_contact))},
            "teacher_bilateral_contact": {"first": int(teacher_bilateral[0]), "last": int(teacher_bilateral[-1]), "frames": int(len(teacher_bilateral))},
            "clone_bilateral_contact_frames": int(len(clone_bilateral)),
            "teacher_maximum_cube_height_gain_m": float(np.max(oracle["post_cube_height_gain_m"])),
            "clone_maximum_cube_height_gain_m": float(max(row["pickup_measurement"]["cube_height_gain_m"] for row in fresh)),
        },
        "aligned_timesteps": {"path": ALIGNED.name, "rows": len(aligned_rows), "sha256": sha256(ALIGNED)},
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
