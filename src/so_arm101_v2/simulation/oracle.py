"""Content-addressed demonstrations from the full privileged controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np

from so_arm101_v2.contracts import (
    PickPlaceEvaluationState,
    evaluate_pick_place_step,
    load_pick_place_contract,
)
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.data.resources import read_resource_bytes

from .adapter import MujocoTaskAdapter
from .privileged import PrivilegedStagedController
from .rollout import _VideoWriter
from .suites import SimulationScenario, SimulationSuite, load_simulation_suite


@dataclass(frozen=True)
class OracleDemonstrationCollection:
    directory: Path
    manifest: Path
    arrays: Path
    collection_digest: str
    scenario_ids: tuple[str, ...]
    rows: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repository: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout
    fingerprint = hashlib.sha256()
    fingerprint.update(subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repository, check=True,
        capture_output=True,
    ).stdout)
    for line in sorted(status.splitlines()):
        fingerprint.update(line.encode("utf-8"))
        if line.startswith("?? "):
            candidate = repository / line[3:]
            if candidate.is_file():
                fingerprint.update(candidate.read_bytes())
    dirty = bool(status.strip())
    return {
        "revision": revision,
        "dirty": dirty,
        "dirty_fingerprint": fingerprint.hexdigest() if dirty else None,
    }


def _validated_preflight(path: Path, suite: SimulationSuite) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("preflight report content hash mismatch")
    rollouts = payload.get("rollouts", [])
    if (
        payload.get("environment_proven") is not True
        or payload.get("deterministic") is not True
        or payload.get("suite", {}).get("suite_id") != suite.suite_id
        or len(rollouts) != 15
        or not all(item.get("success") for item in rollouts)
    ):
        raise ValueError("capture requires a passing 15/15 v3 preflight report")
    return payload


def _select_scenarios(suite: SimulationSuite, selection: str) -> tuple[SimulationScenario, ...]:
    if selection == "all":
        return suite.scenarios
    matches = tuple(item for item in suite.scenarios if item.scenario_id == selection)
    if len(matches) != 1:
        raise ValueError(f"unknown scenario {selection!r} for suite {suite.suite_id}")
    return matches


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable oracle artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def capture_oracle_demonstrations(
    model_path: str | Path,
    suite: SimulationSuite | str,
    preflight_report: str | Path,
    output_dir: str | Path,
    *,
    scenario: str = "nominal",
    record_video: bool = True,
) -> OracleDemonstrationCollection:
    """Capture one deterministic full-horizon teacher episode per scenario."""
    suite = load_simulation_suite(suite) if isinstance(suite, str) else suite
    if suite.task_contract != "fixed_cube_pick_place_v3":
        raise ValueError("oracle demonstrations require the v3 pick-place suite")
    contract = load_pick_place_contract(suite.task_contract)
    preflight = _validated_preflight(Path(preflight_report), suite)
    selected = _select_scenarios(suite, scenario)
    model_path = Path(model_path).resolve()
    repository = Path(__file__).resolve().parents[3]
    provenance = _git_provenance(repository)

    controller_config = asdict(PrivilegedStagedController())
    identity = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "scenario_ids": [item.scenario_id for item in selected],
        "model_sha256": _sha256_file(model_path),
        "suite_resource_sha256": hashlib.sha256(
            read_resource_bytes(f"{suite.suite_id}.json")
        ).hexdigest(),
        "task_resource_sha256": hashlib.sha256(
            read_resource_bytes(f"{suite.task_contract}.json")
        ).hexdigest(),
        "coordinate_contract_sha256": hashlib.sha256(
            read_resource_bytes("act_coordinate_contract.json")
        ).hexdigest(),
        "preflight_content_sha256": preflight["content_sha256"],
        "git": provenance,
        "controller": controller_config,
        "teacher_horizon": 450,
    }

    arrays: dict[str, list[np.ndarray | float | int | bool]] = {
        name: [] for name in (
            "scenario_index", "action_index", "progress", "current_act", "robot_qvel",
            "cube_position", "cube_quaternion_wxyz", "cube_linear_velocity",
            "cube_angular_velocity", "requested_act", "executed_act", "executed_delta_act",
            "post_cube_position", "post_cube_footprint_inside", "post_cube_support_error_m",
            "post_cube_linear_speed_m_s", "post_cube_angular_speed_rad_s", "post_gripper_act",
            "post_jaw_cube_distance_m", "post_any_contact", "post_bilateral_interior_contact",
            "post_strict_bilateral_grasp", "post_cube_height_gain_m", "post_unsafe_contact",
            "post_command_bound_violation", "post_delta_limiter_activated", "post_nonfinite_command",
        )
    }
    episode_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="so_arm101_oracle_") as temporary:
        temporary_path = Path(temporary)
        staged_videos: list[tuple[Path, str]] = []
        for scenario_index, item in enumerate(selected):
            adapter = MujocoTaskAdapter(model_path)
            controller = PrivilegedStagedController()
            wrist_temp = temporary_path / f"{item.scenario_id}.wrist.mp4" if record_video else None
            overview_temp = temporary_path / f"{item.scenario_id}.overview.mp4" if record_video else None
            wrist = _VideoWriter(wrist_temp)
            overview = _VideoWriter(overview_temp)
            state = PickPlaceEvaluationState()
            event_rows: list[dict[str, Any]] = []
            evaluation = None
            safety_counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
            try:
                adapter.reset(item)
                controller.reset(adapter)
                for action_index in range(identity["teacher_horizon"]):
                    snapshot = adapter.privileged_state()
                    raw = adapter.render("wrist_camera")
                    wrist.add(raw)
                    overview.add(adapter.render("camera_side"))
                    requested = controller.predict(raw, snapshot.current_act, adapter)
                    command = adapter.apply_policy_command(requested)
                    adapter.advance_control_period()
                    measurement, _ = adapter.pick_place_measurement(
                        command,
                        footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
                    )
                    if not state.completed:
                        state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                        pickup_events = [event.value for event in evaluation.pickup_events]
                        placement_events = [event.value for event in evaluation.events]
                    else:
                        pickup_events = []
                        placement_events = []

                    if float(np.max(np.abs(command.requested_act - command.executed_act))) > 1e-6:
                        raise RuntimeError("oracle requested and executed commands diverged")
                    safety_counts["clip"] += int(measurement.pickup.command_bound_violation)
                    safety_counts["limit"] += int(measurement.pickup.delta_limiter_activated)
                    safety_counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
                    safety_counts["unsafe"] += int(measurement.pickup.unsafe_contact)

                    values: dict[str, Any] = {
                        "scenario_index": scenario_index,
                        "action_index": action_index,
                        "progress": action_index / (identity["teacher_horizon"] - 1),
                        "current_act": snapshot.current_act,
                        "robot_qvel": snapshot.robot_qvel,
                        "cube_position": snapshot.cube_position,
                        "cube_quaternion_wxyz": snapshot.cube_quaternion_wxyz,
                        "cube_linear_velocity": snapshot.cube_linear_velocity,
                        "cube_angular_velocity": snapshot.cube_angular_velocity,
                        "requested_act": command.requested_act,
                        "executed_act": command.executed_act,
                        "executed_delta_act": command.executed_act - snapshot.current_act,
                        "post_cube_position": adapter.data.body("red_block").xpos.copy(),
                        "post_cube_footprint_inside": measurement.cube_footprint_inside,
                        "post_cube_support_error_m": measurement.cube_support_error_m,
                        "post_cube_linear_speed_m_s": measurement.cube_linear_speed_m_s,
                        "post_cube_angular_speed_rad_s": measurement.cube_angular_speed_rad_s,
                        "post_gripper_act": measurement.gripper_act,
                        "post_jaw_cube_distance_m": measurement.pickup.jaw_cube_distance_m,
                        "post_any_contact": measurement.pickup.any_contact,
                        "post_bilateral_interior_contact": measurement.pickup.bilateral_interior_contact,
                        "post_strict_bilateral_grasp": measurement.pickup.strict_bilateral_grasp,
                        "post_cube_height_gain_m": measurement.pickup.cube_height_gain_m,
                        "post_unsafe_contact": measurement.pickup.unsafe_contact,
                        "post_command_bound_violation": measurement.pickup.command_bound_violation,
                        "post_delta_limiter_activated": measurement.pickup.delta_limiter_activated,
                        "post_nonfinite_command": measurement.pickup.nonfinite_command,
                    }
                    for name, value in values.items():
                        arrays[name].append(value)
                    event_rows.append({
                        "action_index": action_index,
                        "pickup_events": pickup_events,
                        "placement_events": placement_events,
                    })
            finally:
                wrist.close()
                overview.close()
                adapter.close()
            if evaluation is None or not evaluation.success or len(event_rows) != 450:
                raise RuntimeError(f"oracle scenario {item.scenario_id} did not pass v3 within 450 actions")
            if any(safety_counts.values()):
                raise RuntimeError(f"oracle scenario {item.scenario_id} used the safety layer")
            episode_records.append({
                "scenario_id": item.scenario_id,
                "rows": identity["teacher_horizon"],
                "waypoint_boundaries": list(controller.boundaries),
                "solve_diagnostics": controller.solve_diagnostics,
                "safety_counts": safety_counts,
                "events": event_rows,
                "final_evaluation": asdict(evaluation),
            })
            if wrist_temp is not None and overview_temp is not None:
                staged_videos.extend(((wrist_temp, wrist_temp.name), (overview_temp, overview_temp.name)))

        materialized = {name: np.asarray(values) for name, values in arrays.items()}
        if not all(np.all(np.isfinite(value)) for value in materialized.values()):
            raise RuntimeError("oracle arrays contain nonfinite values")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **materialized)
        arrays_bytes = buffer.getvalue()
        arrays_sha256 = hashlib.sha256(arrays_bytes).hexdigest()
        collection_digest = content_sha256({**identity, "arrays_sha256": arrays_sha256})
        destination = Path(output_dir) / "oracle" / suite.suite_id / collection_digest[:16]
        arrays_path = destination / "demonstrations.npz"
        _write_immutable_bytes(arrays_path, arrays_bytes)

        videos: list[dict[str, Any]] = []
        for temporary_video, name in staged_videos:
            video_bytes = temporary_video.read_bytes()
            destination_video = destination / "videos" / name
            _write_immutable_bytes(destination_video, video_bytes)
            videos.append({
                "path": str(Path("videos") / name),
                "sha256": hashlib.sha256(video_bytes).hexdigest(),
            })

    manifest_payload: dict[str, Any] = {
        **identity,
        "collection_digest": collection_digest,
        "arrays": {
            "path": "demonstrations.npz",
            "sha256": arrays_sha256,
            "rows": int(materialized["action_index"].shape[0]),
            "fields": {name: list(value.shape) for name, value in materialized.items()},
        },
        "feature_schemas": {
            "phase_state": ["current_act[6]", "cube_position[3]", "progress[1]"],
            "feedback_state": [
                "current_act[6]", "robot_qvel[6]", "cube_position[3]",
                "cube_quaternion_wxyz[4]", "cube_linear_velocity[3]",
                "cube_angular_velocity[3]",
            ],
        },
        "episodes": episode_records,
        "videos": videos,
        "claim": "privileged_simulation_demonstrations_only_not_deployment_data",
    }
    manifest_payload["content_sha256"] = content_sha256(manifest_payload)
    manifest_path = destination / "manifest.json"
    write_immutable_json(manifest_path, manifest_payload)
    return OracleDemonstrationCollection(
        directory=destination,
        manifest=manifest_path,
        arrays=arrays_path,
        collection_digest=collection_digest,
        scenario_ids=tuple(item.scenario_id for item in selected),
        rows=int(materialized["action_index"].shape[0]),
    )


def load_oracle_demonstrations(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and hash-validate an oracle collection."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stated = manifest.get("content_sha256")
    body = dict(manifest)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("oracle manifest content hash mismatch")
    arrays_path = path.parent / manifest["arrays"]["path"]
    if _sha256_file(arrays_path) != manifest["arrays"]["sha256"]:
        raise ValueError("oracle array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if arrays["action_index"].shape[0] != int(manifest["arrays"]["rows"]):
        raise ValueError("oracle row count mismatch")
    return manifest, arrays


__all__ = [
    "OracleDemonstrationCollection",
    "capture_oracle_demonstrations",
    "load_oracle_demonstrations",
]
