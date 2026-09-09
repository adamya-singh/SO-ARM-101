"""Content-addressed demonstrations from the full privileged controller."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

from so_arm101_v2.contracts import (
    PickPlaceEvaluationState,
    evaluate_pick_place_step,
    load_pick_place_contract,
)
from so_arm101_v2.data._serialization import (
    content_sha256,
    write_immutable_bytes,
    write_immutable_file,
    write_immutable_json,
)
from so_arm101_v2.data.resources import RESOURCE_NAMES, read_resource_bytes

from .adapter import MujocoTaskAdapter
from .parallel import resolve_workers
from .privileged import PrivilegedStagedController
from .rollout import _VideoWriter
from .suites import SimulationScenario, SimulationSuite, load_simulation_suite, suite_payload


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
        or len(rollouts) != len(suite.scenarios) * suite.repeats
        or not all(item.get("success") for item in rollouts)
    ):
        raise ValueError(
            "capture requires a passing full-coverage preflight report "
            f"({len(suite.scenarios) * suite.repeats} rollouts)"
        )
    return payload


def _select_scenarios(suite: SimulationSuite, selection: str) -> tuple[SimulationScenario, ...]:
    if selection == "all":
        return suite.scenarios
    matches = tuple(item for item in suite.scenarios if item.scenario_id == selection)
    if len(matches) != 1:
        raise ValueError(f"unknown scenario {selection!r} for suite {suite.suite_id}")
    return matches


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    write_immutable_bytes(
        path, data, conflict_message=f"immutable oracle artifact differs: {path}"
    )


_ORACLE_FIELDS = (
    "scenario_index", "action_index", "progress", "current_act", "robot_qvel",
    "cube_position", "cube_quaternion_wxyz", "cube_linear_velocity",
    "cube_angular_velocity", "requested_act", "executed_act", "executed_delta_act",
    "post_cube_position", "post_cube_footprint_inside", "post_cube_support_error_m",
    "post_cube_linear_speed_m_s", "post_cube_angular_speed_rad_s", "post_gripper_act",
    "post_jaw_cube_distance_m", "post_any_contact", "post_bilateral_interior_contact",
    "post_strict_bilateral_grasp", "post_cube_height_gain_m", "post_unsafe_contact",
    "post_command_bound_violation", "post_delta_limiter_activated", "post_nonfinite_command",
)


@dataclass(frozen=True)
class _OracleScenarioTask:
    model_path: str
    scenario: SimulationScenario
    scenario_index: int
    teacher_horizon: int
    contract: Any
    temporary_dir: str
    record_video: bool
    frames_path: str | None
    skip_failed_scenarios: bool


@dataclass(frozen=True)
class _OracleScenarioResult:
    scenario_id: str
    error: str | None
    columns: dict[str, list[Any]]
    events: list[dict[str, Any]]
    boundaries: tuple[int, ...]
    solve_diagnostics: list[dict[str, Any]]
    safety_counts: dict[str, int]
    final_evaluation: dict[str, Any] | None
    video_names: tuple[str, ...]


def _capture_scenario(task: _OracleScenarioTask) -> _OracleScenarioResult:
    """One deterministic teacher episode; frames go to this scenario's own memmap slot.

    Exactly the sequential per-scenario body: a fresh adapter and controller and
    the same failure rules. Nothing carries across scenarios, which is what makes
    the process-parallel fan-out byte-identical.
    """
    item = task.scenario
    contract = task.contract
    horizon = task.teacher_horizon
    temporary_path = Path(task.temporary_dir)
    adapter = MujocoTaskAdapter(task.model_path)
    controller = PrivilegedStagedController()
    wrist_temp = temporary_path / f"{item.scenario_id}.wrist.mp4" if task.record_video else None
    overview_temp = temporary_path / f"{item.scenario_id}.overview.mp4" if task.record_video else None
    wrist = _VideoWriter(wrist_temp)
    overview = _VideoWriter(overview_temp)
    frames = np.load(task.frames_path, mmap_mode="r+") if task.frames_path is not None else None
    base = task.scenario_index * horizon
    state = PickPlaceEvaluationState()
    columns: dict[str, list[Any]] = {name: [] for name in _ORACLE_FIELDS}
    event_rows: list[dict[str, Any]] = []
    evaluation = None
    safety_counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    episode_error: str | None = None
    try:
        adapter.reset(item)
        controller.reset(adapter)
        for action_index in range(horizon):
            snapshot = adapter.privileged_state()
            raw = adapter.render_wrist_observation()
            if frames is not None:
                frames[base + action_index] = raw
            wrist.add(raw)
            if overview_temp is not None:
                # The overview render only feeds the video writer; rendering never
                # touches physics, so skipping it without video is digest-neutral.
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
                "scenario_index": task.scenario_index,
                "action_index": action_index,
                "progress": action_index / (horizon - 1),
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
                columns[name].append(value)
            event_rows.append({
                "action_index": action_index,
                "pickup_events": pickup_events,
                "placement_events": placement_events,
            })
    except RuntimeError as exc:
        if not task.skip_failed_scenarios:
            raise
        episode_error = str(exc)
    finally:
        wrist.close()
        overview.close()
        adapter.close()
        if frames is not None:
            frames.flush()
            del frames
    if episode_error is None and (
        evaluation is None
        or not evaluation.success
        or len(event_rows) != horizon
    ):
        episode_error = (
            f"oracle scenario {item.scenario_id} did not pass v3 within "
            f"{horizon} actions"
        )
        if not task.skip_failed_scenarios:
            raise RuntimeError(episode_error)
    if episode_error is None and any(safety_counts.values()):
        episode_error = f"oracle scenario {item.scenario_id} used the safety layer"
        if not task.skip_failed_scenarios:
            raise RuntimeError(episode_error)
    if episode_error is not None:
        return _OracleScenarioResult(
            scenario_id=item.scenario_id, error=episode_error, columns={name: [] for name in _ORACLE_FIELDS},
            events=[], boundaries=(), solve_diagnostics=[], safety_counts=safety_counts, final_evaluation=None, video_names=(),
        )
    video_names = (wrist_temp.name, overview_temp.name) if wrist_temp is not None and overview_temp is not None else ()
    return _OracleScenarioResult(
        scenario_id=item.scenario_id, error=None, columns=columns, events=event_rows,
        boundaries=tuple(controller.boundaries), solve_diagnostics=list(controller.solve_diagnostics),
        safety_counts=safety_counts, final_evaluation=asdict(evaluation), video_names=video_names,
    )


def _execute_capture_tasks(tasks: list[_OracleScenarioTask], *, workers: int) -> list[_OracleScenarioResult]:
    """Run scenario captures in-process or on a spawn pool; results in submission order, progress to stderr."""
    def report(index: int, result: _OracleScenarioResult) -> None:
        status = "kept" if result.error is None else f"skipped ({result.error})"
        print(f"oracle capture {index + 1}/{len(tasks)}: {result.scenario_id} {status}", file=sys.stderr, flush=True)

    if workers <= 1 or len(tasks) <= 1:
        results = []
        for index, task in enumerate(tasks):
            result = _capture_scenario(task)
            report(index, result)
            results.append(result)
        return results
    pool = ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        futures = [pool.submit(_capture_scenario, task) for task in tasks]
        results = []
        for index, future in enumerate(futures):
            result = future.result()
            report(index, result)
            results.append(result)
        return results
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def capture_oracle_demonstrations(
    model_path: str | Path,
    suite: SimulationSuite | str,
    preflight_report: str | Path,
    output_dir: str | Path,
    *,
    scenario: str = "nominal",
    record_video: bool = True,
    teacher_horizon: int = 450,
    store_frames: bool = False,
    skip_failed_scenarios: bool = False,
    workers: int | None = None,
) -> OracleDemonstrationCollection:
    """Capture one deterministic full-horizon teacher episode per scenario.

    ``workers`` fans scenarios out over spawn-context processes; every published
    byte (arrays, frames, manifest, collection digest) is identical to the
    sequential path because each scenario is an independent deterministic
    episode, results are assembled in scenario order, and frames are written to
    disjoint slots of the same memmap. ``None`` = auto (capture always holds an
    EGL context per worker, so the video cap applies).
    """
    suite = load_simulation_suite(suite) if isinstance(suite, str) else suite
    if suite.task_contract not in ("fixed_cube_pick_place_v3", "bench_pick_replace_v1"):
        raise ValueError("oracle demonstrations require the v3 pick-place suite")
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    bench = scene_bench_config(model_path)
    contract = load_pick_place_contract(suite.task_contract, bench_config=bench)
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
        "suite_resource_sha256": (
            hashlib.sha256(read_resource_bytes(f"{suite.suite_id}.json")).hexdigest()
            if f"{suite.suite_id}.json" in RESOURCE_NAMES
            # Generated suites are not packaged resources; hash their
            # canonical payload instead (same collision-resistance role).
            else content_sha256(suite_payload(suite))
        ),
        "task_resource_sha256": hashlib.sha256(
            (json.dumps(asdict(contract), sort_keys=True, default=str).encode() if bench else read_resource_bytes(f"{suite.task_contract}.json"))
        ).hexdigest(),
        "coordinate_contract_sha256": hashlib.sha256(
            read_resource_bytes("act_coordinate_contract.json")
        ).hexdigest(),
        "preflight_content_sha256": preflight["content_sha256"],
        "git": provenance,
        "controller": controller_config,
        # Keyword-parameterized since the horizon-alignment tranche; the
        # default keeps every legacy capture identity (and hence collection
        # digest) byte-identical.
        "teacher_horizon": teacher_horizon,
    }
    if bench is not None:
        from .contact import GRASP_DETECTOR_VERSION
        identity["bench_config"] = asdict(bench)
        identity["scene_dependencies_sha256"] = scene_dependency_hash(model_path)
        identity["grasp_detector"] = GRASP_DETECTOR_VERSION
        identity["joint_map"] = bench.joint_map_object.provenance()
        if bench.lens is not None:
            identity["lens"] = dict(bench.lens)
    if store_frames:
        # Conditionally-present so every legacy capture identity (and hence
        # collection digest) stays byte-identical when frames are off.
        identity["frame_store"] = {
            "format": "npy_memmap_uint8_v1",
            "frame_shape": [256, 256, 3],
        }

    arrays: dict[str, list[np.ndarray | float | int | bool]] = {name: [] for name in _ORACLE_FIELDS}
    episode_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    horizon = int(identity["teacher_horizon"])
    worker_count = resolve_workers(workers, record_video=True, task_count=len(selected))

    frames_sha256: str | None = None
    with tempfile.TemporaryDirectory(prefix="so_arm101_oracle_") as temporary:
        temporary_path = Path(temporary)
        staged_videos: list[tuple[Path, str]] = []
        frames = None
        frames_temp = temporary_path / "images.npy"
        if store_frames:
            total_rows = len(selected) * horizon
            frames = np.lib.format.open_memmap(
                frames_temp, mode="w+", dtype=np.uint8,
                shape=(total_rows, 256, 256, 3),
            )
            frames.flush()
        tasks = [
            _OracleScenarioTask(
                model_path=str(model_path), scenario=item, scenario_index=index, teacher_horizon=horizon,
                contract=contract, temporary_dir=str(temporary_path), record_video=record_video,
                frames_path=(str(frames_temp) if store_frames else None), skip_failed_scenarios=skip_failed_scenarios,
            )
            for index, item in enumerate(selected)
        ]
        results = _execute_capture_tasks(tasks, workers=worker_count)
        kept = 0
        for index, (item, result) in enumerate(zip(selected, results)):
            if result.error is not None:
                skipped_records.append({"scenario_id": item.scenario_id, "reason": result.error})
                continue
            if frames is not None and kept != index:
                # Compact kept episodes forward over skipped slots (ascending, so the
                # destination slot is always free): identical bytes to the sequential
                # running-row layout.
                frames[kept * horizon:(kept + 1) * horizon] = frames[index * horizon:(index + 1) * horizon]
            for name in _ORACLE_FIELDS:
                arrays[name].extend(result.columns[name])
            episode_records.append({
                "scenario_id": item.scenario_id,
                "rows": horizon,
                "waypoint_boundaries": list(result.boundaries),
                "solve_diagnostics": list(result.solve_diagnostics),
                "safety_counts": dict(result.safety_counts),
                "events": list(result.events),
                "final_evaluation": result.final_evaluation,
            })
            for name in result.video_names:
                staged_videos.append((temporary_path / name, name))
            kept += 1

        materialized = {name: np.asarray(values) for name, values in arrays.items()}
        if not all(np.all(np.isfinite(value)) for value in materialized.values()):
            raise RuntimeError("oracle arrays contain nonfinite values")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **materialized)
        arrays_bytes = buffer.getvalue()
        arrays_sha256 = hashlib.sha256(arrays_bytes).hexdigest()
        digest_inputs: dict[str, Any] = {**identity, "arrays_sha256": arrays_sha256}
        if frames is not None:
            frames.flush()
            kept_rows = int(materialized["action_index"].shape[0])
            if kept_rows < frames.shape[0]:
                truncated_path = temporary_path / "images.trunc.npy"
                np.save(truncated_path, np.asarray(frames[:kept_rows]))
                frames_temp = truncated_path
            digest = hashlib.sha256()
            with open(frames_temp, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            frames_sha256 = digest.hexdigest()
            digest_inputs["frames_sha256"] = frames_sha256
        collection_digest = content_sha256(digest_inputs)
        destination = Path(output_dir) / "oracle" / suite.suite_id / collection_digest[:16]
        arrays_path = destination / "demonstrations.npz"
        _write_immutable_bytes(arrays_path, arrays_bytes)
        if frames is not None:
            write_immutable_file(
                destination / "images.npy", frames_temp,
                conflict_message=f"immutable oracle artifact differs: {destination / 'images.npy'}",
            )

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
    if skipped_records:
        manifest_payload["skipped_scenarios"] = skipped_records
    if frames_sha256 is not None:
        manifest_payload["frames"] = {
            "path": "images.npy",
            "sha256": frames_sha256,
            "rows": int(materialized["action_index"].shape[0]),
            "dtype": "uint8",
            "frame_shape": [256, 256, 3],
            "convention": ("raw_wrist_hwc_uint8_preprocess_with_preprocess_wrist_image" if "lens" not in identity
                           else f"raw_wrist_hwc_uint8_lens_{identity['lens']['model']}_{identity['lens']['framing']}_preprocess_with_preprocess_wrist_image"),
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
