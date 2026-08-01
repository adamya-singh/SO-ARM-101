"""Named, reward-independent closed-loop MuJoCo evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from so_arm101_v2.contracts import (
    DiagnosticEvent,
    TaskEvaluationState,
    evaluate_task_step,
    load_task_contract,
)
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

from .adapter import MujocoTaskAdapter
from .privileged import PrivilegedStagedController
from .suites import SimulationScenario, SimulationSuite, load_simulation_suite


class SimulationPolicy(Protocol):
    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: MujocoTaskAdapter | None = None) -> np.ndarray: ...


@dataclass(frozen=True)
class RolloutMetrics:
    policy_id: str
    suite_id: str
    scenario_id: str
    repeat: int
    actions: int
    success: bool
    invalidated: bool
    timed_out: bool
    failure_category: str
    reached: bool
    contacted: bool
    bilateral_contact: bool
    strict_grasp_acquired: bool
    grasp_losses: int
    drops: int
    maximum_height_gain_m: float
    clipping_frames: int
    limiting_frames: int
    nonfinite_frames: int
    unsafe_contact_frames: int
    recovery_opportunity: bool
    recovered: bool | None
    recovery_actions: int | None
    final_robot_qpos: tuple[float, ...]
    final_cube_position: tuple[float, ...]
    telemetry_path: str
    wrist_video_path: str | None
    overview_video_path: str | None


@dataclass(frozen=True)
class SimulationEvaluation:
    suite: SimulationSuite
    rollouts: tuple[RolloutMetrics, ...]
    report_json: Path
    environment_proven: bool | None
    deterministic: bool


class CurrentPosePolicy:
    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: MujocoTaskAdapter | None = None) -> np.ndarray:
        del image, adapter
        return current_act.copy()


@dataclass
class ConstantPosePolicy:
    target_act: np.ndarray

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: MujocoTaskAdapter | None = None) -> np.ndarray:
        del image, current_act, adapter
        return np.asarray(self.target_act, dtype=np.float32)


class TorchCheckpointPolicy:
    def __init__(self, checkpoint: str | Path, *, black_image: bool = False, device: str = "auto") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("checkpoint rollout requires the 'learn' extra") from exc
        from so_arm101_v2.learning.full_dataset import build_small_model
        from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act

        self.torch = torch
        self.normalize_act = normalize_act
        self.denormalize_act = denormalize_act
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        self.kind = str(payload["model_kind"])
        mean = np.asarray(payload["train_mean_target_act"], dtype=np.float32)
        self.model = build_small_model(self.kind, normalize_act(mean))
        self.model.load_state_dict(payload["state_dict"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.black_image = bool(black_image)

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: MujocoTaskAdapter | None = None) -> np.ndarray:
        del adapter
        torch = self.torch
        state = torch.from_numpy(self.normalize_act(current_act)[None]).to(self.device)
        if self.black_image or self.kind == "state_only":
            image_tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=self.device)
        else:
            chw = np.transpose(image.astype(np.float32) / np.float32(255.0), (2, 0, 1))[None]
            image_tensor = torch.from_numpy(chw).to(self.device)
        with torch.inference_mode():
            output = self.model(image_tensor, state).detach().cpu().numpy()[0]
        return self.denormalize_act(output)


class _VideoWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.container = self.stream = None
        if path is None:
            return
        try:
            import av
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rollout video requires the 'data' extra") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=30)
        self.stream.width = 256
        self.stream.height = 256
        self.stream.pix_fmt = "yuv420p"

    def add(self, rgb: np.ndarray) -> None:
        if self.container is None:
            return
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.container is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _failure_category(events: set[str], success: bool, invalidated: bool, max_motion: float) -> str:
    if success:
        return "success"
    if invalidated:
        return "safety_invalidation"
    if max_motion < 0.01:
        return "no_meaningful_motion"
    if DiagnosticEvent.REACH.value not in events:
        return "no_reach"
    if DiagnosticEvent.FIRST_CONTACT.value not in events:
        return "reach_no_contact"
    if DiagnosticEvent.BILATERAL_INTERIOR_CONTACT.value not in events:
        return "push_or_nonbilateral_contact"
    if DiagnosticEvent.STRICT_GRASP_ACQUIRED.value not in events:
        return "bad_closure_timing"
    if DiagnosticEvent.GRASP_LOSS.value in events:
        return "grasp_loss"
    if DiagnosticEvent.LIFT_5MM.value not in events:
        return "grasp_no_lift"
    return "lift_not_sustained"


def _run_rollout(
    model_path: Path,
    suite: SimulationSuite,
    scenario: SimulationScenario,
    repeat: int,
    policy_id: str,
    policy_factory: Callable[[], SimulationPolicy],
    output_dir: Path,
    *,
    record_video: bool,
) -> RolloutMetrics:
    adapter = MujocoTaskAdapter(model_path)
    policy = policy_factory()
    contract = load_task_contract(suite.task_contract)
    try:
        adapter.reset(scenario)
        reset_method = getattr(policy, "reset", None)
        if callable(reset_method):
            try:
                reset_method(adapter)
            except TypeError:
                reset_method()
    except BaseException:
        adapter.close()
        raise
    stem = f"{policy_id}.{scenario.scenario_id}.repeat{repeat}"
    telemetry_path = output_dir / "telemetry" / f"{stem}.json"
    wrist_path = output_dir / "videos" / f"{stem}.wrist.mp4" if record_video else None
    overview_path = output_dir / "videos" / f"{stem}.overview.mp4" if record_video and repeat == 0 else None
    wrist_writer = _VideoWriter(wrist_path)
    overview_writer = _VideoWriter(overview_path)
    state = TaskEvaluationState()
    rows: list[dict[str, Any]] = []
    all_events: set[str] = set()
    initial_qpos = adapter.mujoco_qpos().copy()
    maximum_motion = 0.0
    maximum_height = 0.0
    counts = {"grasp_loss": 0, "drop": 0, "clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    recovery_opportunity = False
    recovered: bool | None = None
    recovery_actions: int | None = None
    probe_active = False
    probe_finished_at: int | None = None
    external_frames = 0
    recovery_streak = 0
    strict_streak = 0
    evaluation = None
    try:
        for action in range(contract.episode.max_actions):
            raw, _, current = adapter.observation()
            wrist_writer.add(raw)
            if overview_path is not None:
                overview_writer.add(adapter.render("camera_side"))
            requested = policy.predict(raw, current, adapter)
            command = adapter.apply_policy_command(requested)
            if suite.recovery_probe is not None:
                probe = suite.recovery_probe
                if not probe_active and probe_finished_at is None and strict_streak >= probe["strict_frames_before_probe"] and action <= probe["latest_trigger_action"]:
                    recovery_opportunity = True
                    probe_active = True
                if probe_active:
                    adapter.apply_external_physical_gripper_opening(probe["opening_units_per_frame"])
                    external_frames += 1
            substeps = adapter.advance_control_period()
            measurement, contact = adapter.measurement(command)
            strict_streak = strict_streak + 1 if measurement.strict_bilateral_grasp else 0
            if probe_active and (not measurement.strict_bilateral_grasp or external_frames >= suite.recovery_probe["maximum_opening_frames"]):
                probe_active = False
                probe_finished_at = action
                recovery_streak = 0
            elif probe_finished_at is not None and recovered is None:
                recovery_streak = recovery_streak + 1 if measurement.strict_bilateral_grasp else 0
                if recovery_streak >= suite.recovery_probe["recovery_strict_frames"]:
                    recovered = True
                    recovery_actions = action - probe_finished_at
                elif action - probe_finished_at >= suite.recovery_probe["recovery_window_actions"]:
                    recovered = False
            state, evaluation = evaluate_task_step(contract, measurement, state)
            event_values = [event.value for event in evaluation.events]
            all_events.update(event_values)
            counts["grasp_loss"] += int(DiagnosticEvent.GRASP_LOSS.value in event_values)
            counts["drop"] += int(DiagnosticEvent.DROP.value in event_values)
            counts["clip"] += int(measurement.command_bound_violation)
            counts["limit"] += int(measurement.delta_limiter_activated)
            counts["nonfinite"] += int(measurement.nonfinite_command)
            counts["unsafe"] += int(measurement.unsafe_contact)
            maximum_height = max(maximum_height, measurement.cube_height_gain_m)
            maximum_motion = max(maximum_motion, float(np.linalg.norm(adapter.mujoco_qpos() - initial_qpos)))
            rows.append({
                "action": action + 1, "simulation_time_s": float(adapter.data.time),
                "physics_substeps": substeps, "current_act": current.tolist(),
                "requested_act": command.requested_act.tolist(), "executed_act": command.executed_act.tolist(),
                "robot_qpos": adapter.mujoco_qpos().tolist(),
                "cube_position": adapter.data.body("red_block").xpos.tolist(),
                "measurement": asdict(measurement), "contact": contact, "events": event_values,
                "external_recovery_probe": bool(probe_active),
            })
            if evaluation.terminated or evaluation.truncated:
                break
    finally:
        wrist_writer.close()
        overview_writer.close()
        adapter.close()
    if evaluation is None:
        raise RuntimeError("rollout produced no task evaluation")
    if recovery_opportunity and recovered is None and probe_finished_at is not None:
        recovered = False
    failure = _failure_category(all_events, evaluation.success, evaluation.invalidated, maximum_motion)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_payload = {
        "schema_version": 1, "policy_id": policy_id, "suite_id": suite.suite_id,
        "scenario_id": scenario.scenario_id, "repeat": repeat,
        "reward_used": False, "rows": rows,
    }
    telemetry_payload["content_sha256"] = content_sha256(telemetry_payload)
    write_immutable_json(telemetry_path, telemetry_payload)
    return RolloutMetrics(
        policy_id=policy_id, suite_id=suite.suite_id, scenario_id=scenario.scenario_id,
        repeat=repeat, actions=evaluation.actions_evaluated, success=evaluation.success,
        invalidated=evaluation.invalidated, timed_out=evaluation.truncated,
        failure_category=failure, reached=DiagnosticEvent.REACH.value in all_events,
        contacted=DiagnosticEvent.FIRST_CONTACT.value in all_events,
        bilateral_contact=DiagnosticEvent.BILATERAL_INTERIOR_CONTACT.value in all_events,
        strict_grasp_acquired=DiagnosticEvent.STRICT_GRASP_ACQUIRED.value in all_events,
        grasp_losses=counts["grasp_loss"], drops=counts["drop"], maximum_height_gain_m=maximum_height,
        clipping_frames=counts["clip"], limiting_frames=counts["limit"],
        nonfinite_frames=counts["nonfinite"], unsafe_contact_frames=counts["unsafe"],
        recovery_opportunity=recovery_opportunity, recovered=recovered, recovery_actions=recovery_actions,
        final_robot_qpos=tuple(float(value) for value in rows[-1]["robot_qpos"]),
        final_cube_position=tuple(float(value) for value in rows[-1]["cube_position"]),
        telemetry_path=str(telemetry_path.resolve()),
        wrist_video_path=None if wrist_path is None else str(wrist_path.resolve()),
        overview_video_path=None if overview_path is None else str(overview_path.resolve()),
    )


def _deterministic(rollouts: list[RolloutMetrics]) -> bool:
    groups: dict[tuple[str, str], list[RolloutMetrics]] = {}
    for item in rollouts:
        groups.setdefault((item.policy_id, item.scenario_id), []).append(item)
    for values in groups.values():
        first = values[0]
        for item in values[1:]:
            if item.success != first.success or item.failure_category != first.failure_category:
                return False
            if np.max(np.abs(np.asarray(item.final_robot_qpos) - np.asarray(first.final_robot_qpos))) > 1e-6:
                return False
            if np.max(np.abs(np.asarray(item.final_cube_position) - np.asarray(first.final_cube_position))) > 1e-6:
                return False
    return True


def _write_evaluation_html(path: Path, payload: Mapping[str, Any]) -> None:
    rows = "".join(
        f"<tr><td>{item['policy_id']}</td><td>{item['scenario_id']}</td><td>{item['repeat']}</td>"
        f"<td>{item['success']}</td><td>{item['failure_category']}</td></tr>"
        for item in payload["rollouts"]
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>MuJoCo evaluation</title>"
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #999;padding:5px}</style>"
        "</head><body><h1>Reward-independent MuJoCo evaluation</h1>"
        f"<p>Environment proven: {payload['environment_proven']} · deterministic: {payload['deterministic']}</p>"
        "<table><tr><th>policy</th><th>scenario</th><th>repeat</th><th>success</th><th>failure</th></tr>"
        + rows + "</table></body></html>"
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != html:
        raise FileExistsError(f"immutable evaluation HTML differs: {path}")
    path.write_bytes(html)


def evaluate_closed_loop(
    model_path: str | Path,
    suite: SimulationSuite | str,
    policies: Mapping[str, Callable[[], SimulationPolicy]],
    output_dir: str | Path,
    *,
    environment_proven: bool | None = None,
    record_video: bool = True,
) -> SimulationEvaluation:
    """Evaluate matched policies without consulting any reward signal."""
    suite = load_simulation_suite(suite) if isinstance(suite, str) else suite
    destination = Path(output_dir) / "policies" / suite.suite_id
    if environment_proven is False:
        report: dict[str, Any] = {
            "schema_version": 1, "suite": asdict(suite), "reward_used": False,
            "environment_proven": False, "deterministic": True, "rollouts": [],
            "interpretation_allowed": False,
            "blocked_reason": "privileged controller did not prove strict task success",
            "requested_policy_ids": list(policies),
        }
        report["content_sha256"] = content_sha256(report)
        report_path = destination / "evaluation.json"
        write_immutable_json(report_path, report)
        _write_evaluation_html(destination / "evaluation.html", report)
        return SimulationEvaluation(suite, (), report_path, False, True)
    rollouts = [
        _run_rollout(Path(model_path), suite, scenario, repeat, policy_id, factory, destination, record_video=record_video)
        for policy_id, factory in policies.items()
        for scenario in suite.scenarios
        for repeat in range(suite.repeats)
    ]
    deterministic = _deterministic(rollouts)
    report: dict[str, Any] = {
        "schema_version": 1, "suite": asdict(suite), "reward_used": False,
        "environment_proven": environment_proven, "deterministic": deterministic,
        "rollouts": [asdict(item) for item in rollouts],
        "interpretation_allowed": bool(environment_proven) if environment_proven is not None else None,
    }
    report["content_sha256"] = content_sha256(report)
    report_path = destination / "evaluation.json"
    write_immutable_json(report_path, report)
    _write_evaluation_html(destination / "evaluation.html", report)
    return SimulationEvaluation(suite, tuple(rollouts), report_path, environment_proven, deterministic)


def run_simulation_preflight(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
) -> SimulationEvaluation:
    """Require the privileged staged controller to prove all fixed scenarios."""
    suite = load_simulation_suite("fixed_pickup_contract_v1")
    destination = Path(output_dir) / "preflight" / suite.suite_id
    rollouts = [
        _run_rollout(
            Path(model_path), suite, scenario, repeat, "privileged_staged",
            PrivilegedStagedController, destination, record_video=record_video,
        )
        for scenario in suite.scenarios
        for repeat in range(suite.repeats)
    ]
    deterministic = _deterministic(rollouts)
    proven = bool(
        len(rollouts) == 15
        and deterministic
        and all(
            item.success and not item.invalidated and item.clipping_frames == 0
            and item.limiting_frames == 0 and item.unsafe_contact_frames == 0
            for item in rollouts
        )
    )
    payload: dict[str, Any] = {
        "schema_version": 1, "suite": asdict(suite), "reward_used": False,
        "environment_proven": proven, "deterministic": deterministic,
        "rollouts": [asdict(item) for item in rollouts],
        "interpretation_allowed": proven,
    }
    payload["content_sha256"] = content_sha256(payload)
    report_path = destination / "evaluation.json"
    write_immutable_json(report_path, payload)
    _write_evaluation_html(destination / "evaluation.html", payload)
    return SimulationEvaluation(suite, tuple(rollouts), report_path, proven, deterministic)


__all__ = [
    "ConstantPosePolicy", "CurrentPosePolicy", "RolloutMetrics", "SimulationEvaluation",
    "SimulationPolicy", "TorchCheckpointPolicy", "evaluate_closed_loop", "run_simulation_preflight",
]
