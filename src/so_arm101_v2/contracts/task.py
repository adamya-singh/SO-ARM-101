"""Domain-neutral task contract and pure fixed-cube-pickup evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from so_arm101_v2.data import load_json_resource

from .coordinates import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    JOINT_NAMES,
    MUJOCO_JOINT_HIGH,
    MUJOCO_JOINT_LOW,
)


class TaskOutcome(str, Enum):
    """Terminal and nonterminal outcomes produced by the evaluator."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    TIMED_OUT = "timed_out"


class DiagnosticEvent(str, Enum):
    """Edge-triggered task diagnostics emitted at most once per episode."""

    REACH = "reach"
    FIRST_CONTACT = "first_contact"
    BILATERAL_INTERIOR_CONTACT = "bilateral_interior_contact"
    STRICT_GRASP_ACQUIRED = "strict_grasp_acquired"
    LIFT_5MM = "lift_5mm"
    LIFT_10MM = "lift_10mm"
    LIFT_20MM = "lift_20mm"
    GRASP_LOSS = "grasp_loss"
    DROP = "drop"
    UNSAFE_CONTACT = "unsafe_contact"
    COMMAND_BOUND_VIOLATION = "command_bound_violation"
    DELTA_LIMITER_ACTIVATED = "delta_limiter_activated"
    NONFINITE_COMMAND = "nonfinite_command"
    SUCCESS = "success"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ObjectSpec:
    kind: str
    edge_length_m: float


@dataclass(frozen=True)
class ResetSpec:
    scenario_id: str
    cube_position_m: tuple[float, float, float]
    cube_xy_tolerance_m: float
    robot_qpos_mujoco: tuple[float, ...]
    robot_qpos_tolerance_rad: float
    fixed_camera_mount: bool
    fixed_appearance: bool


@dataclass(frozen=True)
class ObservationSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]
    coordinate_domain: str | None


@dataclass(frozen=True)
class ControlSpec:
    frequency_hz: float
    action_shape: tuple[int, ...]
    action_dtype: str
    action_coordinate_domain: str
    action_mode: str
    hold_control_periods: int
    action_chunking: bool
    interpolation: bool
    temporal_aggregation: bool


@dataclass(frozen=True)
class EpisodeSpec:
    max_actions: int
    max_seconds: float
    terminate_on_success: bool


@dataclass(frozen=True)
class StrictGraspSpec:
    minimum_force_per_jaw_n: float
    minimum_edge_clearance_m: float
    maximum_alignment_error_deg: float
    acquisition_frames: int
    requires_opposing_interior_faces: bool


@dataclass(frozen=True)
class SuccessSpec:
    minimum_height_gain_m: float
    hold_frames: int
    hold_seconds: float
    requires_strict_grasp_each_frame: bool
    requires_no_safety_violation: bool


@dataclass(frozen=True)
class DiagnosticSpec:
    reach_distance_m: float
    lift_thresholds_m: tuple[float, float, float]
    drop_height_gain_m: float


@dataclass(frozen=True)
class SafetySpec:
    act_command_low: tuple[float, ...]
    act_command_high: tuple[float, ...]
    mujoco_joint_low: tuple[float, ...]
    mujoco_joint_high: tuple[float, ...]
    maximum_act_delta_per_step: tuple[float, ...]


@dataclass(frozen=True)
class TaskContract:
    """Fully resolved, immutable task definition shared by sim and physical adapters."""

    contract_version: int
    task_id: str
    description: str
    object: ObjectSpec
    reset: ResetSpec
    observations: tuple[ObservationSpec, ...]
    control: ControlSpec
    episode: EpisodeSpec
    strict_grasp: StrictGraspSpec
    success: SuccessSpec
    diagnostics: DiagnosticSpec
    safety: SafetySpec

    def __post_init__(self) -> None:
        if self.contract_version != 2 or self.task_id != "fixed_cube_pickup_v1":
            raise ValueError("Unsupported fixed-cube task contract version or id")
        if self.object.kind != "cube" or not math.isclose(self.object.edge_length_m, 0.025):
            raise ValueError("fixed_cube_pickup_v1 requires a 25 mm cube")
        cube_position = np.asarray(self.reset.cube_position_m, dtype=np.float64)
        if cube_position.shape != (3,) or not np.isfinite(cube_position).all():
            raise ValueError("cube reset position must contain three finite coordinates")
        if (
            self.reset.scenario_id != "fixed_front_v1"
            or not np.allclose(cube_position, (0.0, 0.3, 0.0125), atol=1e-9, rtol=0.0)
            or not self.reset.fixed_camera_mount
            or not self.reset.fixed_appearance
        ):
            raise ValueError("fixed_cube_pickup_v1 requires the canonical fixed reset")
        if len(self.reset.robot_qpos_mujoco) != len(JOINT_NAMES):
            raise ValueError("robot reset pose must contain six joints")
        reset = np.asarray(self.reset.robot_qpos_mujoco, dtype=np.float64)
        if not np.isfinite(reset).all():
            raise ValueError("robot reset pose must be finite")
        if np.any(reset < MUJOCO_JOINT_LOW - 1e-6) or np.any(
            reset > MUJOCO_JOINT_HIGH + 1e-6
        ):
            raise ValueError("robot reset pose exceeds MuJoCo joint bounds")
        if self.reset.cube_xy_tolerance_m <= 0 or self.reset.robot_qpos_tolerance_rad <= 0:
            raise ValueError("reset tolerances must be positive")

        observation_by_key = {item.key: item for item in self.observations}
        if len(observation_by_key) != len(self.observations):
            raise ValueError("observation keys must be unique")
        wrist = observation_by_key.get("observation.images.wrist")
        state = observation_by_key.get("observation.state")
        if wrist is None or wrist.dtype != "uint8" or wrist.shape != (256, 256, 3):
            raise ValueError("wrist observation must be uint8 RGB with shape (256, 256, 3)")
        if (
            state is None
            or state.dtype != "float32"
            or state.shape != (len(JOINT_NAMES),)
            or state.coordinate_domain != "act_dataset"
        ):
            raise ValueError("state observation must be six float32 ACT dataset values")

        if self.control.frequency_hz <= 0 or self.control.action_shape != (len(JOINT_NAMES),):
            raise ValueError("control must use positive frequency and six-joint actions")
        if (
            self.control.action_dtype != "float32"
            or self.control.action_coordinate_domain != "act_dataset"
            or self.control.action_mode != "absolute_joint_target"
            or self.control.hold_control_periods != 1
            or self.control.action_chunking
            or self.control.interpolation
            or self.control.temporal_aggregation
        ):
            raise ValueError("fixed_cube_pickup_v1 requires single-step absolute ACT targets")
        if self.episode.max_actions <= 0 or not math.isclose(
            self.episode.max_actions / self.control.frequency_hz,
            self.episode.max_seconds,
            abs_tol=1e-9,
        ):
            raise ValueError("episode action count, rate, and duration disagree")
        if not self.episode.terminate_on_success:
            raise ValueError("fixed_cube_pickup_v1 must terminate on success")
        if (
            not math.isclose(self.strict_grasp.minimum_force_per_jaw_n, 0.1)
            or not math.isclose(self.strict_grasp.minimum_edge_clearance_m, 0.004)
            or not math.isclose(self.strict_grasp.maximum_alignment_error_deg, 25.0)
            or self.strict_grasp.acquisition_frames != 5
            or not self.strict_grasp.requires_opposing_interior_faces
        ):
            raise ValueError("strict grasp settings do not match the v1 geometry contract")
        if self.success.hold_frames <= 0 or not math.isclose(
            self.success.hold_frames / self.control.frequency_hz,
            self.success.hold_seconds,
            abs_tol=1e-9,
        ):
            raise ValueError("success hold frames, rate, and duration disagree")
        if (
            not math.isclose(self.success.minimum_height_gain_m, 0.02)
            or not self.success.requires_strict_grasp_each_frame
            or not self.success.requires_no_safety_violation
        ):
            raise ValueError("success must require a safe, strict 2 cm grasped lift")
        if (
            not math.isclose(self.diagnostics.reach_distance_m, 0.04)
            or not math.isclose(self.diagnostics.drop_height_gain_m, 0.002)
        ):
            raise ValueError("diagnostic thresholds do not match fixed_cube_pickup_v1")
        if tuple(sorted(self.diagnostics.lift_thresholds_m)) != self.diagnostics.lift_thresholds_m:
            raise ValueError("lift diagnostic thresholds must be sorted")
        if not math.isclose(
            self.diagnostics.lift_thresholds_m[-1],
            self.success.minimum_height_gain_m,
            abs_tol=1e-12,
        ):
            raise ValueError("highest lift diagnostic must equal the success height")

        expected_bounds = (
            (self.safety.act_command_low, ACT_DATASET_LOW),
            (self.safety.act_command_high, ACT_DATASET_HIGH),
            (self.safety.mujoco_joint_low, MUJOCO_JOINT_LOW),
            (self.safety.mujoco_joint_high, MUJOCO_JOINT_HIGH),
        )
        for configured, expected in expected_bounds:
            if len(configured) != len(JOINT_NAMES) or not np.allclose(
                configured, expected, atol=1e-6, rtol=0.0
            ):
                raise ValueError("task safety bounds do not match the coordinate contract")
        deltas = np.asarray(self.safety.maximum_act_delta_per_step, dtype=np.float64)
        if deltas.shape != (len(JOINT_NAMES),) or not np.isfinite(deltas).all() or np.any(deltas <= 0):
            raise ValueError("maximum per-step ACT deltas must be six positive finite values")
        if not np.allclose(deltas, (np.pi / 5,) * 5 + (0.34,), atol=1e-9, rtol=0.0):
            raise ValueError("maximum per-step ACT deltas do not match the v1 safety contract")


@dataclass(frozen=True)
class TaskMeasurement:
    """Adapter-provided evidence for one post-action control frame."""

    jaw_cube_distance_m: float
    any_contact: bool
    bilateral_interior_contact: bool
    strict_bilateral_grasp: bool
    cube_height_gain_m: float
    unsafe_contact: bool = False
    command_bound_violation: bool = False
    delta_limiter_activated: bool = False
    nonfinite_command: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.jaw_cube_distance_m) or self.jaw_cube_distance_m < 0:
            raise ValueError("jaw_cube_distance_m must be finite and nonnegative")
        if not math.isfinite(self.cube_height_gain_m):
            raise ValueError("cube_height_gain_m must be finite")
        boolean_fields = (
            self.any_contact,
            self.bilateral_interior_contact,
            self.strict_bilateral_grasp,
            self.unsafe_contact,
            self.command_bound_violation,
            self.delta_limiter_activated,
            self.nonfinite_command,
        )
        if not all(isinstance(value, (bool, np.bool_)) for value in boolean_fields):
            raise ValueError("task measurement flags must be boolean")


@dataclass(frozen=True)
class TaskEvaluationState:
    """All history needed to evaluate the next frame deterministically."""

    actions_evaluated: int = 0
    strict_grasp_frames: int = 0
    success_hold_frames: int = 0
    strict_grasp_acquired: bool = False
    previous_strict_grasp: bool = False
    lifted_5mm: bool = False
    invalidated: bool = False
    completed: bool = False
    emitted_events: frozenset[DiagnosticEvent] = frozenset()


@dataclass(frozen=True)
class TaskEvaluation:
    """Reward-independent outcome and newly emitted events for one frame."""

    outcome: TaskOutcome
    success: bool
    terminated: bool
    truncated: bool
    invalidated: bool
    actions_evaluated: int
    strict_grasp_frames: int
    success_hold_frames: int
    events: tuple[DiagnosticEvent, ...]


def _tuple_of(raw: Any, cast: type, label: str) -> tuple[Any, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a JSON array")
    return tuple(cast(value) for value in raw)


def load_task_contract(name: str) -> TaskContract:
    """Load and validate a named packaged task contract."""
    if name != "fixed_cube_pickup_v1":
        raise ValueError(f"Unknown task contract: {name!r}")
    raw = load_json_resource(f"{name}.json")
    object_raw = raw["object"]
    reset_raw = raw["reset"]
    control_raw = raw["control"]
    episode_raw = raw["episode"]
    grasp_raw = raw["strict_grasp"]
    success_raw = raw["success"]
    diagnostics_raw = raw["diagnostics"]
    safety_raw = raw["safety"]

    return TaskContract(
        contract_version=int(raw["contract_version"]),
        task_id=str(raw["task_id"]),
        description=str(raw["description"]),
        object=ObjectSpec(kind=str(object_raw["kind"]), edge_length_m=float(object_raw["edge_length_m"])),
        reset=ResetSpec(
            scenario_id=str(reset_raw["scenario_id"]),
            cube_position_m=_tuple_of(reset_raw["cube_position_m"], float, "cube_position_m"),
            cube_xy_tolerance_m=float(reset_raw["cube_xy_tolerance_m"]),
            robot_qpos_mujoco=_tuple_of(reset_raw["robot_qpos_mujoco"], float, "robot_qpos_mujoco"),
            robot_qpos_tolerance_rad=float(reset_raw["robot_qpos_tolerance_rad"]),
            fixed_camera_mount=bool(reset_raw["fixed_camera_mount"]),
            fixed_appearance=bool(reset_raw["fixed_appearance"]),
        ),
        observations=tuple(
            ObservationSpec(
                key=str(item["key"]),
                dtype=str(item["dtype"]),
                shape=_tuple_of(item["shape"], int, f"{item['key']} shape"),
                coordinate_domain=(
                    None if item["coordinate_domain"] is None else str(item["coordinate_domain"])
                ),
            )
            for item in raw["observations"]
        ),
        control=ControlSpec(
            frequency_hz=float(control_raw["frequency_hz"]),
            action_shape=_tuple_of(control_raw["action_shape"], int, "action_shape"),
            action_dtype=str(control_raw["action_dtype"]),
            action_coordinate_domain=str(control_raw["action_coordinate_domain"]),
            action_mode=str(control_raw["action_mode"]),
            hold_control_periods=int(control_raw["hold_control_periods"]),
            action_chunking=bool(control_raw["action_chunking"]),
            interpolation=bool(control_raw["interpolation"]),
            temporal_aggregation=bool(control_raw["temporal_aggregation"]),
        ),
        episode=EpisodeSpec(
            max_actions=int(episode_raw["max_actions"]),
            max_seconds=float(episode_raw["max_seconds"]),
            terminate_on_success=bool(episode_raw["terminate_on_success"]),
        ),
        strict_grasp=StrictGraspSpec(
            minimum_force_per_jaw_n=float(grasp_raw["minimum_force_per_jaw_n"]),
            minimum_edge_clearance_m=float(grasp_raw["minimum_edge_clearance_m"]),
            maximum_alignment_error_deg=float(grasp_raw["maximum_alignment_error_deg"]),
            acquisition_frames=int(grasp_raw["acquisition_frames"]),
            requires_opposing_interior_faces=bool(grasp_raw["requires_opposing_interior_faces"]),
        ),
        success=SuccessSpec(
            minimum_height_gain_m=float(success_raw["minimum_height_gain_m"]),
            hold_frames=int(success_raw["hold_frames"]),
            hold_seconds=float(success_raw["hold_seconds"]),
            requires_strict_grasp_each_frame=bool(success_raw["requires_strict_grasp_each_frame"]),
            requires_no_safety_violation=bool(success_raw["requires_no_safety_violation"]),
        ),
        diagnostics=DiagnosticSpec(
            reach_distance_m=float(diagnostics_raw["reach_distance_m"]),
            lift_thresholds_m=_tuple_of(
                diagnostics_raw["lift_thresholds_m"], float, "lift_thresholds_m"
            ),
            drop_height_gain_m=float(diagnostics_raw["drop_height_gain_m"]),
        ),
        safety=SafetySpec(
            act_command_low=_tuple_of(safety_raw["act_command_low"], float, "act_command_low"),
            act_command_high=_tuple_of(safety_raw["act_command_high"], float, "act_command_high"),
            mujoco_joint_low=_tuple_of(safety_raw["mujoco_joint_low"], float, "mujoco_joint_low"),
            mujoco_joint_high=_tuple_of(safety_raw["mujoco_joint_high"], float, "mujoco_joint_high"),
            maximum_act_delta_per_step=_tuple_of(
                safety_raw["maximum_act_delta_per_step"], float, "maximum_act_delta_per_step"
            ),
        ),
    )


def evaluate_task_step(
    contract: TaskContract,
    measurement: TaskMeasurement,
    state: TaskEvaluationState,
) -> tuple[TaskEvaluationState, TaskEvaluation]:
    """Advance the fixed-cube task state by one post-action measurement."""
    if state.completed:
        raise ValueError("cannot evaluate another step after a terminal outcome")

    emitted = set(state.emitted_events)
    events: list[DiagnosticEvent] = []

    def emit(event: DiagnosticEvent) -> None:
        if event not in emitted:
            emitted.add(event)
            events.append(event)

    if measurement.jaw_cube_distance_m <= contract.diagnostics.reach_distance_m:
        emit(DiagnosticEvent.REACH)
    if measurement.any_contact:
        emit(DiagnosticEvent.FIRST_CONTACT)
    if measurement.bilateral_interior_contact:
        emit(DiagnosticEvent.BILATERAL_INTERIOR_CONTACT)

    strict_grasp_frames = (
        state.strict_grasp_frames + 1 if measurement.strict_bilateral_grasp else 0
    )
    strict_grasp_acquired = state.strict_grasp_acquired or (
        strict_grasp_frames >= contract.strict_grasp.acquisition_frames
    )
    if strict_grasp_acquired:
        emit(DiagnosticEvent.STRICT_GRASP_ACQUIRED)

    lift_5mm = state.lifted_5mm
    lift_events = (
        DiagnosticEvent.LIFT_5MM,
        DiagnosticEvent.LIFT_10MM,
        DiagnosticEvent.LIFT_20MM,
    )
    for threshold, event in zip(contract.diagnostics.lift_thresholds_m, lift_events, strict=True):
        if measurement.cube_height_gain_m >= threshold:
            emit(event)
            if event is DiagnosticEvent.LIFT_5MM:
                lift_5mm = True

    if state.strict_grasp_acquired and state.previous_strict_grasp and not measurement.strict_bilateral_grasp:
        emit(DiagnosticEvent.GRASP_LOSS)
    if (
        state.lifted_5mm
        and measurement.cube_height_gain_m <= contract.diagnostics.drop_height_gain_m
        and not measurement.strict_bilateral_grasp
    ):
        emit(DiagnosticEvent.DROP)

    if measurement.unsafe_contact:
        emit(DiagnosticEvent.UNSAFE_CONTACT)
    if measurement.command_bound_violation:
        emit(DiagnosticEvent.COMMAND_BOUND_VIOLATION)
    if measurement.delta_limiter_activated:
        emit(DiagnosticEvent.DELTA_LIMITER_ACTIVATED)
    if measurement.nonfinite_command:
        emit(DiagnosticEvent.NONFINITE_COMMAND)

    invalidated = state.invalidated or any(
        (
            measurement.unsafe_contact,
            measurement.command_bound_violation,
            measurement.delta_limiter_activated,
            measurement.nonfinite_command,
        )
    )
    success_frame = (
        not invalidated
        and measurement.strict_bilateral_grasp
        and measurement.cube_height_gain_m >= contract.success.minimum_height_gain_m
    )
    success_hold_frames = state.success_hold_frames + 1 if success_frame else 0
    success = success_hold_frames >= contract.success.hold_frames
    actions_evaluated = state.actions_evaluated + 1

    if success:
        outcome = TaskOutcome.SUCCESS
        emit(DiagnosticEvent.SUCCESS)
    elif actions_evaluated >= contract.episode.max_actions:
        outcome = TaskOutcome.TIMED_OUT
        emit(DiagnosticEvent.TIMEOUT)
    else:
        outcome = TaskOutcome.IN_PROGRESS

    completed = outcome is not TaskOutcome.IN_PROGRESS
    next_state = TaskEvaluationState(
        actions_evaluated=actions_evaluated,
        strict_grasp_frames=strict_grasp_frames,
        success_hold_frames=success_hold_frames,
        strict_grasp_acquired=strict_grasp_acquired,
        previous_strict_grasp=measurement.strict_bilateral_grasp,
        lifted_5mm=lift_5mm,
        invalidated=invalidated,
        completed=completed,
        emitted_events=frozenset(emitted),
    )
    evaluation = TaskEvaluation(
        outcome=outcome,
        success=success,
        terminated=success and contract.episode.terminate_on_success,
        truncated=outcome is TaskOutcome.TIMED_OUT,
        invalidated=invalidated,
        actions_evaluated=actions_evaluated,
        strict_grasp_frames=strict_grasp_frames,
        success_hold_frames=success_hold_frames,
        events=tuple(events),
    )
    return next_state, evaluation


__all__ = [
    "ControlSpec",
    "DiagnosticEvent",
    "DiagnosticSpec",
    "EpisodeSpec",
    "ObjectSpec",
    "ObservationSpec",
    "ResetSpec",
    "SafetySpec",
    "StrictGraspSpec",
    "SuccessSpec",
    "TaskContract",
    "TaskEvaluation",
    "TaskEvaluationState",
    "TaskMeasurement",
    "TaskOutcome",
    "evaluate_task_step",
    "load_task_contract",
]
