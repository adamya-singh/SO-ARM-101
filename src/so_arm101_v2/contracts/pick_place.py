"""Reward-independent full pick, place, release, and retreat contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from so_arm101_v2.data import load_json_resource

from .task import (
    DiagnosticEvent,
    TaskContract,
    TaskEvaluationState,
    TaskMeasurement,
    TaskOutcome,
    evaluate_task_step,
    load_task_contract,
)


class PickPlaceDiagnosticEvent(str, Enum):
    """Edge-triggered events specific to the full task."""

    ENTERED_PLACEMENT_REGION = "entered_placement_region"
    CUBE_SUPPORTED = "cube_supported"
    RELEASED = "released"
    SETTLED = "settled"
    RETREATED = "retreated"
    SUCCESS = "success"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class PickPlaceSpec:
    footprint_edge_margin_m: float
    maximum_support_error_m: float
    maximum_linear_speed_m_s: float
    maximum_angular_speed_rad_s: float
    settled_frames: int
    minimum_gripper_act_open: float
    minimum_jaw_cube_clearance_m: float


@dataclass(frozen=True)
class PickPlaceContract:
    contract_version: int
    task_id: str
    pickup: TaskContract
    max_actions: int
    max_seconds: float
    placement: PickPlaceSpec

    def __post_init__(self) -> None:
        bench = self.task_id == "bench_pick_replace_v1" and self.contract_version == 4
        if not bench and (self.contract_version != 3 or self.task_id != "fixed_cube_pick_place_v3"):
            raise ValueError("unsupported pick-place contract")
        if self.pickup.task_id != ("bench_pickup_v1" if bench else "fixed_cube_pickup_v1"):
            raise ValueError("v3 must extend the fixed pickup contract")
        if self.max_actions != 480 or not math.isclose(self.max_seconds, 16.0):
            raise ValueError("v3 must run for 480 actions at 30 Hz")
        spec = self.placement
        expected = (0.002, 0.001, 0.005, 0.1, 10, 0.75, 0.04)
        actual = (
            spec.footprint_edge_margin_m,
            spec.maximum_support_error_m,
            spec.maximum_linear_speed_m_s,
            spec.maximum_angular_speed_rad_s,
            spec.settled_frames,
            spec.minimum_gripper_act_open,
            spec.minimum_jaw_cube_clearance_m,
        )
        if not all(math.isclose(float(a), float(b)) for a, b in zip(actual, expected, strict=True)):
            raise ValueError("v3 placement thresholds do not match the reviewed contract")


@dataclass(frozen=True)
class PickPlaceMeasurement:
    pickup: TaskMeasurement
    cube_footprint_inside: bool
    cube_support_error_m: float
    cube_linear_speed_m_s: float
    cube_angular_speed_rad_s: float
    gripper_act: float

    def __post_init__(self) -> None:
        scalars = (
            self.cube_support_error_m,
            self.cube_linear_speed_m_s,
            self.cube_angular_speed_rad_s,
            self.gripper_act,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("pick-place measurement values must be finite")
        if self.cube_linear_speed_m_s < 0 or self.cube_angular_speed_rad_s < 0:
            raise ValueError("cube speeds must be nonnegative")
        if not isinstance(self.cube_footprint_inside, (bool, np.bool_)):
            raise ValueError("cube_footprint_inside must be boolean")


@dataclass(frozen=True)
class PickPlaceEvaluationState:
    actions_evaluated: int = 0
    pickup_state: TaskEvaluationState = TaskEvaluationState()
    pickup_completed: bool = False
    settled_frames: int = 0
    invalidated: bool = False
    completed: bool = False
    emitted_events: frozenset[PickPlaceDiagnosticEvent] = frozenset()


@dataclass(frozen=True)
class PickPlaceEvaluation:
    outcome: TaskOutcome
    success: bool
    terminated: bool
    truncated: bool
    invalidated: bool
    actions_evaluated: int
    pickup_completed: bool
    settled_frames: int
    pickup_events: tuple[DiagnosticEvent, ...]
    events: tuple[PickPlaceDiagnosticEvent, ...]


def load_pick_place_contract(name: str, *, bench_config=None) -> PickPlaceContract:
    if name == "bench_pick_replace_v1":
        from dataclasses import replace
        from .task import ObjectSpec
        if bench_config is None:
            raise ValueError("bench task requires a measured bench configuration")
        base = load_pick_place_contract("fixed_cube_pick_place_v3")
        from .physical import physical_normalized_to_act
        low = list(base.pickup.safety.act_command_low)
        low[1] = float(physical_normalized_to_act([0, bench_config.shoulder_floor, 0, 0, 0, 0])[1])
        pickup = replace(base.pickup, contract_version=3, task_id="bench_pickup_v1",
            description="Strict 20 mm bench cube pickup",
            object=ObjectSpec("cube", bench_config.cube_edge_m),
            safety=replace(base.pickup.safety, act_command_low=tuple(low),
                           mujoco_joint_low=tuple(bench_config.mujoco_low)),
            reset=replace(base.pickup.reset, scenario_id="bench_nominal",
                cube_position_m=bench_config.cube_center,
                robot_qpos_mujoco=tuple(bench_config.reset_qpos)))
        return replace(base, contract_version=4, task_id=name, pickup=pickup)
    if name != "fixed_cube_pick_place_v3":
        raise ValueError(f"unknown pick-place contract: {name!r}")
    raw = load_json_resource(f"{name}.json")
    place = raw["placement"]
    return PickPlaceContract(
        contract_version=int(raw["contract_version"]),
        task_id=str(raw["task_id"]),
        pickup=load_task_contract(str(raw["base_pickup_contract"])),
        max_actions=int(raw["episode"]["max_actions"]),
        max_seconds=float(raw["episode"]["max_seconds"]),
        placement=PickPlaceSpec(
            footprint_edge_margin_m=float(place["footprint_edge_margin_m"]),
            maximum_support_error_m=float(place["maximum_support_error_m"]),
            maximum_linear_speed_m_s=float(place["maximum_linear_speed_m_s"]),
            maximum_angular_speed_rad_s=float(place["maximum_angular_speed_rad_s"]),
            settled_frames=int(place["settled_frames"]),
            minimum_gripper_act_open=float(place["minimum_gripper_act_open"]),
            minimum_jaw_cube_clearance_m=float(place["minimum_jaw_cube_clearance_m"]),
        ),
    )


def evaluate_pick_place_step(
    contract: PickPlaceContract,
    measurement: PickPlaceMeasurement,
    state: PickPlaceEvaluationState,
) -> tuple[PickPlaceEvaluationState, PickPlaceEvaluation]:
    """Advance the v3 evaluator without consulting a reward."""
    if state.completed:
        raise ValueError("cannot evaluate another step after a terminal outcome")

    pickup_state = state.pickup_state
    pickup_completed = state.pickup_completed
    pickup_events: tuple[DiagnosticEvent, ...] = ()
    if not pickup_completed and not pickup_state.completed:
        pickup_state, pickup_evaluation = evaluate_task_step(
            contract.pickup, measurement.pickup, pickup_state
        )
        pickup_events = pickup_evaluation.events
        pickup_completed = pickup_evaluation.success

    safety_invalid = any((
        measurement.pickup.unsafe_contact,
        measurement.pickup.command_bound_violation,
        measurement.pickup.delta_limiter_activated,
        measurement.pickup.nonfinite_command,
    ))
    invalidated = state.invalidated or safety_invalid or pickup_state.invalidated

    spec = contract.placement
    supported = abs(measurement.cube_support_error_m) <= spec.maximum_support_error_m
    released = (
        not measurement.pickup.strict_bilateral_grasp
        and measurement.gripper_act >= spec.minimum_gripper_act_open
    )
    stationary = (
        measurement.cube_linear_speed_m_s <= spec.maximum_linear_speed_m_s
        and measurement.cube_angular_speed_rad_s <= spec.maximum_angular_speed_rad_s
    )
    settled_frame = measurement.cube_footprint_inside and supported and released and stationary
    if contract.task_id == "bench_pick_replace_v1":
        settled_frame = settled_frame and pickup_completed
    settled_frames = state.settled_frames + 1 if settled_frame else 0
    retreated = measurement.pickup.jaw_cube_distance_m >= spec.minimum_jaw_cube_clearance_m

    emitted = set(state.emitted_events)
    events: list[PickPlaceDiagnosticEvent] = []

    def emit(event: PickPlaceDiagnosticEvent) -> None:
        if event not in emitted:
            emitted.add(event)
            events.append(event)

    if measurement.cube_footprint_inside:
        emit(PickPlaceDiagnosticEvent.ENTERED_PLACEMENT_REGION)
    if supported:
        emit(PickPlaceDiagnosticEvent.CUBE_SUPPORTED)
    if released:
        emit(PickPlaceDiagnosticEvent.RELEASED)
    if settled_frames >= spec.settled_frames:
        emit(PickPlaceDiagnosticEvent.SETTLED)
    if retreated:
        emit(PickPlaceDiagnosticEvent.RETREATED)

    success = bool(
        pickup_completed
        and settled_frames >= spec.settled_frames
        and retreated
        and not invalidated
    )
    actions_evaluated = state.actions_evaluated + 1
    pickup_timed_out = pickup_state.completed and not pickup_completed
    if success:
        outcome = TaskOutcome.SUCCESS
        emit(PickPlaceDiagnosticEvent.SUCCESS)
    elif pickup_timed_out or actions_evaluated >= contract.max_actions:
        outcome = TaskOutcome.TIMED_OUT
        emit(PickPlaceDiagnosticEvent.TIMEOUT)
    else:
        outcome = TaskOutcome.IN_PROGRESS

    completed = outcome is not TaskOutcome.IN_PROGRESS
    next_state = PickPlaceEvaluationState(
        actions_evaluated=actions_evaluated,
        pickup_state=pickup_state,
        pickup_completed=pickup_completed,
        settled_frames=settled_frames,
        invalidated=invalidated,
        completed=completed,
        emitted_events=frozenset(emitted),
    )
    evaluation = PickPlaceEvaluation(
        outcome=outcome,
        success=success,
        terminated=success,
        truncated=outcome is TaskOutcome.TIMED_OUT,
        invalidated=invalidated,
        actions_evaluated=actions_evaluated,
        pickup_completed=pickup_completed,
        settled_frames=settled_frames,
        pickup_events=pickup_events,
        events=tuple(events),
    )
    return next_state, evaluation


__all__ = [
    "PickPlaceContract",
    "PickPlaceDiagnosticEvent",
    "PickPlaceEvaluation",
    "PickPlaceEvaluationState",
    "PickPlaceMeasurement",
    "PickPlaceSpec",
    "evaluate_pick_place_step",
    "load_pick_place_contract",
]
