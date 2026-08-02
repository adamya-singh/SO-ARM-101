from __future__ import annotations

import pytest

from so_arm101_v2.contracts import (
    PickPlaceDiagnosticEvent,
    PickPlaceEvaluationState,
    PickPlaceMeasurement,
    TaskMeasurement,
    TaskOutcome,
    evaluate_pick_place_step,
    load_pick_place_contract,
)
from so_arm101_v2.simulation import load_simulation_suite


def _pickup(**overrides: object) -> TaskMeasurement:
    values: dict[str, object] = {
        "jaw_cube_distance_m": 0.02,
        "any_contact": False,
        "bilateral_interior_contact": False,
        "strict_bilateral_grasp": False,
        "cube_height_gain_m": 0.0,
    }
    values.update(overrides)
    return TaskMeasurement(**values)  # type: ignore[arg-type]


def _measurement(**overrides: object) -> PickPlaceMeasurement:
    values: dict[str, object] = {
        "pickup": _pickup(),
        "cube_footprint_inside": False,
        "cube_support_error_m": 0.03,
        "cube_linear_speed_m_s": 0.0,
        "cube_angular_speed_rad_s": 0.0,
        "gripper_act": 0.0,
    }
    values.update(overrides)
    return PickPlaceMeasurement(**values)  # type: ignore[arg-type]


def _complete_pickup(state: PickPlaceEvaluationState) -> PickPlaceEvaluationState:
    contract = load_pick_place_contract("fixed_cube_pick_place_v3")
    grasped = _measurement(
        pickup=_pickup(
            any_contact=True,
            bilateral_interior_contact=True,
            strict_bilateral_grasp=True,
            cube_height_gain_m=0.025,
        )
    )
    for _ in range(30):
        state, _ = evaluate_pick_place_step(contract, grasped, state)
    assert state.pickup_completed
    return state


def test_v3_contract_and_suite_are_distinct_from_v2() -> None:
    contract = load_pick_place_contract("fixed_cube_pick_place_v3")
    suite = load_simulation_suite("fixed_pick_place_v3")
    assert contract.pickup.task_id == "fixed_cube_pickup_v1"
    assert contract.max_actions == 480
    assert contract.placement.settled_frames == 10
    assert suite.task_contract == contract.task_id
    assert len(suite.scenarios) == 5 and suite.repeats == 3


def test_v3_requires_pickup_settle_release_and_retreat() -> None:
    contract = load_pick_place_contract("fixed_cube_pick_place_v3")
    state = _complete_pickup(PickPlaceEvaluationState())
    placed = _measurement(
        pickup=_pickup(jaw_cube_distance_m=0.02),
        cube_footprint_inside=True,
        cube_support_error_m=0.0002,
        cube_linear_speed_m_s=0.001,
        cube_angular_speed_rad_s=0.01,
        gripper_act=0.8,
    )
    for _ in range(10):
        state, result = evaluate_pick_place_step(contract, placed, state)
    assert not result.success
    assert PickPlaceDiagnosticEvent.SETTLED in state.emitted_events

    retreated = _measurement(
        pickup=_pickup(jaw_cube_distance_m=0.05),
        cube_footprint_inside=True,
        cube_support_error_m=0.0002,
        cube_linear_speed_m_s=0.001,
        cube_angular_speed_rad_s=0.01,
        gripper_act=0.8,
    )
    state, result = evaluate_pick_place_step(contract, retreated, state)
    assert result.success and result.terminated
    assert result.outcome is TaskOutcome.SUCCESS
    assert PickPlaceDiagnosticEvent.RETREATED in result.events
    assert PickPlaceDiagnosticEvent.SUCCESS in result.events
    with pytest.raises(ValueError, match="terminal outcome"):
        evaluate_pick_place_step(contract, retreated, state)


@pytest.mark.parametrize(
    "bad_measurement",
    [
        _measurement(cube_footprint_inside=False, cube_support_error_m=0.0, gripper_act=0.8),
        _measurement(cube_footprint_inside=True, cube_support_error_m=0.002, gripper_act=0.8),
        _measurement(cube_footprint_inside=True, cube_support_error_m=0.0, cube_linear_speed_m_s=0.006, gripper_act=0.8),
        _measurement(cube_footprint_inside=True, cube_support_error_m=0.0, cube_angular_speed_rad_s=0.2, gripper_act=0.8),
        _measurement(cube_footprint_inside=True, cube_support_error_m=0.0, gripper_act=0.7),
        _measurement(
            pickup=_pickup(strict_bilateral_grasp=True),
            cube_footprint_inside=True,
            cube_support_error_m=0.0,
            gripper_act=0.8,
        ),
    ],
)
def test_v3_rejects_each_incomplete_place_condition(
    bad_measurement: PickPlaceMeasurement,
) -> None:
    contract = load_pick_place_contract("fixed_cube_pick_place_v3")
    state = _complete_pickup(PickPlaceEvaluationState())
    for _ in range(12):
        state, result = evaluate_pick_place_step(contract, bad_measurement, state)
    assert not result.success
    assert state.settled_frames == 0


def test_v3_placement_without_pickup_and_safety_violation_cannot_pass() -> None:
    contract = load_pick_place_contract("fixed_cube_pick_place_v3")
    placed = _measurement(
        pickup=_pickup(jaw_cube_distance_m=0.05),
        cube_footprint_inside=True,
        cube_support_error_m=0.0,
        gripper_act=0.8,
    )
    state = PickPlaceEvaluationState()
    for _ in range(20):
        state, result = evaluate_pick_place_step(contract, placed, state)
    assert not result.success and not state.pickup_completed

    state = _complete_pickup(PickPlaceEvaluationState())
    unsafe = _measurement(pickup=_pickup(unsafe_contact=True))
    state, result = evaluate_pick_place_step(contract, unsafe, state)
    assert result.invalidated
    for _ in range(12):
        state, result = evaluate_pick_place_step(contract, placed, state)
    assert not result.success
