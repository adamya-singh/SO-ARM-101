from __future__ import annotations

from dataclasses import fields

import pytest

from so_arm101_v2.contracts import (
    DiagnosticEvent,
    TaskEvaluation,
    TaskEvaluationState,
    TaskMeasurement,
    TaskOutcome,
    evaluate_task_step,
    load_task_contract,
)


def measurement(**overrides: object) -> TaskMeasurement:
    values: dict[str, object] = {
        "jaw_cube_distance_m": 0.2,
        "any_contact": False,
        "bilateral_interior_contact": False,
        "strict_bilateral_grasp": False,
        "cube_height_gain_m": 0.0,
    }
    values.update(overrides)
    return TaskMeasurement(**values)  # type: ignore[arg-type]


def test_canonical_contract_values_and_semantics() -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    observations = {item.key: item for item in contract.observations}

    assert contract.reset.scenario_id == "fixed_front_v1"
    assert contract.reset.cube_position_m == (0.0, 0.3, 0.0125)
    assert contract.reset.robot_qpos_mujoco == pytest.approx(
        (-0.0010472, -3.31, 3.13, 1.1599458, -1.5154479, 0.0314159)
    )
    assert observations["observation.images.wrist"].shape == (256, 256, 3)
    assert observations["observation.state"].coordinate_domain == "act_dataset"
    assert contract.control.frequency_hz == 30.0
    assert contract.control.action_mode == "absolute_joint_target"
    assert contract.episode.max_actions == 450
    assert contract.episode.max_seconds == 15.0
    assert contract.success.minimum_height_gain_m == 0.02
    assert contract.success.hold_frames == 30
    assert contract.success.hold_seconds == 1.0
    assert contract.safety.maximum_act_delta_per_step == pytest.approx(
        (0.6283185307,) * 5 + (0.34,)
    )


def test_unknown_task_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown task contract"):
        load_task_contract("future_task")


def test_success_requires_thirty_consecutive_grasped_lift_frames() -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    state = TaskEvaluationState()
    grasped_lift = measurement(
        jaw_cube_distance_m=0.01,
        any_contact=True,
        bilateral_interior_contact=True,
        strict_bilateral_grasp=True,
        cube_height_gain_m=0.02,
    )

    for _ in range(29):
        state, result = evaluate_task_step(contract, grasped_lift, state)
        assert not result.success
        assert result.outcome is TaskOutcome.IN_PROGRESS

    state, result = evaluate_task_step(contract, grasped_lift, state)
    assert result.success
    assert result.terminated
    assert not result.truncated
    assert result.success_hold_frames == 30
    assert DiagnosticEvent.SUCCESS in result.events
    assert state.completed


def test_grasp_break_and_height_drop_reset_success_hold() -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    state = TaskEvaluationState()
    grasped_lift = measurement(strict_bilateral_grasp=True, cube_height_gain_m=0.025)

    for _ in range(20):
        state, _ = evaluate_task_step(contract, grasped_lift, state)
    state, result = evaluate_task_step(
        contract,
        measurement(strict_bilateral_grasp=False, cube_height_gain_m=0.025),
        state,
    )
    assert result.success_hold_frames == 0

    for _ in range(10):
        state, _ = evaluate_task_step(contract, grasped_lift, state)
    state, result = evaluate_task_step(
        contract,
        measurement(strict_bilateral_grasp=True, cube_height_gain_m=0.019),
        state,
    )
    assert result.success_hold_frames == 0
    assert not result.success


def test_one_frame_height_spike_cannot_succeed() -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    state = TaskEvaluationState()
    state, result = evaluate_task_step(
        contract,
        measurement(strict_bilateral_grasp=True, cube_height_gain_m=0.02),
        state,
    )
    assert result.success_hold_frames == 1
    for _ in range(40):
        state, result = evaluate_task_step(
            contract,
            measurement(strict_bilateral_grasp=True, cube_height_gain_m=0.0),
            state,
        )
    assert not result.success
    assert result.success_hold_frames == 0


@pytest.mark.parametrize(
    ("flag", "event"),
    [
        ("unsafe_contact", DiagnosticEvent.UNSAFE_CONTACT),
        ("command_bound_violation", DiagnosticEvent.COMMAND_BOUND_VIOLATION),
        ("delta_limiter_activated", DiagnosticEvent.DELTA_LIMITER_ACTIVATED),
        ("nonfinite_command", DiagnosticEvent.NONFINITE_COMMAND),
    ],
)
def test_safety_violations_permanently_prevent_success(flag: str, event: DiagnosticEvent) -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    state, result = evaluate_task_step(contract, measurement(**{flag: True}), TaskEvaluationState())
    assert result.invalidated
    assert event in result.events

    grasped_lift = measurement(strict_bilateral_grasp=True, cube_height_gain_m=0.03)
    for _ in range(40):
        state, result = evaluate_task_step(contract, grasped_lift, state)
    assert result.invalidated
    assert not result.success
    assert result.success_hold_frames == 0


def test_diagnostics_are_edge_triggered_once() -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    state = TaskEvaluationState()
    contact_grasp = measurement(
        jaw_cube_distance_m=0.03,
        any_contact=True,
        bilateral_interior_contact=True,
        strict_bilateral_grasp=True,
        cube_height_gain_m=0.005,
    )

    state, first = evaluate_task_step(contract, contact_grasp, state)
    assert set(first.events) == {
        DiagnosticEvent.REACH,
        DiagnosticEvent.FIRST_CONTACT,
        DiagnosticEvent.BILATERAL_INTERIOR_CONTACT,
        DiagnosticEvent.LIFT_5MM,
    }
    for _ in range(4):
        state, acquired = evaluate_task_step(contract, contact_grasp, state)
    assert acquired.events == (DiagnosticEvent.STRICT_GRASP_ACQUIRED,)

    state, lifted = evaluate_task_step(
        contract, measurement(strict_bilateral_grasp=True, cube_height_gain_m=0.02), state
    )
    assert lifted.events == (DiagnosticEvent.LIFT_10MM, DiagnosticEvent.LIFT_20MM)

    state, lost = evaluate_task_step(
        contract, measurement(strict_bilateral_grasp=False, cube_height_gain_m=0.02), state
    )
    assert lost.events == (DiagnosticEvent.GRASP_LOSS,)
    state, dropped = evaluate_task_step(contract, measurement(cube_height_gain_m=0.0), state)
    assert dropped.events == (DiagnosticEvent.DROP,)
    state, repeated = evaluate_task_step(contract, measurement(cube_height_gain_m=0.0), state)
    assert repeated.events == ()


def test_timeout_occurs_at_action_450_and_is_reward_independent() -> None:
    contract = load_task_contract("fixed_cube_pickup_v1")
    state = TaskEvaluationState()
    for _ in range(449):
        state, result = evaluate_task_step(contract, measurement(), state)
        assert result.outcome is TaskOutcome.IN_PROGRESS

    state, result = evaluate_task_step(contract, measurement(), state)
    assert result.outcome is TaskOutcome.TIMED_OUT
    assert result.truncated
    assert not result.success
    assert DiagnosticEvent.TIMEOUT in result.events
    assert "reward" not in {field.name for field in fields(TaskMeasurement)}
    assert "reward" not in {field.name for field in fields(TaskEvaluation)}

    with pytest.raises(ValueError, match="terminal outcome"):
        evaluate_task_step(contract, measurement(), state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jaw_cube_distance_m": float("nan")},
        {"jaw_cube_distance_m": -0.1},
        {"cube_height_gain_m": float("inf")},
        {"any_contact": "yes"},
    ],
)
def test_measurements_reject_invalid_evidence(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        measurement(**kwargs)
