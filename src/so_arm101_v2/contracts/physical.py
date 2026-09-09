"""Pure ACT-to-physical command diagnostics using pinned calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from so_arm101_v2.data.resources import load_json_resource

from .coordinates import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    JOINT_NAMES,
    act_to_mujoco_qpos,
    clip_mujoco_qpos,
)


PHYSICAL_NORMALIZED_LOW = np.array([-100.0] * 5 + [0.0], dtype=np.float32)
PHYSICAL_NORMALIZED_HIGH = np.array([100.0] * 6, dtype=np.float32)
PHYSICAL_NORMALIZED_LOW.setflags(write=False)
PHYSICAL_NORMALIZED_HIGH.setflags(write=False)


def _readonly(values: Any, dtype: Any) -> NDArray[np.generic]:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _pose(values: Any, label: str) -> NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim == 0 or result.shape[-1] != 6:
        raise ValueError(f"{label} must have final dimension 6, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains nonfinite values")
    return result


@dataclass(frozen=True)
class JointCalibration:
    name: str
    motor_id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    def __post_init__(self) -> None:
        if self.name not in JOINT_NAMES:
            raise ValueError(f"unknown joint calibration: {self.name}")
        if self.drive_mode not in (0, 1):
            raise ValueError(f"invalid drive_mode for {self.name}")
        if self.range_max <= self.range_min:
            raise ValueError(f"non-increasing raw range for {self.name}")


@dataclass(frozen=True)
class PhysicalCalibration:
    joints: tuple[JointCalibration, ...]
    resource_name: str

    def __post_init__(self) -> None:
        if tuple(item.name for item in self.joints) != JOINT_NAMES:
            raise ValueError("calibration joints are not in canonical order")


@dataclass(frozen=True)
class PhysicalCommandEvaluation:
    requested_act: NDArray[np.float32]
    hard_clipped_act: NDArray[np.float32]
    act_clip_mask: NDArray[np.bool_]
    requested_mujoco: NDArray[np.float32]
    clipped_mujoco: NDArray[np.float32]
    mujoco_clip_mask: NDArray[np.bool_]
    requested_physical: NDArray[np.float32]
    hard_clipped_physical: NDArray[np.float32]
    physical_clip_mask: NDArray[np.bool_]
    current_physical: NDArray[np.float32]
    relative_limited_physical: NDArray[np.float32]
    relative_limit_mask: NDArray[np.bool_]
    raw_goal_ticks: NDArray[np.int32]
    max_relative_target: float

    def __post_init__(self) -> None:
        float_names = (
            "requested_act", "hard_clipped_act", "requested_mujoco", "clipped_mujoco",
            "requested_physical", "hard_clipped_physical", "current_physical",
            "relative_limited_physical",
        )
        mask_names = (
            "act_clip_mask", "mujoco_clip_mask", "physical_clip_mask", "relative_limit_mask",
        )
        for name in float_names:
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float32))
        for name in mask_names:
            object.__setattr__(self, name, _readonly(getattr(self, name), np.bool_))
        object.__setattr__(self, "raw_goal_ticks", _readonly(self.raw_goal_ticks, np.int32))


def load_physical_calibration(
    name: str = "physical_inference_calibration_20260620.json",
) -> PhysicalCalibration:
    payload = load_json_resource(name)
    if not isinstance(payload, dict) or tuple(payload) != JOINT_NAMES:
        raise ValueError("physical calibration keys are not in canonical order")
    joints = []
    for joint_name in JOINT_NAMES:
        item = payload[joint_name]
        if not isinstance(item, dict):
            raise ValueError(f"calibration for {joint_name} must be an object")
        try:
            joints.append(JointCalibration(
                name=joint_name,
                motor_id=int(item["id"]),
                drive_mode=int(item["drive_mode"]),
                homing_offset=int(item["homing_offset"]),
                range_min=int(item["range_min"]),
                range_max=int(item["range_max"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid calibration for {joint_name}") from exc
    return PhysicalCalibration(tuple(joints), name)


def act_to_physical_normalized(values: Any) -> NDArray[np.float32]:
    source = _pose(values, "ACT pose")
    fraction = (source - ACT_DATASET_LOW) / (ACT_DATASET_HIGH - ACT_DATASET_LOW)
    return np.asarray(
        PHYSICAL_NORMALIZED_LOW
        + fraction * (PHYSICAL_NORMALIZED_HIGH - PHYSICAL_NORMALIZED_LOW),
        dtype=np.float32,
    )


def physical_normalized_to_act(values: Any) -> NDArray[np.float32]:
    source = _pose(values, "physical normalized pose")
    fraction = (source - PHYSICAL_NORMALIZED_LOW) / (
        PHYSICAL_NORMALIZED_HIGH - PHYSICAL_NORMALIZED_LOW
    )
    return np.asarray(
        ACT_DATASET_LOW + fraction * (ACT_DATASET_HIGH - ACT_DATASET_LOW),
        dtype=np.float32,
    )


def _raw_ticks(
    physical: NDArray[np.float32], calibration: PhysicalCalibration
) -> NDArray[np.int32]:
    result = np.empty(6, dtype=np.int32)
    for index, item in enumerate(calibration.joints):
        value = float(physical[index])
        if item.drive_mode:
            value = -value if index < 5 else 100.0 - value
        if index < 5:
            fraction = (value + 100.0) / 200.0
        else:
            fraction = value / 100.0
        result[index] = int(fraction * (item.range_max - item.range_min) + item.range_min)
    return result


def evaluate_physical_command(
    current_act: Any,
    target_act: Any,
    *,
    calibration: PhysicalCalibration | None = None,
    max_relative_target: float = 20.0,
    shoulder_floor: float | None = None,
    joint_map: Any | None = None,
) -> PhysicalCommandEvaluation:
    """Evaluate hard clipping, relative limiting, and raw ticks without I/O.

    ``joint_map`` selects the ACT<->MuJoCo leg (a ``JointMap``); ``None`` keeps
    the legacy affine map so every legacy evaluation stays bit-identical.
    """
    current = _pose(current_act, "current ACT pose")
    target = _pose(target_act, "target ACT pose")
    if current.shape != (6,) or target.shape != (6,):
        raise ValueError("physical command evaluation accepts single six-joint poses")
    if not np.isfinite(max_relative_target) or max_relative_target <= 0:
        raise ValueError("max_relative_target must be positive and finite")
    calibration = calibration or load_physical_calibration()

    clipped_act = np.clip(target, ACT_DATASET_LOW, ACT_DATASET_HIGH).astype(np.float32)
    act_mask = clipped_act != target
    if joint_map is None:
        requested_mujoco = act_to_mujoco_qpos(target)
        clipped_mujoco, mujoco_mask = clip_mujoco_qpos(requested_mujoco)
    else:
        requested_mujoco = joint_map.act_to_mujoco(target)
        clipped_mujoco, mujoco_mask = joint_map.clip_mujoco(requested_mujoco)
    requested_physical = act_to_physical_normalized(target)
    physical_low = PHYSICAL_NORMALIZED_LOW.copy()
    if shoulder_floor is not None:
        if not np.isfinite(shoulder_floor) or not -100 <= shoulder_floor <= 100:
            raise ValueError("invalid shoulder floor")
        physical_low[1] = max(physical_low[1], shoulder_floor)
    clipped_physical = np.clip(
        requested_physical, physical_low, PHYSICAL_NORMALIZED_HIGH
    ).astype(np.float32)
    physical_mask = clipped_physical != requested_physical
    if shoulder_floor is not None:
        # LeRobot truncates normalized goals to integer encoder ticks. A
        # nominally legal boundary target can therefore cross the wire limit.
        joint = calibration.joints[1]
        raw = _raw_ticks(requested_physical, calibration)[1]
        decoded = (raw - joint.range_min) / (joint.range_max - joint.range_min) * 200 - 100
        if joint.drive_mode:
            decoded = -decoded
        if decoded < shoulder_floor:
            physical_mask[1] = True
    current_physical = np.clip(
        act_to_physical_normalized(current),
        PHYSICAL_NORMALIZED_LOW,
        PHYSICAL_NORMALIZED_HIGH,
    ).astype(np.float32)
    delta = clipped_physical - current_physical
    limited = current_physical + np.clip(
        delta, -max_relative_target, max_relative_target
    )
    limited = np.asarray(limited, dtype=np.float32)
    # Compare the requested physical delta directly. Reconstructing ACT and
    # physical values introduces float32 round-off that must not masquerade as
    # a real limiter intervention.
    relative_mask = np.abs(delta) > np.float32(max_relative_target)
    return PhysicalCommandEvaluation(
        target, clipped_act, act_mask, requested_mujoco, clipped_mujoco,
        mujoco_mask, requested_physical, clipped_physical, physical_mask,
        current_physical, limited, relative_mask, _raw_ticks(limited, calibration),
        float(max_relative_target),
    )


@dataclass(frozen=True)
class HoldDecision:
    """One gated control step: what the policy asked, what will be executed, and why it was held.

    Shared by the simulator adapter and the physical runner so both apply the
    identical bench rule: a nonfinite request becomes the current pose, and any
    clip or limiter mask holds the current pose outright (no partial limiting).
    ``sent_physical`` is the servo-normalized command to write: the
    relative-limited request, or the measured present pose when held.
    """

    policy_act: NDArray[np.float32]
    requested_act: NDArray[np.float32]
    executed_act: NDArray[np.float32]
    sent_physical: NDArray[np.float32]
    raw_goal_ticks: NDArray[np.int32]
    held: bool
    hold_reason: str
    nonfinite: bool
    evaluation: PhysicalCommandEvaluation

    @property
    def command_bound_violation(self) -> bool:
        e = self.evaluation
        return bool(np.any(e.act_clip_mask) or np.any(e.mujoco_clip_mask) or np.any(e.physical_clip_mask))

    @property
    def delta_limiter_activated(self) -> bool:
        return bool(np.any(self.evaluation.relative_limit_mask))


def _mask_reason(evaluation: PhysicalCommandEvaluation) -> str:
    parts = []
    for name in ("act_clip", "mujoco_clip", "physical_clip", "relative_limit"):
        mask = getattr(evaluation, f"{name}_mask")
        if np.any(mask):
            joints = ",".join(JOINT_NAMES[i] for i in np.flatnonzero(mask))
            parts.append(f"{name}:{joints}")
    return ";".join(parts)


def bench_hold_decision(
    current_act: Any,
    policy_act: Any,
    *,
    shoulder_floor: float | None,
    joint_map: Any | None,
    max_relative_target: float = 20.0,
    calibration: PhysicalCalibration | None = None,
    hold_on_any_mask: bool = True,
) -> HoldDecision:
    """Gate one policy command exactly as ``MujocoTaskAdapter.apply_policy_command`` does.

    With ``hold_on_any_mask`` (the bench rule) any mask holds the current pose;
    without it (legacy lane) the executed command is the relative-limited one.
    """
    current = np.asarray(current_act, dtype=np.float32)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise ValueError("current ACT pose must be six finite values")
    policy = np.asarray(policy_act, dtype=np.float32)
    nonfinite = policy.shape != (6,) or not np.all(np.isfinite(policy))
    requested = current.copy() if nonfinite else policy.copy()
    evaluation = evaluate_physical_command(
        current, requested, calibration=calibration, max_relative_target=max_relative_target,
        shoulder_floor=shoulder_floor, joint_map=joint_map,
    )
    executed = physical_normalized_to_act(evaluation.relative_limited_physical)
    sent = evaluation.relative_limited_physical
    reason = _mask_reason(evaluation)
    held = False
    if hold_on_any_mask and reason:
        held = True
        executed = current.copy()
        sent = evaluation.current_physical
    if nonfinite:
        reason = "nonfinite" + (";" + reason if reason else "")
    return HoldDecision(
        policy_act=policy, requested_act=requested, executed_act=np.asarray(executed, dtype=np.float32),
        sent_physical=np.asarray(sent, dtype=np.float32), raw_goal_ticks=evaluation.raw_goal_ticks,
        held=held, hold_reason=reason, nonfinite=bool(nonfinite), evaluation=evaluation,
    )


__all__ = [
    "PHYSICAL_NORMALIZED_HIGH",
    "PHYSICAL_NORMALIZED_LOW",
    "HoldDecision",
    "JointCalibration",
    "PhysicalCalibration",
    "PhysicalCommandEvaluation",
    "act_to_physical_normalized",
    "bench_hold_decision",
    "evaluate_physical_command",
    "load_physical_calibration",
    "physical_normalized_to_act",
]
