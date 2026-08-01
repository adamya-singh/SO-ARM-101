"""Coordinate transforms between ACT dataset values and MuJoCo joint angles.

The physical dataset does not contain mechanical joint radians. The recorder
first reads LeRobot range-normalized motor positions and then encodes body
joints from [-100, 100] as [-pi, pi] and the gripper from [0, 100] as
[0, 1.7]. MuJoCo qpos values are mechanical hinge angles in radians.

Both representations are angle-shaped floats, but they are different affine
coordinate systems. These helpers make that boundary explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - lightweight tooling may omit torch
    torch = None


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# MuJoCo mechanical hinge limits from model/so101_new_calib.xml.
MUJOCO_JOINT_LOW = np.array(
    [
        -1.9198621771937616,
        -1.7453292519943224,
        -1.69,
        -1.6580628494556928,
        -2.7438472969992493,
        -0.17453297762778586,
    ],
    dtype=np.float32,
)
MUJOCO_JOINT_HIGH = np.array(
    [
        1.9198621771937634,
        1.7453292519943366,
        1.69,
        1.6580627293335335,
        2.841206309382605,
        1.7453291995659765,
    ],
    dtype=np.float32,
)

# Encoding written by imitation-learning/record_single_arm.py.
ACT_DATASET_LOW = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0.0], dtype=np.float32)
ACT_DATASET_HIGH = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, 1.7], dtype=np.float32)


def _validate_last_dim(values: Any, label: str) -> None:
    if values.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"{label} must have last dimension {len(JOINT_NAMES)}, got {tuple(values.shape)}")


def _numpy_affine(
    values: np.ndarray,
    source_low: np.ndarray,
    source_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    _validate_last_dim(values, "joint values")
    fraction = (values - source_low) / (source_high - source_low)
    return (target_low + fraction * (target_high - target_low)).astype(np.float32, copy=False)


def mujoco_qpos_to_act(values: np.ndarray) -> np.ndarray:
    """Convert MuJoCo mechanical radians to the physical ACT dataset encoding."""
    return _numpy_affine(
        values,
        MUJOCO_JOINT_LOW,
        MUJOCO_JOINT_HIGH,
        ACT_DATASET_LOW,
        ACT_DATASET_HIGH,
    )


def act_to_mujoco_qpos(values: np.ndarray) -> np.ndarray:
    """Convert physical ACT dataset values to MuJoCo mechanical radians."""
    return _numpy_affine(
        values,
        ACT_DATASET_LOW,
        ACT_DATASET_HIGH,
        MUJOCO_JOINT_LOW,
        MUJOCO_JOINT_HIGH,
    )


def clip_mujoco_qpos(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip MuJoCo targets and return a per-joint clipping mask."""
    values = np.asarray(values, dtype=np.float32)
    _validate_last_dim(values, "MuJoCo targets")
    clipped = np.clip(values, MUJOCO_JOINT_LOW, MUJOCO_JOINT_HIGH)
    return clipped, np.not_equal(clipped, values)


def _torch_bounds(reference: "torch.Tensor", values: np.ndarray) -> "torch.Tensor":
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _torch_affine(
    values: "torch.Tensor",
    source_low: np.ndarray,
    source_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is required for differentiable ACT coordinate conversion")
    _validate_last_dim(values, "joint values")
    source_low_tensor = _torch_bounds(values, source_low)
    source_high_tensor = _torch_bounds(values, source_high)
    target_low_tensor = _torch_bounds(values, target_low)
    target_high_tensor = _torch_bounds(values, target_high)
    fraction = (values - source_low_tensor) / (source_high_tensor - source_low_tensor)
    return target_low_tensor + fraction * (target_high_tensor - target_low_tensor)


def mujoco_qpos_to_act_torch(values: "torch.Tensor") -> "torch.Tensor":
    """Differentiable MuJoCo mechanical radians -> ACT dataset conversion."""
    return _torch_affine(
        values,
        MUJOCO_JOINT_LOW,
        MUJOCO_JOINT_HIGH,
        ACT_DATASET_LOW,
        ACT_DATASET_HIGH,
    )


def act_to_mujoco_qpos_torch(values: "torch.Tensor") -> "torch.Tensor":
    """Differentiable ACT dataset -> MuJoCo mechanical radians conversion."""
    return _torch_affine(
        values,
        ACT_DATASET_LOW,
        ACT_DATASET_HIGH,
        MUJOCO_JOINT_LOW,
        MUJOCO_JOINT_HIGH,
    )
