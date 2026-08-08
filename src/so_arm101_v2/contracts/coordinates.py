"""Explicit coordinate transforms for the SO-ARM-101 ACT dataset and MuJoCo.

The physical ACT dataset stores calibrated motor-range encodings, not MuJoCo
mechanical radians.  Conversion happens in two composed steps:

1. The original affine endpoint mapping onto the LEGACY simulation model's
   calibrated joint ranges (``so101_new_calib.xml``, whose zeros are the
   LeRobot calibration pose).  Those endpoints are preserved verbatim as
   ``LEGACY_MODEL_JOINT_LOW/HIGH`` because they encode the physical
   calibration-endpoint correspondence recorded in
   ``act_coordinate_contract.json``.
2. A per-joint rigid conversion ``q_menagerie = SIGN * q_legacy + OFFSET``
   into the vendored MuJoCo Menagerie ``trs_so_arm100`` model's CAD joint
   conventions.  SIGN/OFFSET were derived numerically by matching physical
   invariants (joint axis lines, distal body centers of mass) between the two
   models and are re-verified by ``tests/test_model_conversion.py``.  See
   ``data/resources/simulation_model_contract_v2.json``.

Conversion deliberately does not clip; callers must request clipping explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Calibrated joint ranges of the legacy so101_new_calib.xml model. These are
# the target endpoints of the original ACT endpoint mapping and must stay
# byte-equivalent to the values implied by act_coordinate_contract.json.
LEGACY_MODEL_JOINT_LOW = np.array(
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
LEGACY_MODEL_JOINT_HIGH = np.array(
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

# Per-joint rigid conversion from legacy calibrated radians to the Menagerie
# model's CAD joint conventions (derived by FK invariant matching; verified in
# tests/test_model_conversion.py against both vendored models).
MODEL_CONVERSION_SIGN = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
MODEL_CONVERSION_OFFSET = np.array(
    [0.0, -np.pi / 2, np.pi / 2, 0.0, -0.04867319, 0.0], dtype=np.float32
)

ACT_DATASET_LOW = np.array(
    [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0.0], dtype=np.float32
)
ACT_DATASET_HIGH = np.array(
    [np.pi, np.pi, np.pi, np.pi, np.pi, 1.7], dtype=np.float32
)

# Menagerie trs_so_arm100 mechanical joint ranges (upstream so_arm100.xml).
_MENAGERIE_MECHANICAL_LOW = np.array(
    [-1.92, -3.32, -0.174, -1.66, -2.79, -0.174], dtype=np.float32
)
_MENAGERIE_MECHANICAL_HIGH = np.array(
    [1.92, 0.174, 3.14, 1.66, 2.79, 1.75], dtype=np.float32
)

# Safety/clipping envelope in the new model's coordinates: the converted
# calibrated endpoints (elementwise ordered, because SIGN flips swap low/high)
# intersected with the Menagerie mechanical joint ranges.
_converted_a = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_LOW + MODEL_CONVERSION_OFFSET
_converted_b = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_HIGH + MODEL_CONVERSION_OFFSET
MUJOCO_JOINT_LOW = np.maximum(
    np.minimum(_converted_a, _converted_b), _MENAGERIE_MECHANICAL_LOW
).astype(np.float32)
MUJOCO_JOINT_HIGH = np.minimum(
    np.maximum(_converted_a, _converted_b), _MENAGERIE_MECHANICAL_HIGH
).astype(np.float32)

for _bounds in (
    LEGACY_MODEL_JOINT_LOW,
    LEGACY_MODEL_JOINT_HIGH,
    MODEL_CONVERSION_SIGN,
    MODEL_CONVERSION_OFFSET,
    MUJOCO_JOINT_LOW,
    MUJOCO_JOINT_HIGH,
    ACT_DATASET_LOW,
    ACT_DATASET_HIGH,
):
    _bounds.setflags(write=False)


def _as_joint_values(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim == 0 or result.shape[-1] != len(JOINT_NAMES):
        raise ValueError(
            f"{label} must have last dimension {len(JOINT_NAMES)}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _affine(
    values: Any,
    *,
    label: str,
    source_low: np.ndarray,
    source_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> np.ndarray:
    source = _as_joint_values(values, label)
    fraction = (source - source_low) / (source_high - source_low)
    return np.asarray(target_low + fraction * (target_high - target_low), dtype=np.float32)


def mujoco_qpos_to_act(values: Any) -> np.ndarray:
    """Convert MuJoCo (Menagerie) joint coordinates to ACT dataset coordinates."""
    qpos = _as_joint_values(values, "MuJoCo qpos")
    legacy = (qpos - MODEL_CONVERSION_OFFSET) * MODEL_CONVERSION_SIGN
    return _affine(
        legacy,
        label="legacy qpos",
        source_low=LEGACY_MODEL_JOINT_LOW,
        source_high=LEGACY_MODEL_JOINT_HIGH,
        target_low=ACT_DATASET_LOW,
        target_high=ACT_DATASET_HIGH,
    )


def act_to_mujoco_qpos(values: Any) -> np.ndarray:
    """Convert ACT dataset coordinates to MuJoCo (Menagerie) joint coordinates."""
    legacy = _affine(
        values,
        label="ACT dataset values",
        source_low=ACT_DATASET_LOW,
        source_high=ACT_DATASET_HIGH,
        target_low=LEGACY_MODEL_JOINT_LOW,
        target_high=LEGACY_MODEL_JOINT_HIGH,
    )
    return np.asarray(
        MODEL_CONVERSION_SIGN * legacy + MODEL_CONVERSION_OFFSET, dtype=np.float32
    )


def clip_mujoco_qpos(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Clip MuJoCo targets and return the clipped values and per-joint mask."""
    source = _as_joint_values(values, "MuJoCo targets")
    clipped = np.clip(source, MUJOCO_JOINT_LOW, MUJOCO_JOINT_HIGH).astype(
        np.float32, copy=False
    )
    return clipped, np.not_equal(clipped, source)


def effective_safe_act_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Per-joint ACT box whose commands raise no act or mujoco clip masks.

    The intersection of the ACT dataset box with the pullback of the MuJoCo
    joint envelope through :func:`mujoco_qpos_to_act`.  Joint 0's conversion
    sign is negative, so the pulled-back endpoints are sorted per joint before
    intersecting.  The physical-normalized clip is the exact affine image of
    the ACT box and adds no constraint.
    """
    pullback = np.sort(
        np.stack([
            mujoco_qpos_to_act(MUJOCO_JOINT_LOW),
            mujoco_qpos_to_act(MUJOCO_JOINT_HIGH),
        ]),
        axis=0,
    )
    low = np.maximum(ACT_DATASET_LOW, pullback[0]).astype(np.float32)
    high = np.minimum(ACT_DATASET_HIGH, pullback[1]).astype(np.float32)
    if np.any(low >= high):
        raise RuntimeError("effective safe act box is empty")
    return low, high


__all__ = [
    "ACT_DATASET_HIGH",
    "ACT_DATASET_LOW",
    "JOINT_NAMES",
    "LEGACY_MODEL_JOINT_HIGH",
    "LEGACY_MODEL_JOINT_LOW",
    "MODEL_CONVERSION_OFFSET",
    "MODEL_CONVERSION_SIGN",
    "MUJOCO_JOINT_HIGH",
    "MUJOCO_JOINT_LOW",
    "act_to_mujoco_qpos",
    "clip_mujoco_qpos",
    "effective_safe_act_bounds",
    "mujoco_qpos_to_act",
]
