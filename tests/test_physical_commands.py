from __future__ import annotations

import numpy as np
import pytest

from so_arm101_v2.contracts import ACT_DATASET_HIGH, ACT_DATASET_LOW
from so_arm101_v2.contracts.physical import (
    PHYSICAL_NORMALIZED_HIGH,
    PHYSICAL_NORMALIZED_LOW,
    JointCalibration,
    PhysicalCalibration,
    act_to_physical_normalized,
    evaluate_physical_command,
    load_physical_calibration,
    physical_normalized_to_act,
)


def test_act_physical_endpoints_and_round_trip() -> None:
    np.testing.assert_allclose(act_to_physical_normalized(ACT_DATASET_LOW), PHYSICAL_NORMALIZED_LOW)
    np.testing.assert_allclose(act_to_physical_normalized(ACT_DATASET_HIGH), PHYSICAL_NORMALIZED_HIGH)
    values = np.stack([ACT_DATASET_LOW, np.zeros(6, dtype=np.float32), ACT_DATASET_HIGH])
    values[1, 5] = 0.85
    np.testing.assert_allclose(
        physical_normalized_to_act(act_to_physical_normalized(values)), values, atol=1e-6
    )


def test_raw_tick_endpoints_match_lerobot_formula() -> None:
    calibration = load_physical_calibration()
    low = evaluate_physical_command(ACT_DATASET_LOW, ACT_DATASET_LOW, calibration=calibration)
    high = evaluate_physical_command(ACT_DATASET_HIGH, ACT_DATASET_HIGH, calibration=calibration)
    np.testing.assert_array_equal(low.raw_goal_ticks, [item.range_min for item in calibration.joints])
    np.testing.assert_array_equal(high.raw_goal_ticks, [item.range_max for item in calibration.joints])


def test_hard_clipping_and_relative_limiting_are_explicit() -> None:
    current = np.zeros(6, dtype=np.float32)
    current[5] = 0.85
    target = np.array([10, -10, 0, 0, 0, 3], dtype=np.float32)
    result = evaluate_physical_command(current, target, max_relative_target=20)
    np.testing.assert_array_equal(result.act_clip_mask, [True, True, False, False, False, True])
    np.testing.assert_array_equal(result.physical_clip_mask, result.act_clip_mask)
    np.testing.assert_array_equal(result.mujoco_clip_mask, result.act_clip_mask)
    np.testing.assert_allclose(
        result.relative_limited_physical - result.current_physical,
        [20, -20, 0, 0, 0, 20],
    )
    np.testing.assert_array_equal(result.relative_limit_mask, [True, True, False, False, False, True])


def test_physical_conversion_rejects_bad_inputs_and_calibration() -> None:
    with pytest.raises(ValueError, match="single six-joint"):
        evaluate_physical_command(np.zeros((2, 6)), np.zeros((2, 6)))
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_physical_command(np.zeros(6), [0, 0, 0, 0, 0, np.nan])
    with pytest.raises(ValueError, match="positive"):
        evaluate_physical_command(np.zeros(6), np.zeros(6), max_relative_target=0)
    bad = tuple(
        JointCalibration(name, index + 1, 0, 0, 10, 20)
        for index, name in enumerate(reversed((
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
            "wrist_roll", "gripper",
        )))
    )
    with pytest.raises(ValueError, match="canonical order"):
        PhysicalCalibration(bad, "bad.json")


def test_roundoff_does_not_report_a_relative_limiter_activation() -> None:
    current = np.array([0.016, 0.34, 1.26, -0.14, -1.29, 0.002], dtype=np.float32)
    target = np.array([0.017, 0.416, 1.322, -0.1045, -1.289, 0.0017], dtype=np.float32)
    result = evaluate_physical_command(current, target)
    assert not np.any(result.relative_limit_mask)
