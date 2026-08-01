from __future__ import annotations

import numpy as np
import pytest

from so_arm101_v2.contracts import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    JOINT_NAMES,
    LEGACY_MODEL_JOINT_HIGH,
    LEGACY_MODEL_JOINT_LOW,
    MODEL_CONVERSION_OFFSET,
    MODEL_CONVERSION_SIGN,
    MUJOCO_JOINT_HIGH,
    MUJOCO_JOINT_LOW,
    act_to_mujoco_qpos,
    clip_mujoco_qpos,
    mujoco_qpos_to_act,
)


def test_joint_order_is_the_recording_order() -> None:
    assert JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )


def test_coordinate_bounds_are_read_only() -> None:
    for bounds in (ACT_DATASET_LOW, ACT_DATASET_HIGH, MUJOCO_JOINT_LOW, MUJOCO_JOINT_HIGH):
        assert not bounds.flags.writeable


def test_model_conversion_constants_are_pinned() -> None:
    np.testing.assert_allclose(MODEL_CONVERSION_SIGN, [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        MODEL_CONVERSION_OFFSET,
        [0.0, -np.pi / 2, np.pi / 2, 0.0, -0.04867319, 0.0],
        atol=1e-7,
    )


def test_act_endpoints_map_to_converted_calibrated_endpoints() -> None:
    converted_low = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_LOW + MODEL_CONVERSION_OFFSET
    converted_high = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_HIGH + MODEL_CONVERSION_OFFSET
    np.testing.assert_allclose(act_to_mujoco_qpos(ACT_DATASET_LOW), converted_low, atol=1e-6)
    np.testing.assert_allclose(act_to_mujoco_qpos(ACT_DATASET_HIGH), converted_high, atol=1e-6)
    np.testing.assert_allclose(mujoco_qpos_to_act(converted_low), ACT_DATASET_LOW, atol=1e-6)
    np.testing.assert_allclose(mujoco_qpos_to_act(converted_high), ACT_DATASET_HIGH, atol=1e-6)


def test_mujoco_bounds_are_intersection_of_envelope_and_mechanics() -> None:
    converted_low = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_LOW + MODEL_CONVERSION_OFFSET
    converted_high = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_HIGH + MODEL_CONVERSION_OFFSET
    envelope_low = np.minimum(converted_low, converted_high)
    envelope_high = np.maximum(converted_low, converted_high)
    assert np.all(MUJOCO_JOINT_LOW >= envelope_low - 1e-6)
    assert np.all(MUJOCO_JOINT_HIGH <= envelope_high + 1e-6)
    assert np.all(MUJOCO_JOINT_LOW < MUJOCO_JOINT_HIGH)


def test_midpoints_include_expected_gripper_scaling() -> None:
    act_midpoint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.85], dtype=np.float32)
    legacy_midpoint = (LEGACY_MODEL_JOINT_LOW + LEGACY_MODEL_JOINT_HIGH) / 2
    expected = MODEL_CONVERSION_SIGN * legacy_midpoint + MODEL_CONVERSION_OFFSET
    np.testing.assert_allclose(act_to_mujoco_qpos(act_midpoint), expected, atol=1e-5)
    round_trip = mujoco_qpos_to_act(act_to_mujoco_qpos(act_midpoint))
    assert float(round_trip[5]) == pytest.approx(0.85, abs=1e-6)


def test_single_and_batched_round_trips_preserve_float32() -> None:
    single = np.linspace(0.05, 0.95, 6, dtype=np.float32)
    single_qpos = MUJOCO_JOINT_LOW + single * (MUJOCO_JOINT_HIGH - MUJOCO_JOINT_LOW)
    single_round_trip = act_to_mujoco_qpos(mujoco_qpos_to_act(single_qpos))
    assert single_round_trip.dtype == np.float32
    np.testing.assert_allclose(single_round_trip, single_qpos, atol=1e-6)

    rng = np.random.default_rng(7)
    fraction = rng.uniform(0.0, 1.0, size=(4, 30, 6)).astype(np.float32)
    batch = MUJOCO_JOINT_LOW + fraction * (MUJOCO_JOINT_HIGH - MUJOCO_JOINT_LOW)
    np.testing.assert_allclose(
        act_to_mujoco_qpos(mujoco_qpos_to_act(batch)), batch, atol=1e-6
    )


def test_fixed_golden_values_match_the_composed_v2_mapping() -> None:
    act_values = np.array([0.0, -1.0, 1.0, 2.0, -2.0, 0.85], dtype=np.float32)
    # legacy endpoint mapping gives [0.0, -0.5555556, 0.5379437, 1.0555555,
    # -1.7290983, 0.7853981]; composed with SIGN/OFFSET:
    expected_qpos = np.array(
        [0.0, -2.1263518, 2.1087403, 1.0555555, -1.7777715, 0.7853981],
        dtype=np.float32,
    )
    np.testing.assert_allclose(act_to_mujoco_qpos(act_values), expected_qpos, atol=1e-6)


def test_conversion_extrapolates_instead_of_silently_clipping() -> None:
    outside_act = ACT_DATASET_HIGH + np.float32(0.5)
    at_high = act_to_mujoco_qpos(ACT_DATASET_HIGH)
    converted = act_to_mujoco_qpos(outside_act)
    # extrapolation continues past the envelope in the direction of each
    # joint's conversion sign instead of silently clipping
    assert np.all((converted - at_high) * MODEL_CONVERSION_SIGN > 0)


def test_clipping_is_explicit_and_returns_per_joint_mask() -> None:
    targets = np.stack(
        [MUJOCO_JOINT_LOW - 0.1, (MUJOCO_JOINT_LOW + MUJOCO_JOINT_HIGH) / 2, MUJOCO_JOINT_HIGH + 0.1]
    )
    clipped, mask = clip_mujoco_qpos(targets)
    np.testing.assert_allclose(clipped[0], MUJOCO_JOINT_LOW)
    np.testing.assert_allclose(clipped[2], MUJOCO_JOINT_HIGH)
    assert mask.dtype == np.bool_
    assert mask[0].all()
    assert not mask[1].any()
    assert mask[2].all()


def test_historical_wrist_flex_value_does_not_saturate() -> None:
    dataset_action = np.zeros(6, dtype=np.float32)
    dataset_action[3] = 2.8487709
    dataset_action[5] = 0.3197835
    qpos = act_to_mujoco_qpos(dataset_action)
    assert float(qpos[3]) < float(MUJOCO_JOINT_HIGH[3])
    assert float(qpos[3]) == pytest.approx(1.50356, abs=1e-4)
    _, mask = clip_mujoco_qpos(qpos)
    assert not mask.any()


@pytest.mark.parametrize(
    "values",
    [0.0, np.zeros(5), np.zeros((2, 7)), np.zeros((2, 3, 5))],
)
def test_invalid_final_dimensions_are_rejected(values: object) -> None:
    with pytest.raises(ValueError, match="last dimension 6"):
        act_to_mujoco_qpos(values)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected(bad: float) -> None:
    values = np.zeros(6, dtype=np.float32)
    values[2] = bad
    with pytest.raises(ValueError, match="finite"):
        mujoco_qpos_to_act(values)
    with pytest.raises(ValueError, match="finite"):
        clip_mujoco_qpos(values)

