from __future__ import annotations

import hashlib

import numpy as np

from so_arm101_v2.contracts import (
    JOINT_NAMES,
    LEGACY_MODEL_JOINT_HIGH,
    LEGACY_MODEL_JOINT_LOW,
    MODEL_CONVERSION_OFFSET,
    MODEL_CONVERSION_SIGN,
    act_to_mujoco_qpos,
)
from so_arm101_v2.data import load_json_resource, read_resource_bytes


COORDINATE_CONTRACT_SHA256 = "f95eaaf09a9635529077049b7332ee40359ba4d5c64dd8b7d23c8dd9cbad0ae7"
MODEL_CONTRACT_V2_SHA256 = "4bcde83242a31fcd69cf6d565075349a2d3df2603aeb8d435c563478c19ef5a1"


def test_packaged_coordinate_contract_is_the_pinned_legacy_artifact() -> None:
    payload = read_resource_bytes("act_coordinate_contract.json")
    assert hashlib.sha256(payload).hexdigest() == COORDINATE_CONTRACT_SHA256


def test_pinned_calibration_hash_and_joint_contract() -> None:
    contract = load_json_resource("act_coordinate_contract.json")
    calibration_name = contract["current_physical_inference_calibration"]["pinned_copy"]
    calibration_bytes = read_resource_bytes(calibration_name)
    assert (
        hashlib.sha256(calibration_bytes).hexdigest()
        == contract["current_physical_inference_calibration"]["pinned_copy_sha256"]
    )

    calibration = load_json_resource(calibration_name)
    assert tuple(calibration) == JOINT_NAMES
    assert all(calibration[name]["drive_mode"] == 0 for name in JOINT_NAMES)
    assert all(
        calibration[name]["range_min"] < calibration[name]["range_max"]
        for name in JOINT_NAMES
    )


def test_raw_calibration_endpoints_map_to_mujoco_endpoints() -> None:
    contract = load_json_resource("act_coordinate_contract.json")
    calibration = load_json_resource(
        contract["current_physical_inference_calibration"]["pinned_copy"]
    )
    encoded_low = np.empty(6, dtype=np.float32)
    encoded_high = np.empty(6, dtype=np.float32)

    for index, name in enumerate(JOINT_NAMES):
        motor = calibration[name]
        raw = np.array([motor["range_min"], motor["range_max"]], dtype=np.float32)
        if index < 5:
            normalized = (raw - motor["range_min"]) / (
                motor["range_max"] - motor["range_min"]
            ) * 200.0 - 100.0
            encoded = normalized / 100.0 * np.pi
        else:
            normalized = (raw - motor["range_min"]) / (
                motor["range_max"] - motor["range_min"]
            ) * 100.0
            encoded = normalized / 100.0 * 1.7
        encoded_low[index], encoded_high[index] = encoded

    converted_low = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_LOW + MODEL_CONVERSION_OFFSET
    converted_high = MODEL_CONVERSION_SIGN * LEGACY_MODEL_JOINT_HIGH + MODEL_CONVERSION_OFFSET
    np.testing.assert_allclose(act_to_mujoco_qpos(encoded_low), converted_low, atol=1e-6)
    np.testing.assert_allclose(act_to_mujoco_qpos(encoded_high), converted_high, atol=1e-6)


def test_model_contract_v2_is_pinned_and_consistent_with_code() -> None:
    payload = read_resource_bytes("simulation_model_contract_v2.json")
    assert hashlib.sha256(payload).hexdigest() == MODEL_CONTRACT_V2_SHA256

    contract = load_json_resource("simulation_model_contract_v2.json")
    assert tuple(contract["joint_order"]) == JOINT_NAMES
    mapping = contract["act_to_mujoco"]
    np.testing.assert_allclose(
        mapping["legacy_model_joint_low"], LEGACY_MODEL_JOINT_LOW, atol=1e-6
    )
    np.testing.assert_allclose(
        mapping["legacy_model_joint_high"], LEGACY_MODEL_JOINT_HIGH, atol=1e-6
    )
    np.testing.assert_allclose(
        mapping["model_conversion_sign"], MODEL_CONVERSION_SIGN, atol=0
    )
    np.testing.assert_allclose(
        mapping["model_conversion_offset"], MODEL_CONVERSION_OFFSET, atol=1e-7
    )

