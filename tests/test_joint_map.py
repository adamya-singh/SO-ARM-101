"""The measured physical-to-MuJoCo joint map and its relationship to the legacy affine map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from so_arm101_v2.contracts import act_to_mujoco_qpos, mujoco_qpos_to_act
from so_arm101_v2.contracts.coordinates import JOINT_NAMES, _MENAGERIE_MECHANICAL_HIGH, _MENAGERIE_MECHANICAL_LOW
from so_arm101_v2.contracts.joint_map import LEGACY_JOINT_MAP, load_joint_map
from so_arm101_v2.contracts.physical import load_physical_calibration, physical_normalized_to_act
from so_arm101_v2.data.resources import read_resource_bytes

MEASURED_RESOURCE_SHA256 = "6afc2d93235394f8fa27f8a188dbf1066d81cec1d18ee0788dc44f4c51e4d8be"
# Read-only encoder readings of 2026-09-07 (artifacts/.../joint_references/).
ZERO_REFERENCE_NORMALIZED = [0.49315068493149283, -5.782918149466184, 5.13513513513513, 2.40279598077764, 52.03907203907204, 6.83526999316473]
REST_JAWS_HORIZONTAL_NORMALIZED = [0.38356164383561975, -92.08185053380782, 100.0, 43.7308868501529, -3.2478632478632505, 12.918660287081341]


def test_measured_resource_is_pinned_and_matches_the_calibration():
    assert hashlib.sha256(read_resource_bytes("physical_joint_map_20260907.json")).hexdigest() == MEASURED_RESOURCE_SHA256
    resource = json.loads(read_resource_bytes("physical_joint_map_20260907.json"))
    calibration = {j.name: j for j in load_physical_calibration().joints}
    for name, joint in resource["joints"].items():
        assert (joint["range_min"], joint["range_max"], joint["drive_mode"]) == (calibration[name].range_min, calibration[name].range_max, calibration[name].drive_mode)
    assert resource["ticks_per_turn"] == 4096 and load_joint_map().resource_sha256 == MEASURED_RESOURCE_SHA256


def test_hand_held_zero_reference_maps_to_the_model_zero_pose():
    q = load_joint_map().act_to_mujoco(physical_normalized_to_act(ZERO_REFERENCE_NORMALIZED))
    assert np.allclose(np.degrees(q[:5]), 0.0, atol=0.05)


def test_rest_pose_is_upper_arm_up_forearm_forward_jaws_horizontal():
    q = np.degrees(load_joint_map().act_to_mujoco(physical_normalized_to_act(REST_JAWS_HORIZONTAL_NORMALIZED)))
    assert np.allclose(q[:5], [0.1, -85.3, 92.5, 41.6, -99.5], atol=0.1)
    legacy = np.degrees(LEGACY_JOINT_MAP.act_to_mujoco(physical_normalized_to_act(REST_JAWS_HORIZONTAL_NORMALIZED)))
    # The legacy map put three joints roughly a quarter turn away.
    assert abs(legacy[1] - q[1]) > 80 and abs(legacy[2] - q[2]) > 80 and abs(legacy[4] - q[4]) > 80


def test_scale_is_exactly_the_encoder_resolution():
    measured = load_joint_map(); calibration = {j.name: j for j in load_physical_calibration().joints}
    for i, name in enumerate(JOINT_NAMES[:5]):
        span_ticks = calibration[name].range_max - calibration[name].range_min
        a = np.zeros(6); b = np.zeros(6); a[i] = -50; b[i] = 50  # normalized units
        delta = abs(float(measured.act_to_mujoco(physical_normalized_to_act(b))[i] - measured.act_to_mujoco(physical_normalized_to_act(a))[i]))
        assert delta == pytest.approx(0.5 * span_ticks * 2 * np.pi / 4096, rel=1e-6)


def test_round_trip_and_gripper_channel_shared_with_legacy():
    measured = load_joint_map()
    act = physical_normalized_to_act([12.3, -45.6, 78.9, -3.2, 50.1, 33.3])
    assert np.allclose(measured.mujoco_to_act(measured.act_to_mujoco(act)), act, atol=1e-6)
    assert measured.act_to_mujoco(act)[5] == LEGACY_JOINT_MAP.act_to_mujoco(act)[5]
    batch = np.stack([act, act * 0.5])
    assert measured.act_to_mujoco(batch).shape == (2, 6) and measured.act_to_mujoco(batch).dtype == np.float32


def test_legacy_map_is_bit_identical_to_the_module_functions():
    act = physical_normalized_to_act([12.3, -45.6, 78.9, -3.2, 50.1, 33.3])
    assert np.array_equal(LEGACY_JOINT_MAP.act_to_mujoco(act), act_to_mujoco_qpos(act))
    assert np.array_equal(LEGACY_JOINT_MAP.mujoco_to_act(act_to_mujoco_qpos(act)), mujoco_qpos_to_act(act_to_mujoco_qpos(act)))
    assert LEGACY_JOINT_MAP.legacy and not load_joint_map().legacy


def test_measured_bounds_are_physical_range_within_mechanical_limits():
    measured = load_joint_map()
    assert np.all(measured.mujoco_low >= _MENAGERIE_MECHANICAL_LOW - 1e-6) and np.all(measured.mujoco_high <= _MENAGERIE_MECHANICAL_HIGH + 1e-6)
    # The physical elbow cannot fold past its calibrated maximum, which is the gravity-rest pose.
    assert np.degrees(measured.mujoco_high[2]) == pytest.approx(92.5, abs=0.1)
    with pytest.raises(ValueError, match="unknown joint map"):
        load_joint_map("made_up")
