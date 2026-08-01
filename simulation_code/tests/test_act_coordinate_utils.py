from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from act_coordinate_utils import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    JOINT_NAMES,
    MUJOCO_JOINT_HIGH,
    MUJOCO_JOINT_LOW,
    act_to_mujoco_qpos,
    act_to_mujoco_qpos_torch,
    clip_mujoco_qpos,
    mujoco_qpos_to_act,
    mujoco_qpos_to_act_torch,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_DIR
    / "imitation-learning"
    / "datasets"
    / "so101_pickplace_v1"
    / "meta"
    / "act_coordinate_contract.json"
)


class ACTCoordinateUtilsTest(unittest.TestCase):
    def test_joint_order_is_the_recording_order(self) -> None:
        self.assertEqual(
            JOINT_NAMES,
            (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ),
        )

    def test_endpoints_map_exactly(self) -> None:
        np.testing.assert_allclose(mujoco_qpos_to_act(MUJOCO_JOINT_LOW), ACT_DATASET_LOW, atol=1e-6)
        np.testing.assert_allclose(mujoco_qpos_to_act(MUJOCO_JOINT_HIGH), ACT_DATASET_HIGH, atol=1e-6)
        np.testing.assert_allclose(act_to_mujoco_qpos(ACT_DATASET_LOW), MUJOCO_JOINT_LOW, atol=1e-6)
        np.testing.assert_allclose(act_to_mujoco_qpos(ACT_DATASET_HIGH), MUJOCO_JOINT_HIGH, atol=1e-6)

    def test_midpoints_and_gripper_scaling(self) -> None:
        mujoco_midpoint = (MUJOCO_JOINT_LOW + MUJOCO_JOINT_HIGH) / 2
        act_midpoint = mujoco_qpos_to_act(mujoco_midpoint)
        np.testing.assert_allclose(act_midpoint[:5], np.zeros(5), atol=1e-6)
        self.assertAlmostEqual(float(act_midpoint[5]), 0.85, places=6)

    def test_batched_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        fraction = rng.uniform(0.0, 1.0, size=(4, 30, 6)).astype(np.float32)
        qpos = MUJOCO_JOINT_LOW + fraction * (MUJOCO_JOINT_HIGH - MUJOCO_JOINT_LOW)
        np.testing.assert_allclose(act_to_mujoco_qpos(mujoco_qpos_to_act(qpos)), qpos, atol=1e-6)

    def test_physical_motor_encoding_matches_sim_round_trip(self) -> None:
        motor_values = np.array([-25.0, 30.0, -60.0, 90.0, 10.0, 25.0], dtype=np.float32)
        dataset_values = np.empty(6, dtype=np.float32)
        dataset_values[:5] = motor_values[:5] / 100.0 * np.pi
        dataset_values[5] = motor_values[5] / 100.0 * 1.7
        sim_qpos = act_to_mujoco_qpos(dataset_values)
        np.testing.assert_allclose(mujoco_qpos_to_act(sim_qpos), dataset_values, atol=1e-6)

    def test_representative_wrist_flex_no_longer_saturates(self) -> None:
        dataset_action = np.zeros(6, dtype=np.float32)
        dataset_action[3] = 2.8487709
        dataset_action[5] = 0.3197835
        qpos = act_to_mujoco_qpos(dataset_action)
        self.assertLess(float(qpos[3]), float(MUJOCO_JOINT_HIGH[3]))
        self.assertAlmostEqual(float(qpos[3]), 1.50356, places=4)
        _clipped, mask = clip_mujoco_qpos(qpos)
        self.assertFalse(bool(mask.any()))

    def test_torch_mapping_is_differentiable(self) -> None:
        act_values = torch.zeros((2, 30, 6), dtype=torch.float32, requires_grad=True)
        qpos = act_to_mujoco_qpos_torch(act_values)
        round_trip = mujoco_qpos_to_act_torch(qpos)
        torch.testing.assert_close(round_trip, act_values)
        qpos.sum().backward()
        self.assertIsNotNone(act_values.grad)
        self.assertTrue(torch.isfinite(act_values.grad).all())

    def test_pinned_physical_calibration_has_matching_joint_contract(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        calibration_path = CONTRACT_PATH.parent / contract["current_physical_inference_calibration"]["pinned_copy"]
        calibration_bytes = calibration_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(calibration_bytes).hexdigest(),
            contract["current_physical_inference_calibration"]["pinned_copy_sha256"],
        )
        calibration = json.loads(calibration_bytes)
        self.assertEqual(tuple(calibration), JOINT_NAMES)
        self.assertTrue(all(calibration[name]["drive_mode"] == 0 for name in JOINT_NAMES))
        self.assertTrue(
            all(calibration[name]["range_min"] < calibration[name]["range_max"] for name in JOINT_NAMES)
        )

    def test_current_calibration_raw_endpoints_map_to_mujoco_endpoints(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        calibration_path = CONTRACT_PATH.parent / contract["current_physical_inference_calibration"]["pinned_copy"]
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

        encoded_low = np.empty(6, dtype=np.float32)
        encoded_high = np.empty(6, dtype=np.float32)
        for index, name in enumerate(JOINT_NAMES):
            motor = calibration[name]
            raw_values = np.array([motor["range_min"], motor["range_max"]], dtype=np.float32)
            if index < 5:
                normalized = (raw_values - motor["range_min"]) / (
                    motor["range_max"] - motor["range_min"]
                ) * 200.0 - 100.0
                encoded = normalized / 100.0 * np.pi
            else:
                normalized = (raw_values - motor["range_min"]) / (
                    motor["range_max"] - motor["range_min"]
                ) * 100.0
                encoded = normalized / 100.0 * 1.7
            encoded_low[index], encoded_high[index] = encoded

        np.testing.assert_allclose(act_to_mujoco_qpos(encoded_low), MUJOCO_JOINT_LOW, atol=1e-6)
        np.testing.assert_allclose(act_to_mujoco_qpos(encoded_high), MUJOCO_JOINT_HIGH, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
