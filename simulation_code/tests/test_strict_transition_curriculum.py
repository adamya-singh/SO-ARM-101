import unittest
from unittest import mock

import mujoco
import numpy as np

import so101_mujoco_utils as rewards
import train_act_in_sim as train
from run_strict_transition_9h_pipeline import adaptive_transition, strictly_better_precursor


STRICT_DIAGNOSTICS = {
    "face_alignment": 1.0, "face_opposition": 1.0, "face_jaw_axis_alignment": 1.0,
    "face_corner_rejection": False, "face_corner_rejection_count": 0,
    "interior_face_contact": True, "interior_face_contact_score": 1.0,
    "interior_face_contact_count": 2, "face_contact_quality": 1.0,
    "fixed_interior_face_contact": True, "moving_interior_face_contact": True,
    "bilateral_interior_face_contact": True, "bilateral_opposition_quality": 1.0,
}


class StrictTransitionRewardTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(train.SCRIPT_DIR / "model/scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[6:9] = [0.0, 0.24, 0.0125]
        self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self.state = rewards.create_reward_state_tracker()
        self.kwargs = {
            "strict_transition": True,
            "pregrasp_alignment_reward_scale": 0.0,
            "aligned_close_reward": 0.0,
            "jaw_centered_contact_reward": 0.0,
            "contact_persistence_reward": 0.0,
        }

    def step(self, height, strict=True, corner=False):
        diagnostics = dict(STRICT_DIAGNOSTICS)
        if corner:
            diagnostics.update(interior_face_contact=False,
                               bilateral_interior_face_contact=False,
                               face_corner_rejection=True)
        self.data.qpos[8] = height
        mujoco.mj_forward(self.model, self.data)
        with mock.patch.object(rewards, "check_gripper_block_contact", return_value=True), \
             mock.patch.object(rewards, "check_block_gripped_with_force", return_value=(True, 2.0)), \
             mock.patch.object(rewards, "check_block_face_gripped",
                               return_value=(strict, 2.0 if strict else 0.0, diagnostics)):
            return rewards.compute_pickup_reward_from_state(
                self.model, self.data, self.state, **self.kwargs
            )

    def test_thresholds_fire_once_and_held_height_has_no_lift_reward(self):
        for _ in range(5):
            self.step(0.0125)
        _, _, micro = self.step(0.0180)
        _, _, held = self.step(0.0180)
        _, _, lift = self.step(0.0220)
        _, done, success = self.step(0.0230)
        _, done_again, held_success = self.step(0.0230)
        self.assertEqual(micro["micro_lift_bonus"], 0.8)
        self.assertEqual(held["micro_lift_bonus"], 0.0)
        self.assertEqual(held["lift_progress_reward"], 0.0)
        self.assertEqual(lift["lift_bonus_reward"], 2.0)
        self.assertTrue(done)
        self.assertEqual(success["success_lift_bonus"], 6.0)
        self.assertFalse(done_again)
        self.assertEqual(held_success["success_lift_bonus"], 0.0)

    def test_corner_contact_cannot_enable_lift_or_success(self):
        for _ in range(5):
            self.step(0.0125, strict=False, corner=True)
        _, done, metrics = self.step(0.0240, strict=False, corner=True)
        self.assertFalse(done)
        self.assertEqual(metrics["lift_progress_reward"], 0.0)
        self.assertEqual(metrics["lift_bonus_reward"], 0.0)
        self.assertEqual(metrics["success_lift_bonus"], 0.0)
        self.assertEqual(metrics["corner_only_contact_penalty"], -0.08)

    def test_success_requires_current_five_step_strict_grasp_at_crossing(self):
        for _ in range(4):
            self.step(0.0125)
        _, done, metrics = self.step(0.0240)
        self.assertTrue(done)
        self.assertEqual(metrics["strict_grasp_streak"], 5)
        # The fifth strict step coincides with a crossing, so it is permitted.
        # Four strict steps would not be.
        state = rewards.create_reward_state_tracker()
        self.state = state
        for _ in range(3):
            self.step(0.0125)
        _, done, _ = self.step(0.0240)
        self.assertFalse(done)

    def test_strict_telemetry_is_profile_independent_and_one_shot(self):
        self.kwargs["strict_transition"] = False
        for _ in range(5):
            self.step(0.0125)
        _, baseline_done, crossed = self.step(0.0240)
        _, held_done, held = self.step(0.0240)
        self.assertTrue(baseline_done)
        self.assertTrue(held_done)  # Legacy baseline termination remains unchanged.
        self.assertTrue(crossed["independent_strict_success_crossed"])
        self.assertTrue(crossed["strict_lift_success"])
        self.assertFalse(held["independent_strict_success_crossed"])
        self.assertFalse(held["strict_lift_success"])


class ResetCurriculumTest(unittest.TestCase):
    def test_mixed_ratios_and_boundaries(self):
        self.assertEqual(train.reset_stage_probabilities("mixed", 0, "lift"),
                         {"normal": .25, "pregrasp": .35, "grasped": .40})
        self.assertEqual(train.reset_stage_probabilities("mixed", 30, "lift"),
                         {"normal": .50, "pregrasp": .35, "grasped": .15})
        self.assertEqual(train.reset_stage_probabilities("mixed", 60, "lift"),
                         {"normal": .75, "pregrasp": .25, "grasped": 0.0})

class AdaptiveControllerTest(unittest.TestCase):
    def test_zero_metrics_neither_advance_nor_replace(self):
        zero = {}
        self.assertEqual(adaptive_transition("lift", zero, 0), ("lift", 0))
        self.assertFalse(strictly_better_precursor(zero, zero))

    def test_advancement_gates_and_two_full_successes(self):
        self.assertEqual(
            adaptive_transition("lift", {"strict_success_rate": .20}, 0), ("grasp", 0)
        )
        self.assertEqual(
            adaptive_transition(
                "grasp", {"sustained_grasp_rate": .10, "strict_lift_rate": .05}, 0
            ),
            ("full", 0),
        )
        self.assertEqual(
            adaptive_transition("full", {"strict_success_rate": .05}, 0), ("full", 1)
        )
        self.assertEqual(
            adaptive_transition("full", {"strict_success_rate": .05}, 1), ("normal", 2)
        )


if __name__ == "__main__":
    unittest.main()
