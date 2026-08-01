import inspect
import unittest

import so101_mujoco_utils as utils
import train_act_in_sim as train


class ACTRewardProfilesTest(unittest.TestCase):
    def test_legacy_profiles_are_exact(self):
        legacy = {key: train.ACT_REWARD_PROFILES[key] for key in (
            "baseline", "jaw_quality", "jaw_quality_rebalanced"
        )}
        self.assertEqual(legacy, {
            "baseline": {},
            "jaw_quality": {
                "pregrasp_alignment_reward_scale": 0.16,
                "aligned_close_reward": 0.14,
                "jaw_centered_contact_reward": 0.16,
                "recent_jaw_centered_contact_window": 20,
            },
            "jaw_quality_rebalanced": {
                "pregrasp_alignment_reward_scale": 0.16,
                "aligned_close_reward": 0.14,
                "jaw_centered_contact_reward": 0.16,
                "recent_jaw_centered_contact_window": 20,
                "bilateral_grasp_bonus": 0.8,
                "grasp_persistence_reward": 0.25,
                "side_push_penalty": -0.06,
                "block_displacement_penalty_scale": 0.45,
            },
        })

    def test_strict_transition_profile_is_exact(self):
        self.assertEqual(train.ACT_REWARD_PROFILES["strict_transition"], {
            "strict_transition": True,
            "strict_grasp_required_steps": 5,
            "pregrasp_alignment_reward_scale": 0.0,
            "aligned_close_reward": 0.0,
            "jaw_centered_contact_reward": 0.0,
            "contact_persistence_reward": 0.0,
            "bilateral_grasp_bonus": 1.10,
            "grasp_persistence_reward": 0.10,
            "grasped_vertical_lift_reward_scale": 40.0,
            "grasped_vertical_lift_reward_cap": 0.40,
            "micro_lift_bonus": 0.80,
            "lift_bonus": 2.0,
            "success_lift_bonus": 6.0,
            "pregrasp_potential_scale": 0.20,
            "interior_contact_potential_scale": 0.30,
            "bilateral_opposition_potential_scale": 0.50,
            "corner_only_contact_penalty": -0.08,
        })

    def test_strict_profiles_are_unavailable(self):
        strict_profiles = (
            "strict_face_baseline",
            "strict_face_quality",
            "strict_face_rebalanced",
            "strict_face_axis",
            "strict_face_axis_penalized",
            "strict_face_axis_rebalanced",
            "strict_face_contact_curriculum",
            "strict_face_contact_rebalanced",
            "strict_face_pair_curriculum",
            "strict_face_pair_rebalanced",
        )
        for profile in strict_profiles:
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(ValueError, "Unknown reward profile"):
                    train.resolve_reward_profile(profile)

    def test_profile_resolution_returns_a_copy(self):
        profile = train.resolve_reward_profile("jaw_quality")
        profile["aligned_close_reward"] = 999
        self.assertEqual(train.resolve_reward_profile("jaw_quality")["aligned_close_reward"], 0.14)

    def test_default_reward_behavior_remains_legacy(self):
        defaults = inspect.signature(utils.compute_reward).parameters
        self.assertEqual(defaults["pregrasp_alignment_reward_scale"].default, 0.12)
        self.assertEqual(defaults["aligned_close_reward"].default, 0.08)
        self.assertEqual(defaults["jaw_centered_contact_reward"].default, 0.08)
        self.assertEqual(defaults["bilateral_grasp_bonus"].default, 1.10)
        self.assertEqual(defaults["grasp_persistence_reward"].default, 0.45)
        retired_parameters = {
            "strict_face_grasp",
            "face_axis_guidance_reward_scale",
            "force_only_grasp_penalty",
            "interior_face_contact_reward",
            "face_contact_quality_reward",
            "bilateral_interior_contact_reward",
        }
        self.assertTrue(retired_parameters.isdisjoint(defaults))
        stateful_defaults = inspect.signature(
            utils.compute_pickup_reward_from_state
        ).parameters
        self.assertTrue(retired_parameters.isdisjoint(stateful_defaults))

        reward_source = inspect.getsource(utils.compute_pickup_reward_from_state)
        self.assertIn("gripped = face_gripped if strict_transition else force_gripped", reward_source)
        self.assertIn("grip_force = face_grip_force if strict_transition else force_grip_force", reward_source)
        self.assertNotIn("reward += face_axis_guidance", reward_source)
        for helper in (
            "allow_jaw_centered_contact_reward",
            "compute_guidance_potential_reward",
            "compute_face_contact_curriculum_reward",
        ):
            self.assertFalse(hasattr(utils, helper))

    def test_unknown_profile_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown reward profile"):
            train.resolve_reward_profile("not-a-profile")


if __name__ == "__main__":
    unittest.main()
