import unittest

import numpy as np

from so101_mujoco_utils import (
    evaluate_face_grasp_contacts,
    evaluate_pregrasp_face_guidance,
)


def contact(side, position, normal, force=0.2):
    return {
        "side": side,
        "position_local": np.asarray(position, dtype=np.float64),
        "normal_local": np.asarray(normal, dtype=np.float64),
        "force": force,
    }


class FaceGraspDetectorTest(unittest.TestCase):
    def test_valid_opposing_face_grip(self):
        contacts = [
            # Deliberately reverse one normal to verify robust orientation.
            contact("fixed", (-0.0125, 0.0, 0.0), (1.0, 0.0, 0.0)),
            contact("moving", (0.0125, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ]

        gripped, force, diagnostics = evaluate_face_grasp_contacts(
            contacts, np.array([1.0, 0.0, 0.0])
        )

        self.assertTrue(gripped)
        self.assertAlmostEqual(force, 0.2)
        self.assertAlmostEqual(diagnostics["face_alignment"], 1.0)
        self.assertAlmostEqual(diagnostics["face_opposition"], 1.0)
        self.assertFalse(diagnostics["face_corner_rejection"])
        self.assertTrue(diagnostics["interior_face_contact"])
        self.assertEqual(diagnostics["interior_face_contact_count"], 2)

    def test_corner_pinch_is_rejected(self):
        contacts = [
            contact("fixed", (-0.0125, 0.009, 0.0), (-1.0, 0.0, 0.0)),
            contact("moving", (0.0125, 0.009, 0.0), (1.0, 0.0, 0.0)),
        ]

        gripped, force, diagnostics = evaluate_face_grasp_contacts(
            contacts, np.array([1.0, 0.0, 0.0])
        )

        self.assertFalse(gripped)
        self.assertEqual(force, 0.0)
        self.assertTrue(diagnostics["face_corner_rejection"])
        self.assertEqual(diagnostics["face_corner_rejection_count"], 2)

    def test_one_sided_contact_is_rejected(self):
        gripped, force, _ = evaluate_face_grasp_contacts(
            [contact("fixed", (-0.0125, 0.0, 0.0), (-1.0, 0.0, 0.0))],
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertFalse(gripped)
        self.assertEqual(force, 0.0)

    def test_misaligned_normals_are_rejected(self):
        contacts = [
            contact("fixed", (-0.0125, 0.0, 0.0), (0.0, 1.0, 0.0)),
            contact("moving", (0.0125, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ]

        gripped, force, diagnostics = evaluate_face_grasp_contacts(
            contacts, np.array([1.0, 0.0, 0.0])
        )

        self.assertFalse(gripped)
        self.assertEqual(force, 0.0)
        self.assertLess(diagnostics["face_alignment"], 0.5)

    def test_pregrasp_guidance_prefers_centered_face_axis(self):
        centered = evaluate_pregrasp_face_guidance(
            np.eye(3),
            np.zeros(3),
            np.array([-0.02, 0.0, 0.0]),
            np.array([0.02, 0.0, 0.0]),
        )
        diagonal = evaluate_pregrasp_face_guidance(
            np.eye(3),
            np.zeros(3),
            np.array([-0.02, -0.02, 0.0]),
            np.array([0.02, 0.02, 0.0]),
        )
        off_center = evaluate_pregrasp_face_guidance(
            np.eye(3),
            np.zeros(3),
            np.array([-0.02, 0.02, 0.015]),
            np.array([0.02, 0.02, 0.015]),
        )

        self.assertAlmostEqual(centered["pregrasp_face_axis_alignment"], 1.0)
        self.assertAlmostEqual(centered["pregrasp_face_guidance_score"], 1.0)
        self.assertAlmostEqual(diagonal["pregrasp_face_guidance_score"], 0.0)
        self.assertGreater(
            centered["pregrasp_face_guidance_score"],
            off_center["pregrasp_face_guidance_score"],
        )
        self.assertAlmostEqual(off_center["pregrasp_face_depth_error"], 0.02)
        self.assertAlmostEqual(off_center["pregrasp_face_height_error"], 0.015)

    def test_single_interior_face_contact_is_diagnosed(self):
        _, _, diagnostics = evaluate_face_grasp_contacts(
            [contact("fixed", (-0.0125, 0.0, 0.0), (-1.0, 0.0, 0.0))],
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertTrue(diagnostics["interior_face_contact"])
        self.assertAlmostEqual(diagnostics["interior_face_contact_score"], 1.0)

    def test_corner_rejection_coexists_with_interior_contact(self):
        corner_contacts = [
            contact("fixed", (-0.0125, 0.009, 0.0), (-1.0, 0.0, 0.0)),
        ]
        _, _, corner_diagnostics = evaluate_face_grasp_contacts(
            corner_contacts,
            np.array([1.0, 0.0, 0.0]),
        )
        self.assertTrue(corner_diagnostics["face_corner_rejection"])
        self.assertFalse(corner_diagnostics["interior_face_contact"])

        mixed_contacts = corner_contacts + [
            contact("moving", (0.0125, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ]
        _, _, mixed_diagnostics = evaluate_face_grasp_contacts(
            mixed_contacts,
            np.array([1.0, 0.0, 0.0]),
        )
        self.assertTrue(mixed_diagnostics["face_corner_rejection"])
        self.assertTrue(mixed_diagnostics["interior_face_contact"])

    def test_face_contact_quality_increases_with_edge_clearance(self):
        qualities = []
        for tangent in (0.0125, 0.0115, 0.0105, 0.0085):
            _, _, diagnostics = evaluate_face_grasp_contacts(
                [
                    contact(
                        "fixed",
                        (-0.0125, tangent, 0.0),
                        (-1.0, 0.0, 0.0),
                    )
                ],
                np.array([1.0, 0.0, 0.0]),
            )
            qualities.append(diagnostics["face_contact_quality"])

        self.assertEqual(qualities[0], 0.0)
        self.assertGreater(qualities[1], qualities[0])
        self.assertGreater(qualities[2], qualities[1])
        self.assertAlmostEqual(qualities[3], 1.0)

    def test_bilateral_interior_precedes_strict_opposition_gate(self):
        angle = np.deg2rad(24.0)
        contacts = [
            contact(
                "fixed",
                (-0.0125, 0.0, 0.0),
                (-np.cos(angle), np.sin(angle), 0.0),
            ),
            contact(
                "moving",
                (0.0125, 0.0, 0.0),
                (np.cos(angle), np.sin(angle), 0.0),
            ),
        ]
        gripped, _, diagnostics = evaluate_face_grasp_contacts(
            contacts,
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertFalse(gripped)
        self.assertTrue(diagnostics["fixed_interior_face_contact"])
        self.assertTrue(diagnostics["moving_interior_face_contact"])
        self.assertTrue(diagnostics["bilateral_interior_face_contact"])
        self.assertGreater(diagnostics["bilateral_opposition_quality"], 0.0)
        self.assertLess(diagnostics["bilateral_opposition_quality"], np.cos(np.deg2rad(25.0)))


if __name__ == "__main__":
    unittest.main()
