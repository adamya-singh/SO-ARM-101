"""Per-arm gripper geometry the scripted teacher and the grasp detector need (2026-11-06, handoff step 3).

The bench ran on Menagerie's SO-100 (trs_so_arm100) until the Menagerie SO-101 (robotstudio_so101, vendored as
simulation_code/model/menagerie_so101/so101_bench.xml) arrived. Both share joint names, the body names ``gripper`` /
``moving_jaw_so101_v1``, the pad names ``fixed_jaw_pad_N`` / ``moving_jaw_pad_N`` and the tip sites, but their gripper
body frames differ: on the SO-100 the jaw mouth opens toward -x of the gripper frame and the jaw depth (palm -> tips)
runs along -y; on the SO-101 the mouth opens toward +x and the depth runs along -z. The SO-101 also has only two
moving-jaw pads (tip pads 1 and 2), so its detector closing axis uses pad 1.

The profile is detected from the compiled model: the vendored SO-101 carries the marker site
``arm_profile_so101_menagerie``. Nothing in a bench config names the arm, so every SO-100 suite id, capture digest and
detector identity is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SO101_MARKER_SITE = "arm_profile_so101_menagerie"


@dataclass(frozen=True)
class ArmProfile:
    name: str
    # Gripper-frame unit vectors: the fixed pad's inner normal (toward the mouth) and the jaw depth (palm -> tips).
    pad_normal_local: tuple[float, float, float]
    depth_local: tuple[float, float, float]
    # The pad pair whose x axes give the detector's closing axis.
    closing_pads: tuple[str, str]
    # Suffix folded into the detector identity ("" keeps the SO-100's historical identity).
    detector_suffix: str

    @property
    def mouth_sign(self) -> float:
        """+1 when the mouth opens toward +x of the gripper frame, -1 toward -x."""
        return float(np.sign(self.pad_normal_local[0]))


SO100_MENAGERIE = ArmProfile(name="so100_menagerie", pad_normal_local=(-1.0, 0.0, 0.0), depth_local=(0.0, -1.0, 0.0),
                             closing_pads=("fixed_jaw_pad_4", "moving_jaw_pad_4"), detector_suffix="")
SO101_MENAGERIE = ArmProfile(name="so101_menagerie", pad_normal_local=(1.0, 0.0, 0.0), depth_local=(0.0, 0.0, -1.0),
                             closing_pads=("fixed_jaw_pad_1", "moving_jaw_pad_1"), detector_suffix="+so101_menagerie")


def arm_profile(model: Any) -> ArmProfile:
    """The profile of a compiled MjModel (SO-101 when the vendored marker site is present, else the SO-100)."""
    import mujoco
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SO101_MARKER_SITE) >= 0:
        return SO101_MENAGERIE
    return SO100_MENAGERIE


def scene_arm_profile(model_path: Any) -> ArmProfile:
    """Profile of a scene file (compiles it; for provenance records, not per-step use)."""
    import mujoco
    return arm_profile(mujoco.MjModel.from_xml_path(str(model_path)))


def grasp_detector_version(profile: ArmProfile) -> str:
    from .contact import GRASP_DETECTOR_VERSION
    return GRASP_DETECTOR_VERSION + profile.detector_suffix


__all__ = ["ArmProfile", "SO100_MENAGERIE", "SO101_MENAGERIE", "SO101_MARKER_SITE", "arm_profile", "grasp_detector_version",
           "scene_arm_profile"]
