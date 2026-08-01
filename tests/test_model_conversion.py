"""Permanent guards for the legacy -> Menagerie model conversion.

Re-verifies MODEL_CONVERSION_SIGN/OFFSET and the base alignment by comparing
physical invariants (joint axis lines, distal body COMs) between the legacy
model and the patched Menagerie model, and asserts the patched model is the
pristine upstream model plus only the documented additions/renames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so_arm101_v2.contracts import (
    JOINT_NAMES,
    MODEL_CONVERSION_OFFSET,
    MODEL_CONVERSION_SIGN,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCENE = REPOSITORY_ROOT / "simulation_code/model/scene.xml"
MENAGERIE_DIR = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100"
UPSTREAM_XML = MENAGERIE_DIR / "upstream/so_arm100.xml"
PATCHED_XML = MENAGERIE_DIR / "so_arm101_v2.xml"

UPSTREAM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
UPSTREAM_BODIES = (
    "Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw",
)
LEGACY_CHILD_BODIES = (
    "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1",
)
PATCHED_BODIES = (
    "Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "gripper", "moving_jaw_so101_v1",
)


@pytest.fixture(scope="module")
def models() -> dict:
    out = {}
    for key, path in (("legacy", LEGACY_SCENE), ("patched", PATCHED_XML), ("upstream", UPSTREAM_XML)):
        model = mujoco.MjModel.from_xml_path(str(path))
        out[key] = (model, mujoco.MjData(model))
    return out


def _fk(model, data, joint_names, q):
    mujoco.mj_resetData(model, data)
    for name, value in zip(joint_names, q, strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint >= 0, name
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)


def _joint_state(model, data, joint_names):
    axes, anchors = [], []
    for name in joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        axes.append(data.xaxis[joint].copy())
        anchors.append(data.xanchor[joint].copy())
    return axes, anchors


def _line_distance(p1, a1, p2):
    a1 = a1 / np.linalg.norm(a1)
    d = p2 - p1
    return float(np.linalg.norm(d - np.dot(d, a1) * a1))


def test_conversion_matches_physical_invariants(models) -> None:
    legacy_m, legacy_d = models["legacy"]
    patched_m, patched_d = models["patched"]
    low = np.array([legacy_m.jnt_range[mujoco.mj_name2id(legacy_m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                    for n in JOINT_NAMES])
    high = np.array([legacy_m.jnt_range[mujoco.mj_name2id(legacy_m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                     for n in JOINT_NAMES])
    rng = np.random.default_rng(0)
    worst_line = worst_axis = worst_com = 0.0
    for _ in range(40):
        q_legacy = rng.uniform(low, high)
        q_patched = MODEL_CONVERSION_SIGN * q_legacy + MODEL_CONVERSION_OFFSET
        _fk(legacy_m, legacy_d, JOINT_NAMES, q_legacy)
        _fk(patched_m, patched_d, JOINT_NAMES, q_patched)
        l_axes, l_anchors = _joint_state(legacy_m, legacy_d, JOINT_NAMES)
        p_axes, p_anchors = _joint_state(patched_m, patched_d, JOINT_NAMES)
        for i in range(6):
            worst_line = max(worst_line, _line_distance(l_anchors[i], l_axes[i], p_anchors[i]))
            cos = abs(np.dot(l_axes[i] / np.linalg.norm(l_axes[i]),
                             p_axes[i] / np.linalg.norm(p_axes[i])))
            worst_axis = max(worst_axis, float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        for legacy_body, patched_body in zip(LEGACY_CHILD_BODIES[4:], PATCHED_BODIES[4:], strict=True):
            l_com = legacy_d.xipos[legacy_m.body(legacy_body).id]
            p_com = patched_d.xipos[patched_m.body(patched_body).id]
            worst_com = max(worst_com, float(np.linalg.norm(l_com - p_com)))
    assert worst_line < 0.002, f"joint axis-line residual {worst_line * 1000:.2f}mm"
    assert worst_axis < 0.5, f"joint axis misalignment {worst_axis:.2f}deg"
    # COM residual reflects real SO-100 vs SO-101 CAD deltas; guard the bound.
    assert worst_com < 0.010, f"distal COM residual {worst_com * 1000:.2f}mm"


def test_patched_model_is_upstream_plus_documented_changes(models) -> None:
    upstream_m, _ = models["upstream"]
    patched_m, _ = models["patched"]
    # joint/actuator renames resolve, ranges identical
    for upstream_name, patched_name in zip(UPSTREAM_JOINTS, JOINT_NAMES, strict=True):
        uj = mujoco.mj_name2id(upstream_m, mujoco.mjtObj.mjOBJ_JOINT, upstream_name)
        pj = mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_JOINT, patched_name)
        assert uj >= 0 and pj >= 0
        np.testing.assert_allclose(upstream_m.jnt_range[uj], patched_m.jnt_range[pj])
        ua = mujoco.mj_name2id(upstream_m, mujoco.mjtObj.mjOBJ_ACTUATOR, upstream_name)
        pa = mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_ACTUATOR, patched_name)
        np.testing.assert_allclose(upstream_m.actuator_gainprm[ua], patched_m.actuator_gainprm[pa])
        np.testing.assert_allclose(upstream_m.actuator_forcerange[ua], patched_m.actuator_forcerange[pa])
    # bodies: identical inertials; local poses identical except the Base
    for upstream_name, patched_name in zip(UPSTREAM_BODIES, PATCHED_BODIES, strict=True):
        ub = upstream_m.body(upstream_name)
        pb = patched_m.body(patched_name)
        np.testing.assert_allclose(ub.mass, pb.mass)
        np.testing.assert_allclose(ub.inertia, pb.inertia)
        np.testing.assert_allclose(ub.pos, pb.pos)
        np.testing.assert_allclose(ub.quat, pb.quat)
    # documented additions: wrist camera + mount body/geom + two sites; no keyframes
    assert mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera") >= 0
    assert mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera_mount") >= 0
    assert mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_SITE, "fixed_jaw_tip") >= 0
    assert mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_SITE, "moving_jaw_tip") >= 0
    assert patched_m.nkey == 0
    assert patched_m.ngeom == upstream_m.ngeom + 1  # mount visual
    # fingertip pads keep their upstream names on both jaws
    for pad in ("fixed_jaw_pad_1", "fixed_jaw_pad_4", "moving_jaw_pad_1", "moving_jaw_pad_4"):
        assert mujoco.mj_name2id(patched_m, mujoco.mjtObj.mjOBJ_GEOM, pad) >= 0
