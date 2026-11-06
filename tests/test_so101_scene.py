"""Menagerie SO-101 bench scenes (2026-11-06, notes/handoff-20261105.md step 3)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT

MODELS = REPOSITORY_ROOT / "simulation_code/model"
SCENE_SO100 = MODELS / "bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
SCENE_SERVO = MODELS / "bench_pick_replace_v1_servo/scene_bench_pick_replace_v1.xml"
SCENE_SO101 = MODELS / "bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml"
SCENE_SO101_MENAGERIE = MODELS / "bench_pick_replace_v1_so101_menagerie/scene_bench_pick_replace_v1.xml"
PY = sys.executable


def _run(*args):
    return subprocess.run([PY, *args], cwd=REPOSITORY_ROOT, capture_output=True, text=True,
                          env={"PYTHONPATH": str(REPOSITORY_ROOT / "src"), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"})


def test_generated_files_are_current():
    pytest.importorskip("mujoco")
    for tool in ("tools/vendor_menagerie_so101.py", "tools/derive_joint_map_so101.py"):
        result = _run(tool, "--check")
        assert result.returncode == 0, (tool, result.stdout, result.stderr[-500:])


def test_profiles_are_detected_and_so100_constants_are_unchanged():
    pytest.importorskip("mujoco")
    import mujoco
    from so_arm101_v2.simulation import privileged
    from so_arm101_v2.simulation.arm_profiles import SO100_MENAGERIE, SO101_MENAGERIE, arm_profile, grasp_detector_version
    assert [arm_profile(mujoco.MjModel.from_xml_path(str(p))).name for p in (SCENE_SO100, SCENE_SERVO, SCENE_SO101, SCENE_SO101_MENAGERIE)] == \
        ["so100_menagerie", "so100_menagerie", "so101_menagerie", "so101_menagerie"]
    assert grasp_detector_version(SO100_MENAGERIE) == "pad_normals_v2"
    assert grasp_detector_version(SO101_MENAGERIE) == "pad_normals_v2+so101_menagerie"
    assert tuple(privileged.PAD_NORMAL_LOCAL) == SO100_MENAGERIE.pad_normal_local
    assert tuple(privileged.DEPTH_LOCAL) == SO100_MENAGERIE.depth_local


def test_joint_map_is_the_so100_map_in_so101_conventions():
    from so_arm101_v2.contracts.coordinates import MODEL_CONVERSION_OFFSET, MODEL_CONVERSION_SIGN
    from so_arm101_v2.contracts.joint_map import load_joint_map
    so100, so101 = load_joint_map("measured_20260908b"), load_joint_map("measured_20260908b_so101")
    rng = np.random.default_rng(1)
    act = rng.uniform(-2.5, 2.5, (300, 6)); act[:, 5] = rng.uniform(0.0, 1.6, 300)
    q100 = so100.act_to_mujoco(act).astype(np.float64)
    q101 = so101.act_to_mujoco(act).astype(np.float64)
    assert np.abs(MODEL_CONVERSION_SIGN * q101 + MODEL_CONVERSION_OFFSET - q100).max() < 2e-6
    assert np.abs(so101.mujoco_to_act(q101) - act).max() < 5e-6


def _camera_and_tips(path):
    import mujoco
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    adapter = MujocoTaskAdapter(path)
    try:
        out = {}
        for label, qpos in (("reset", adapter.bench.reset_qpos), ("survey", adapter.bench.viewing_qpos)):
            for address, value in zip(adapter._joint_qpos, qpos):
                adapter.data.qpos[address] = value
            mujoco.mj_kinematics(adapter.model, adapter.data)
            mujoco.mj_camlight(adapter.model, adapter.data)
            camera = adapter.model.camera("wrist_camera").id
            out[label] = (adapter.data.cam_xpos[camera].copy(), adapter.data.cam_xmat[camera].reshape(3, 3).copy(),
                          adapter.data.site("fixed_jaw_tip").xpos.copy(), adapter.data.site("moving_jaw_tip").xpos.copy())
        return out
    finally:
        adapter.close()


@pytest.mark.parametrize("scene", [SCENE_SO101, SCENE_SO101_MENAGERIE])
def test_camera_and_tip_sites_sit_where_they_did_on_the_so100(scene):
    pytest.importorskip("mujoco")
    reference, got = _camera_and_tips(SCENE_SO100), _camera_and_tips(scene)
    for label in ("reset", "survey"):
        (rc, rR, rf, rm), (gc, gR, gf, gm) = reference[label], got[label]
        angle = np.degrees(np.arccos(np.clip((np.trace(rR.T @ gR) - 1) / 2, -1, 1)))
        assert np.linalg.norm(gc - rc) < 1.5e-3 and angle < 0.5, label
        assert np.linalg.norm(gf - rf) < 1.5e-3 and np.linalg.norm(gm - rm) < 1.5e-3, label


def _without_paths(path: Path) -> str:
    root = ET.parse(path).getroot()
    root.find("compiler").attrib.pop("meshdir", None)
    for mesh in root.iter("mesh"):
        if mesh.get("name") == "xyz_cube_visual":
            mesh.attrib.pop("file", None)
    return ET.tostring(root, encoding="unicode")


@pytest.mark.parametrize("scene,args", [(SCENE_SO100, []), (SCENE_SERVO, []), (SCENE_SO101, ["--arm", "so101"]),
                                        (SCENE_SO101_MENAGERIE, ["--arm", "so101", "--dynamics", "menagerie"])])
def test_scenes_regenerate_from_the_generator(tmp_path, scene, args):
    pytest.importorskip("mujoco")
    source_config = SCENE_SERVO.parent / "bench_config.json" if scene == SCENE_SERVO else SCENE_SO100.parent / "bench_config.json"
    result = _run("tools/prepare_bench_scene.py", "--bench-config", str(source_config), "--output-dir", str(tmp_path), *args)
    assert result.returncode == 0, result.stderr[-800:]
    assert _without_paths(tmp_path / scene.name) == _without_paths(scene)
    assert (tmp_path / "bench_config.json").read_bytes() == (scene.parent / "bench_config.json").read_bytes()


@pytest.mark.parametrize("scene", [SCENE_SO101, SCENE_SO101_MENAGERIE])
def test_teacher_completes_the_task_with_strict_grasps_on_the_so101(scene):
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import certification_suite
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    adapter = MujocoTaskAdapter(scene)
    try:
        assert adapter.arm_profile.name == "so101_menagerie"
        contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
        adapter.reset(certification_suite(adapter.bench, repeats=1).scenarios[0])
        teacher = PrivilegedStagedController(); teacher.reset(adapter)
        state, strict, safety, result = PickPlaceEvaluationState(), 0, 0, None
        for _ in range(contract.max_actions):
            command = adapter.apply_policy_command(teacher.predict(None, adapter.current_act(), adapter))
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(command)
            p = measurement.pickup
            strict += p.strict_bilateral_grasp
            safety += p.unsafe_contact + p.command_bound_violation + p.delta_limiter_activated + p.nonfinite_command
            state, result = evaluate_pick_place_step(contract, measurement, state)
            if result.terminated or result.truncated or result.invalidated:
                break
        assert result.success and not result.invalidated and safety == 0 and strict >= 100
    finally:
        adapter.close()


def test_so100_contract_is_unchanged_and_so101_contract_uses_its_envelope():
    from dataclasses import asdict
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.contracts.coordinates import MUJOCO_JOINT_HIGH
    from so_arm101_v2.contracts.pick_place import load_pick_place_contract
    so100 = load_pick_place_contract("bench_pick_replace_v1", bench_config=scene_bench_config(SCENE_SO100))
    so101_config = scene_bench_config(SCENE_SO101)
    so101 = load_pick_place_contract("bench_pick_replace_v1", bench_config=so101_config)
    assert np.allclose(so100.pickup.safety.mujoco_joint_high, MUJOCO_JOINT_HIGH)
    assert np.allclose(so101.pickup.safety.mujoco_joint_high, so101_config.mujoco_high)
    assert json.dumps(asdict(so100), sort_keys=True, default=str) != json.dumps(asdict(so101), sort_keys=True, default=str)
