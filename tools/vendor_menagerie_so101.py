"""Generate the bench's SO-101 arm (simulation_code/model/menagerie_so101/so101_bench.xml) from the pristine Menagerie file.

Handoff step 3 (notes/handoff-20261105.md): the simulator moves from Menagerie's SO-100 to Menagerie's SO-101
(robotstudio_so101, box/sphere collision pads). This script is the single record of every patch; VENDOR.md explains
them and the numbers it measured. Patches:

1. base yawed 90 deg into the bench world frame (the legacy so101_new_calib.xml base quat): the SO-101's joint
   conventions then equal the legacy ones, so the repo's SIGN/OFFSET conversion to the SO-100 holds exactly;
2. the tip pads renamed to the code's names: fixed_jaw_box4..7 -> fixed_jaw_pad_1..4, moving_jaw_box2/3 ->
   moving_jaw_pad_1/2 (collision unchanged);
3. sites fixed_jaw_tip / moving_jaw_tip and the camera mount body (wrist_camera_mount, the repo's mount mesh as a
   visual, camera wrist_camera) placed at the SAME physical pose as on the SO-100 bench model: the SO-100 declared
   poses are carried through the rigid transform between the two models' gripper (and moving-jaw) frames at matching
   joint angles, so the closure metric, the real-arm jaw-height guard and the hand-eye calibrated camera keep their
   meaning across arms;
4. Menagerie's own camera_mount body (camera, PCB, two collision boxes) and its two unused mesh assets removed;
5. the part materials mapped onto the bench's 'white' (printed parts) and 'black' (servos) materials that the
   appearance randomization and the scene builder resolve by name;
6. meshdir points at the repo's shared simulation_code/model/assets; a marker site arm_profile_so101_menagerie
   identifies the arm to the code (simulation/arm_profiles.py).

Physics (option, defaults, actuators, collision classes) is untouched here; tools/prepare_bench_scene.py --dynamics
chooses between the bench's settings and Menagerie's.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/vendor_menagerie_so101.py [--check]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VENDOR = ROOT / "simulation_code/model/menagerie_so101"
UPSTREAM = VENDOR / "upstream/so101.xml"
OUTPUT = VENDOR / "so101_bench.xml"
SO100 = ROOT / "simulation_code/model/menagerie_so_arm100"
BASE_QUAT = "0.7071068 0 0 0.7071068"          # legacy so101_new_calib.xml:46
PAD_RENAMES = {"fixed_jaw_box4": "fixed_jaw_pad_1", "fixed_jaw_box5": "fixed_jaw_pad_2", "fixed_jaw_box6": "fixed_jaw_pad_3",
               "fixed_jaw_box7": "fixed_jaw_pad_4", "moving_jaw_box2": "moving_jaw_pad_1", "moving_jaw_box3": "moving_jaw_pad_2"}
UNUSED_MESHES = ("moving_jaw_so101_gripper_v1", "wrist_roll_follower_so101_camera_mount")
MARKER_SITE = "arm_profile_so101_menagerie"


def _fmt(values) -> str:
    return " ".join(f"{float(v):.7f}" for v in values)


def _rigid(data, body_id):
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy(), np.asarray(data.xquat[body_id], dtype=np.float64).copy()


def _relative(mujoco, parent, child):
    """Pose of ``child`` expressed in ``parent`` (both world poses (pos, quat))."""
    inv = np.zeros(4); mujoco.mju_negQuat(inv, parent[1])
    pos = np.zeros(3); mujoco.mju_rotVecQuat(pos, child[0] - parent[0], inv)
    quat = np.zeros(4); mujoco.mju_mulQuat(quat, inv, child[1])
    return pos, quat / np.linalg.norm(quat)


def _apply(mujoco, pose, local_pos, local_quat=(1.0, 0.0, 0.0, 0.0)):
    pos = np.zeros(3); mujoco.mju_rotVecQuat(pos, np.asarray(local_pos, dtype=np.float64), pose[1])
    quat = np.zeros(4); mujoco.mju_mulQuat(quat, pose[1], np.asarray(local_quat, dtype=np.float64))
    return pos + pose[0], quat / np.linalg.norm(quat)


def patched_tree() -> ET.ElementTree:
    tree = ET.parse(UPSTREAM)
    root = tree.getroot()
    root.set("model", "so101_bench")
    root.find("compiler").set("meshdir", "../assets")
    base = root.find(".//body[@name='base']")
    base.set("quat", BASE_QUAT)
    ET.SubElement(base, "site", name=MARKER_SITE, group="5", size="0.001", rgba="0 0 0 0")
    for old, new in PAD_RENAMES.items():
        root.find(f".//geom[@name='{old}']").set("name", new)
    gripper = root.find(".//body[@name='gripper']")
    gripper.remove(gripper.find("body[@name='camera_mount']"))
    assets = root.find("asset")
    for mesh in list(assets.findall("mesh")):
        if Path(mesh.get("file")).stem in UNUSED_MESHES:
            assets.remove(mesh)
    # Materials: printed parts -> white (the scene builder makes it near-black PLA), servos -> black.
    printed, servos = [], []
    for material in list(assets.findall("material")):
        (servos if material.get("name").startswith("sts3215") else printed).append(material.get("name"))
        assets.remove(material)
    ET.SubElement(assets, "material", name="white", rgba="1 1 1 1")
    ET.SubElement(assets, "material", name="black", rgba="0.1 0.1 0.1 1")
    # The repo's mount mesh, declared exactly as on the SO-100 (it is in millimetres: scale 0.001).
    so100_mount = ET.parse(SO100 / "so_arm101_v2.xml").getroot().find(".//asset/mesh[@name='so101_camera_wrist_mount']")
    assets.append(ET.fromstring(ET.tostring(so100_mount)))
    for geom in root.iter("geom"):
        name = geom.get("material")
        if name in printed:
            geom.set("material", "white")
        elif name in servos:
            geom.set("material", "black")
    return tree


def _compile(tree: ET.ElementTree, mujoco):
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=VENDOR, delete=False) as handle:
        tree.write(handle, encoding="unicode")
        path = Path(handle.name)
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink()


def build(check_poses: int = 200) -> tuple[ET.ElementTree, dict]:
    import mujoco
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES, MODEL_CONVERSION_OFFSET, MODEL_CONVERSION_SIGN
    tree = patched_tree()
    new = _compile(tree, mujoco)
    old = mujoco.MjModel.from_xml_path(str(SO100 / "scene_v2.xml"))
    dn, do = mujoco.MjData(new), mujoco.MjData(old)
    qn = [int(new.jnt_qposadr[new.joint(n).id]) for n in JOINT_NAMES]
    qo = [int(old.jnt_qposadr[old.joint(n).id]) for n in JOINT_NAMES]
    rng = np.random.default_rng(20261106)
    low = np.array([new.jnt_range[new.joint(n).id][0] for n in JOINT_NAMES])
    high = np.array([new.jnt_range[new.joint(n).id][1] for n in JOINT_NAMES])
    gripper = {}
    for label, body in (("gripper", "gripper"), ("moving_jaw", "moving_jaw_so101_v1")):
        poses = []
        for k in range(check_poses):
            q = np.zeros(6) if k == 0 else rng.uniform(low, high)
            dn.qpos[:] = 0; do.qpos[:] = 0
            dn.qpos[qn] = q
            do.qpos[qo] = MODEL_CONVERSION_SIGN * q + MODEL_CONVERSION_OFFSET
            mujoco.mj_kinematics(new, dn); mujoco.mj_kinematics(old, do)
            poses.append(_relative(mujoco, _rigid(dn, new.body(body).id), _rigid(do, old.body(body).id)))
        pos = np.array([p for p, _ in poses]); quat = np.array([q for _, q in poses])
        angles = 2 * np.degrees(np.arccos(np.clip(np.abs(quat @ quat[0]), 0.0, 1.0)))   # sign-free rotation difference
        gripper[label] = dict(pos=pos[0], quat=quat[0], pos_spread_mm=float(1000 * np.max(np.linalg.norm(pos - pos[0], axis=1))),
                              quat_spread=float(angles.max()))
    # Carry the SO-100's declared site and mount poses across (declared local poses, never compiled mesh poses).
    so100 = ET.parse(SO100 / "so_arm101_v2.xml").getroot()
    t_g = (gripper["gripper"]["pos"], gripper["gripper"]["quat"])
    t_m = (gripper["moving_jaw"]["pos"], gripper["moving_jaw"]["quat"])
    root = tree.getroot()
    g_body = root.find(".//body[@name='gripper']")
    m_body = g_body.find("body[@name='moving_jaw_so101_v1']")
    placed = {}
    for site_name, body_el, transform in (("fixed_jaw_tip", g_body, t_g), ("moving_jaw_tip", m_body, t_m)):
        src = so100.find(f".//site[@name='{site_name}']")
        pos, _ = _apply(mujoco, transform, [float(v) for v in src.get("pos").split()])
        ET.SubElement(body_el, "site", group="3", name=site_name, pos=_fmt(pos), size="0.004",
                      rgba="0 0 1 0.4" if site_name.startswith("fixed") else "0 1 0 0.4")
        placed[site_name] = pos.tolist()
    mount_src = so100.find(".//body[@name='wrist_camera_mount']")
    mount = ET.Element("body", name="wrist_camera_mount", pos=_fmt(t_g[0]), quat=_fmt(t_g[1]))
    for child in mount_src:
        copy = ET.fromstring(ET.tostring(child))
        if copy.tag == "geom":
            # Visual only and massless, as the SO-100's visual class makes it (the SO-101's visual class sets no density).
            copy.set("material", "white")
            copy.set("density", "0")
        mount.append(copy)
    g_body.insert(list(g_body).index(m_body), mount)
    placed["wrist_camera_mount"] = dict(pos=t_g[0].tolist(), quat=t_g[1].tolist())
    record = dict(upstream_sha256=__import__("hashlib").sha256(UPSTREAM.read_bytes()).hexdigest(),
                  gripper_frame_transform=dict(pos=t_g[0].tolist(), quat=t_g[1].tolist(),
                                               spread_over_poses_mm=gripper["gripper"]["pos_spread_mm"],
                                               rotation_spread_deg=gripper["gripper"]["quat_spread"]),
                  moving_jaw_frame_transform=dict(pos=t_m[0].tolist(), quat=t_m[1].tolist(),
                                                  spread_over_poses_mm=gripper["moving_jaw"]["pos_spread_mm"],
                                                  rotation_spread_deg=gripper["moving_jaw"]["quat_spread"]),
                  placed=placed, poses_checked=check_poses)
    ET.indent(tree)
    return tree, record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="regenerate in memory and fail if so101_bench.xml differs")
    args = p.parse_args(argv)
    tree, record = build()
    text = ET.tostring(tree.getroot(), encoding="unicode") + "\n"
    if args.check:
        same = OUTPUT.is_file() and OUTPUT.read_text() == text
        print("so101_bench.xml is current" if same else "so101_bench.xml differs from the generator's output")
        return 0 if same else 1
    OUTPUT.write_text(text)
    (VENDOR / "vendor_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "placed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
