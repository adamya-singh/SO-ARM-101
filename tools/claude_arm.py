"""Step-at-a-time control of the physical SO-101 for an agent that looks at the camera between moves.

Each call connects, does one thing, and disconnects WITHOUT touching torque, so the arm holds its
last goal between calls. Moves are min-jerk trajectories at 30 Hz, gated on every waypoint:
shoulder floor (-92), the bench's effective joint bounds, a jaw-tip height floor above the table
(computed with the calibrated MuJoCo model), and a stall check that re-seeds the goal to the
present pose (torque stays on) when a joint stops tracking.

Frames: x = right (seen from behind the arm), y = forward from the base, z = up, millimetres;
the "pocket" is the grasp point between the jaws at the fixed pad (the teacher's grasp station).

    claude_arm.py status
    claude_arm.py snap NAME
    claude_arm.py joints --set shoulder_pan=10 --delta elbow_flex=-5
    claude_arm.py goto X Y Z [--yaw DEG] [--pitch DEG] [--grip G]
    claude_arm.py grip G
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts.bench import BenchConfig  # noqa: E402
from so_arm101_v2.contracts.coordinates import JOINT_NAMES  # noqa: E402
from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act  # noqa: E402

SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
OUT = ROOT / "artifacts/claude_arm"
HZ = 30.0
MAX_SPEED = 25.0          # normalized units per second, per joint
MIN_TIP_Z_M = 0.004       # jaw tips stay this far above the table unless --allow-low
LOW_TIP_Z_M = -0.001      # absolute floor even with --allow-low (tips may touch the 1 mm square, not dig)
STALL_UNITS = 12.0        # joint lagging its trajectory by this much for STALL_STEPS steps -> stop
STALL_STEPS = 10


class Kinematics:
    def __init__(self) -> None:
        import mujoco
        self.mj = mujoco
        self.bench = BenchConfig.load(SCENE.parent / "bench_config.json")
        self.jm = self.bench.joint_map_object
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self._joint_qpos = [int(self.model.joint(n).qposadr[0]) for n in JOINT_NAMES]

    def qpos(self, physical) -> np.ndarray:
        return self.jm.act_to_mujoco(physical_normalized_to_act(np.asarray(physical, dtype=np.float32))).astype(np.float64)

    def physical(self, qpos) -> np.ndarray:
        return np.asarray(act_to_physical_normalized(self.jm.mujoco_to_act(np.asarray(qpos, dtype=np.float32))), dtype=np.float64)

    def set(self, qpos) -> None:
        for a, v in zip(self._joint_qpos, qpos):
            self.data.qpos[a] = v
        self.mj.mj_forward(self.model, self.data)

    def frame(self, physical) -> dict:
        from so_arm101_v2.simulation.privileged import PrivilegedStagedController
        self.set(self.qpos(physical))
        pocket, normal, depth = PrivilegedStagedController._pocket_frame(self)
        tips = [self.data.site(n).xpos.copy() for n in ("fixed_jaw_tip", "moving_jaw_tip")]
        cam = self.data.cam("wrist_camera").xpos.copy()
        yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0]))) if abs(normal[2]) < 0.9 else float("nan")
        return dict(pocket_mm=(pocket * 1000).round(1).tolist(), depth_axis=depth.round(3).tolist(),
                    pitch_from_down_deg=round(float(np.degrees(np.arccos(np.clip(-depth[2], -1, 1)))), 1),
                    jaw_yaw_deg=round(yaw, 1), tips_z_mm=[round(float(t[2]) * 1000, 1) for t in tips],
                    lowest_jaw_mm=round(self.lowest_jaw(physical) * 1000, 1),
                    camera_mm=None if cam is None else (cam * 1000).round(1).tolist())

    def lowest_jaw(self, physical) -> float:
        """Lowest z of any jaw/gripper collision geom vertex-ish (box corners, capsule/sphere radius)."""
        self.set(self.qpos(physical))
        low = 1.0
        for g in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[g]).name
            if body not in ("gripper", "moving_jaw_so101_v1"):
                continue
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                continue
            t = self.model.geom_type[g]
            pos = self.data.geom_xpos[g]
            mat = self.data.geom_xmat[g].reshape(3, 3)
            size = self.model.geom_size[g]
            if t == self.mj.mjtGeom.mjGEOM_BOX:
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
                z = (pos + corners @ mat.T)[:, 2].min()
            elif t in (self.mj.mjtGeom.mjGEOM_SPHERE,):
                z = pos[2] - size[0]
            elif t in (self.mj.mjtGeom.mjGEOM_CAPSULE, self.mj.mjtGeom.mjGEOM_CYLINDER):
                axis = mat[:, 2] * size[1]
                z = min(pos[2] + axis[2], pos[2] - axis[2]) - size[0]
            else:
                z = pos[2] - float(self.model.geom_rbound[g])
            low = min(low, float(z))
        return low

    def ik(self, target_m, yaw_deg, pitch_deg, seed_physical, gripper_physical) -> np.ndarray:
        from so_arm101_v2.simulation.privileged import PrivilegedStagedController
        c = PrivilegedStagedController()
        c.approach_pitch_rad = np.deg2rad(pitch_deg)
        c.grasp_yaw_rad = float(np.deg2rad(yaw_deg))
        grip_q = self.qpos(np.r_[seed_physical[:5], gripper_physical])[5]
        seeds = [self.qpos(seed_physical), np.asarray(self.bench.viewing_qpos, dtype=np.float64)]
        # a few extra seeds across the workspace (pan toward the target)
        pan = float(np.arctan2(-target_m[0], target_m[1])) if np.hypot(target_m[0], target_m[1]) > 0.02 else 0.0
        for lift, elbow, wflex in ((-1.0, 1.0, 1.2), (-0.5, 0.5, 1.5), (-1.5, 1.6, 1.0), (-2.2, 2.0, 1.4)):
            seeds.append(np.array([pan, lift, elbow, wflex, self.bench.viewing_qpos[4], 0.0]))
        errors = []
        q = None
        for seed in seeds:
            seed = np.clip(seed, self.bench.mujoco_low, self.bench.mujoco_high)
            try:
                q = c._solve(self, seed.astype(np.float32), np.asarray(target_m, dtype=np.float64), float(grip_q))
                break
            except RuntimeError as exc:
                errors.append(str(exc).split(": ", 1)[-1])
        if q is None:
            raise RuntimeError("IK failed from every seed: " + " | ".join(errors))
        out = self.physical(q)
        out[5] = gripper_physical
        return out


def connect():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only
    robot = SO101Follower(SO101FollowerConfig(id="None", port="/dev/ttyACM0", cameras={}, use_degrees=False))
    connect_read_only(robot)
    assert_pinned_calibration(robot)
    return robot


def disconnect(robot):
    from so_arm101_v2.contracts.physical_io import disconnect_read_only
    disconnect_read_only(robot)


def present(robot) -> np.ndarray:
    p = robot.bus.sync_read("Present_Position")
    return np.array([p[n] for n in JOINT_NAMES], dtype=np.float64)


def write_goal(robot, physical) -> None:
    robot.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(JOINT_NAMES, physical)})


def check_target(kin: Kinematics, physical, allow_low: bool) -> str | None:
    p = np.asarray(physical)
    if p[1] < kin.bench.shoulder_floor:
        return f"shoulder_lift {p[1]:.1f} below floor {kin.bench.shoulder_floor}"
    if np.any(np.abs(p[:5]) > 100) or not 0 <= p[5] <= 100:
        return "outside normalized range"
    try:
        kin.bench.validate_qpos(kin.qpos(p))
    except ValueError as exc:
        return str(exc)
    low = kin.lowest_jaw(p)
    floor = LOW_TIP_Z_M if allow_low else MIN_TIP_Z_M
    if low < floor:
        return f"jaw would reach z={low * 1000:.1f} mm (< {floor * 1000:.1f} mm floor{'' if allow_low else '; pass --allow-low to go to the table'})"
    return None


def move(robot, kin: Kinematics, target, *, allow_low=False, settle_s=1.0, correct=True) -> dict:
    target = np.asarray(target, dtype=np.float64)
    torque = robot.bus.sync_read("Torque_Enable", normalize=False)
    start = present(robot)
    if not all(int(v) == 1 for v in torque.values()):
        write_goal(robot, start)
        robot.bus.enable_torque()
        time.sleep(0.2)
    reason = check_target(kin, target, allow_low)
    if reason:
        return dict(refused=reason)
    span = float(np.max(np.abs(target - start)))
    n = max(2, int(np.ceil(span / MAX_SPEED * HZ)))
    # gate every intermediate waypoint (straight joint-space lines can dip)
    for k in range(1, n + 1, max(1, n // 12)):
        s = k / n
        w = start + (target - start) * (10 * s**3 - 15 * s**4 + 6 * s**5)
        r = check_target(kin, w, allow_low)
        if r:
            return dict(refused=f"waypoint {k}/{n}: {r}")
    lag_count = np.zeros(6, dtype=int)
    t0 = time.monotonic()
    aborted = None
    for k in range(1, n + 1):
        s = k / n
        w = start + (target - start) * (10 * s**3 - 15 * s**4 + 6 * s**5)
        write_goal(robot, w)
        meas = present(robot)
        lag = np.abs(w - meas)
        lag[5] = 0  # gripper may stall on the cube by design
        lag_count = np.where(lag > STALL_UNITS, lag_count + 1, 0)
        if np.any(lag_count >= STALL_STEPS):
            j = int(np.argmax(lag_count))
            write_goal(robot, meas)
            aborted = f"{JOINT_NAMES[j]} stopped tracking (cmd {w[j]:.1f}, at {meas[j]:.1f}); goal re-seeded to present pose"
            break
        delay = t0 + k / HZ - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    if aborted:
        return dict(aborted=aborted, measured=present(robot).round(2).tolist())
    time.sleep(settle_s)
    goal = target.copy()
    if correct:
        # The STS3215s are P-only: lift/elbow/wrist rest a few units short under gravity. Nudge the goal by the
        # residual (capped) twice so the pose matches the plan; never on the gripper.
        for _ in range(2):
            err = target - present(robot)
            fix = np.clip(err, -6, 6)
            fix[[0, 4, 5]] = np.clip(err[[0, 4]], -3, 3).tolist() + [0.0]
            if np.max(np.abs(fix)) < 0.6:
                break
            goal = goal + fix
            if check_target(kin, goal, allow_low):
                goal = goal - fix
                break
            write_goal(robot, goal)
            time.sleep(0.6)
    meas = present(robot)
    load = robot.bus.sync_read("Present_Load", normalize=False)
    return dict(target=target.round(2).tolist(), goal=goal.round(2).tolist(), measured=meas.round(2).tolist(),
                error=(target - meas).round(2).tolist(), load=load)


def snap(name: str) -> str:
    import cv2
    from so_arm101_v2.physical.camera import FrameGrabber
    OUT.mkdir(parents=True, exist_ok=True)
    with FrameGrabber(0) as g:
        time.sleep(0.4)
        _, _, f = g.latest()
        f = f.copy()
    full = OUT / f"{name}_full.jpg"
    small = OUT / f"{name}.jpg"
    cv2.imwrite(str(full), f)
    s = cv2.resize(f, (960, 540))
    for x in range(0, 960, 120):
        cv2.line(s, (x, 0), (x, 539), (0, 255, 255), 1)
        cv2.putText(s, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, 540, 90):
        cv2.line(s, (0, y), (959, y), (0, 255, 255), 1)
        cv2.putText(s, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(str(small), s)
    return str(small)


def pixel_to_table(kin: "Kinematics", physical, u_full: float, v_full: float, z_m: float) -> list:
    """Back-project a full-resolution pixel through the calibrated lens and the FK wrist camera onto z = z_m."""
    import cv2
    lens = kin.bench.lens
    K = np.array([[lens["fx"], 0, lens["cx"]], [0, lens["fy"], lens["cy"]], [0, 0, 1.0]])
    n = cv2.undistortPoints(np.array([[[u_full, v_full]]], dtype=np.float64), K, np.asarray(lens["dist"], dtype=np.float64))[0, 0]
    kin.set(kin.qpos(physical))
    cam = kin.data.cam("wrist_camera")
    R = np.asarray(cam.xmat).reshape(3, 3)
    ray = R @ np.array([n[0], -n[1], -1.0])
    origin = np.asarray(cam.xpos)
    t = (z_m - origin[2]) / ray[2]
    return (origin + t * ray).round(4).tolist()


def find_cube(path_full: str):
    """Darkest compact blob lying on the bright towel: returns its full-res centroid and bbox."""
    import cv2
    img = cv2.imread(path_full)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    towel = (gray > 150).astype(np.uint8)
    towel = cv2.morphologyEx(towel, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))
    filled = np.zeros_like(towel)
    contours, _ = cv2.findContours(towel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    dark = ((gray < 90) & (filled > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, labels, stats, cents = cv2.connectedComponentsWithStats(dark)
    if count < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = stats[i]
    return dict(centroid=cents[i].round(1).tolist(), bbox=[int(x), int(y), int(w), int(h)], area=int(a))


def log(record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(t=time.time(), **record)) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("snap"); s.add_argument("name")
    j = sub.add_parser("joints")
    j.add_argument("--set", nargs="*", default=[]); j.add_argument("--delta", nargs="*", default=[])
    g = sub.add_parser("goto")
    g.add_argument("x", type=float); g.add_argument("y", type=float); g.add_argument("z", type=float)
    g.add_argument("--yaw", type=float, default=0.0); g.add_argument("--pitch", type=float, default=5.0)
    g.add_argument("--grip", type=float, default=None)
    gr = sub.add_parser("grip"); gr.add_argument("value", type=float)
    sub.add_parser("fk")
    lo = sub.add_parser("locate"); lo.add_argument("name"); lo.add_argument("--u", type=float); lo.add_argument("--v", type=float)
    lo.add_argument("--z", type=float, default=21.0)
    for q in (j, g, gr):
        q.add_argument("--allow-low", action="store_true")
        q.add_argument("--no-correct", action="store_true")
        q.add_argument("--snap", default=None)
        q.add_argument("--dry", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "snap":
        print(snap(args.name))
        return 0
    kin = Kinematics()
    robot = connect()
    try:
        cur = present(robot)
        if args.cmd == "locate":
            if args.u is None:
                found = find_cube(str(OUT / f"{args.name}_full.jpg"))
                if found is None:
                    print(json.dumps(dict(found=None))); return 1
                u, v = found["centroid"]
            else:
                found, u, v = None, args.u, args.v
            xyz = pixel_to_table(kin, cur, u, v, args.z / 1000)
            print(json.dumps(dict(detection=found, pixel=[u, v], table_mm=[round(c * 1000, 1) for c in xyz])))
            return 0
        if args.cmd in ("status", "fk"):
            out = dict(physical=dict(zip(JOINT_NAMES, cur.round(2).tolist())), **kin.frame(cur))
            if args.cmd == "status":
                out["torque"] = robot.bus.sync_read("Torque_Enable", normalize=False)
                out["load"] = robot.bus.sync_read("Present_Load", normalize=False)
                out["volts"] = robot.bus.sync_read("Present_Voltage", normalize=False)["gripper"] / 10
            print(json.dumps(out))
            return 0
        if args.cmd == "joints":
            target = cur.copy()
            for item in args.set:
                k, v = item.split("="); target[JOINT_NAMES.index(k)] = float(v)
            for item in args.delta:
                k, v = item.split("="); target[JOINT_NAMES.index(k)] += float(v)
        elif args.cmd == "grip":
            target = cur.copy(); target[5] = args.value
        else:
            grip = cur[5] if args.grip is None else args.grip
            target = kin.ik(np.array([args.x, args.y, args.z]) / 1000, args.yaw, args.pitch, cur, grip)
        plan = dict(cmd=args.cmd, argv=sys.argv[1:], start=cur.round(2).tolist(), target=np.round(target, 2).tolist(),
                    target_frame=kin.frame(target))
        if args.dry:
            plan["check"] = check_target(kin, target, args.allow_low) or "ok"
            print(json.dumps(plan)); return 0
        result = move(robot, kin, target, allow_low=args.allow_low, correct=not args.no_correct and args.cmd != "grip")
        result["frame"] = kin.frame(present(robot))
        log(dict(plan=plan, result=result))
        print(json.dumps(dict(plan_target_frame=plan["target_frame"], **result)))
    finally:
        disconnect(robot)
    if getattr(args, "snap", None):
        print(snap(args.snap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
