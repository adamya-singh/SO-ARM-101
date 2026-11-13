"""Write the position-loop P gain of the arm servos (on the user's explicit request), and nothing else.

LeRobot's ``configure()`` left every servo at P 16 (factory 32); ``tools/read_servo_registers.py`` reads that. This
tool sets ``P_Coefficient`` to 16 or 32 on the chosen joints. The value is stored in the servo's EPROM, so it
persists across power cycles; run it again with ``--p 16`` to go back.

It refuses unless torque is off on every servo (a gain step under load makes the arm jump), opens the bus the way
the read-only tools do (no connect()/configure()), unlocks the EPROM of one servo at a time, writes, locks it again,
reads every servo back, and records before/after under physical/. The gripper is left alone unless named: its
squeeze was tuned at P 16.

  PYTHONNOUSERSITE=1 PYTHONPATH=src:/home/win10ubuntu/projects/lerobot-fork/src \\
      ~/miniforge3/envs/lerobot/bin/python tools/write_servo_gain.py --p 32
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from so_arm101_v2.contracts.coordinates import JOINT_NAMES
from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only, disconnect_read_only
from so_arm101_v2.data._serialization import write_immutable_json

ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = tuple(name for name in JOINT_NAMES if name != "gripper")


def read(robot, register: str) -> dict[str, int]:
    values = robot.bus.sync_read(register, normalize=False)
    return {motor: int(values[motor]) for motor in JOINT_NAMES}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p", type=int, required=True, choices=(16, 32), help="P_Coefficient to write (16 = LeRobot's, 32 = factory)")
    p.add_argument("--joints", nargs="+", default=list(ARM_JOINTS), choices=list(JOINT_NAMES))
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/physical")
    args = p.parse_args(argv)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(id="None", port=args.port, cameras={}, use_degrees=False))
    assert_pinned_calibration(robot)
    connect_read_only(robot)
    try:
        torque = read(robot, "Torque_Enable")
        if any(torque.values()):
            raise SystemExit(f"torque is on ({torque}): park the arm and run tools/torque_off.py first; nothing written")
        before = read(robot, "P_Coefficient")
        for motor in args.joints:
            robot.bus.write("Lock", motor, 0)
            try:
                robot.bus.write("P_Coefficient", motor, args.p)
            finally:
                robot.bus.write("Lock", motor, 1)
        after, lock = read(robot, "P_Coefficient"), read(robot, "Lock")
    finally:
        disconnect_read_only(robot)
    ok = all(after[m] == args.p for m in args.joints) and all(after[m] == before[m] for m in JOINT_NAMES if m not in args.joints)
    stamp = datetime.now(timezone.utc)
    record = dict(tool="write_servo_gain", timestamp=stamp.isoformat(), port=args.port, requested_p=args.p, joints=list(args.joints),
                  before=before, after=after, lock_after=lock, torque_enable=torque, verified=ok)
    path = args.output_dir / f"servo_gain_write_{stamp.strftime('%Y%m%dT%H%M%S')}.json"
    write_immutable_json(path, record)
    print(f"P_Coefficient before {before}\nP_Coefficient after  {after}\nlock {lock}\nverified: {ok}\nrecord: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
