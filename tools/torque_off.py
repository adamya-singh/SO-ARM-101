"""Disable torque on all six servos (on the user's explicit request), recording the pose before and after.

The physical runner never disables torque; this is the one tool that does,
for parking the arm at the end of a session. The arm sags under gravity
toward its rest pose once torque is off, so run it only with the arm near
the reset/rest pose or supported by hand. Nothing else is written to the bus.
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

from so_arm101_v2.contracts.physical import act_to_physical_normalized  # noqa: E402
from so_arm101_v2.contracts.physical_io import connect_read_only, disconnect_read_only, read_measured_act  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-port", default="/dev/ttyACM0")
    p.add_argument("--settle-seconds", type=float, default=1.5)
    args = p.parse_args(argv)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(id="None", port=args.robot_port, cameras={}, max_relative_target=20.0, use_degrees=False))
    connect_read_only(robot)
    try:
        before = act_to_physical_normalized(read_measured_act(robot))
        robot.bus.disable_torque()
        time.sleep(args.settle_seconds)
        after = act_to_physical_normalized(read_measured_act(robot))
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
    finally:
        disconnect_read_only(robot)
    record = dict(before_physical=[round(float(v), 2) for v in before], after_physical=[round(float(v), 2) for v in after],
                  sag_units=[round(float(v), 2) for v in (after - before)], torque_enable=torque, at=time.time())
    print(json.dumps(record))
    return 0 if all(int(v) == 0 for v in torque.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
