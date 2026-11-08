"""Read the servos' position-loop gains and related registers, strictly read-only (2026-11-08).

LeRobot's ``SOFollower.connect()`` -> ``configure()`` writes P_Coefficient 16 (factory 32), I 0, D 32, acceleration 254
and gripper torque/current limits, with the EPROM unlocked, so the values persist. Every ``robot.connect()`` since
January did that (the June ACT script, the teleop scripts); this repo's runner never does. This tool reads what is in
the servos now and writes nothing: it opens the bus with ``connect_read_only`` (handshake reads only), reads each
register with ``sync_read(normalize=False)``, and closes with ``disconnect_read_only`` (port closed, torque untouched).
It never calls connect(), configure(), disconnect(), enable_torque(), disable_torque() or torque_disabled().

  PYTHONNOUSERSITE=1 PYTHONPATH=src:/home/win10ubuntu/projects/lerobot-fork/src \\
      ~/miniforge3/envs/lerobot/bin/python tools/read_servo_registers.py [--port /dev/ttyACM0]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from so_arm101_v2.contracts.coordinates import JOINT_NAMES
from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only, disconnect_read_only
from so_arm101_v2.data._serialization import write_immutable_json

ROOT = Path(__file__).resolve().parents[1]
# Feetech STS3215 control table names (lerobot-fork src/lerobot/motors/feetech/tables.py).
REGISTERS = ("P_Coefficient", "D_Coefficient", "I_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone", "Minimum_Startup_Force",
             "Max_Torque_Limit", "Torque_Limit", "Protection_Current", "Overload_Torque", "Acceleration", "Maximum_Acceleration",
             "Return_Delay_Time", "Operating_Mode", "Lock", "Torque_Enable", "Present_Voltage")
FACTORY = {"P_Coefficient": 32, "D_Coefficient": 32, "I_Coefficient": 0}
LEROBOT_CONFIGURE = {"P_Coefficient": 16, "D_Coefficient": 32, "I_Coefficient": 0}


def read_registers(robot) -> dict[str, dict[str, int]]:
    """Every register in REGISTERS for every motor, raw. The caller opens and closes the bus read-only."""
    values: dict[str, dict[str, int]] = {}
    for name in REGISTERS:
        read = robot.bus.sync_read(name, normalize=False)
        values[name] = {motor: int(read[motor]) for motor in JOINT_NAMES}
    return values


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/physical")
    args = p.parse_args(argv)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(id="None", port=args.port, cameras={}, use_degrees=False))
    assert_pinned_calibration(robot)
    connect_read_only(robot)
    try:
        values = read_registers(robot)
    finally:
        disconnect_read_only(robot)
    stamp = datetime.now(timezone.utc)
    p_values = set(values["P_Coefficient"].values())
    verdict = ("lerobot_configure (P 16, written by robot.connect())" if p_values == {16}
               else "factory (P 32)" if p_values == {32} else f"mixed P values {sorted(p_values)}")
    record = dict(tool="read_servo_registers", read_only=True, timestamp=stamp.isoformat(), port=args.port, registers=values,
                  reference=dict(factory=FACTORY, lerobot_configure=LEROBOT_CONFIGURE), verdict=verdict)
    path = args.output_dir / f"servo_registers_{stamp.strftime('%Y%m%dT%H%M%S')}.json"
    write_immutable_json(path, record)
    width = max(len(n) for n in REGISTERS)
    print("register".ljust(width), *[m[:11].rjust(12) for m in JOINT_NAMES])
    for name in REGISTERS:
        print(name.ljust(width), *[str(values[name][m]).rjust(12) for m in JOINT_NAMES])
    print(f"verdict: {verdict}\nrecord: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
