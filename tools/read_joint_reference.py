"""Record a read-only joint reference reading for the coordinate-contract fix.

The arm is posed BY HAND (torque off) at a configuration the user describes;
this tool reads Present_Position for all six joints (normalized and raw ticks,
20 samples, median) and writes an immutable JSON under
artifacts/so_arm101_v2/bench_pick_replace_v1/joint_references/. It never
writes a motor register or changes torque. Encoder: 4096 ticks per turn.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from so_arm101_v2.contracts.coordinates import JOINT_NAMES
from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only, disconnect_read_only
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True, help='short slug for the posed configuration, e.g. wrist_roll_jaws_horizontal')
    p.add_argument('--note', required=True, help='what the user physically did / what the pose is meant to be')
    p.add_argument('--port', default='/dev/ttyACM0')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/joint_references')
    args = p.parse_args(argv)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(id='None', port=args.port, cameras={}, use_degrees=False))
    assert_pinned_calibration(robot)
    norm_rows, raw_rows = [], []
    connect_read_only(robot)
    try:
        torque = {n: int(robot.bus.read('Torque_Enable', n, normalize=False)) for n in JOINT_NAMES}
        for _ in range(args.samples):
            norm = robot.bus.sync_read('Present_Position'); raw = robot.bus.sync_read('Present_Position', normalize=False)
            norm_rows.append([float(norm[n]) for n in JOINT_NAMES]); raw_rows.append([int(raw[n]) for n in JOINT_NAMES])
            time.sleep(0.05)
    finally:
        disconnect_read_only(robot)
    raw_med = np.median(raw_rows, axis=0); spread = np.ptp(raw_rows, axis=0)
    cal = robot.calibration
    record = dict(
        timestamp=datetime.now(timezone.utc).isoformat(), label=args.label, note=args.note, port=args.port,
        joint_names=list(JOINT_NAMES), torque_enable=torque, samples=args.samples,
        normalized_median=np.median(norm_rows, axis=0).tolist(), raw_tick_median=raw_med.tolist(), raw_tick_spread=spread.tolist(),
        degrees_from_calibration_mid=[float((raw_med[i] - (cal[n].range_min + cal[n].range_max) / 2) * 360 / 4096) for i, n in enumerate(JOINT_NAMES)],
        calibration={n: dict(range_min=cal[n].range_min, range_max=cal[n].range_max, homing_offset=cal[n].homing_offset, drive_mode=cal[n].drive_mode) for n in JOINT_NAMES},
        ticks_per_turn=4096, moving=bool(np.any(spread > 8)),
    )
    digest = content_sha256(record)
    dest = args.output_dir / f"{record['timestamp'][:19].replace(':', '')}_{args.label}_{digest[:8]}.json"
    write_immutable_json(dest, record)
    print(json.dumps(dict(file=str(dest), **{k: record[k] for k in ('label', 'normalized_median', 'raw_tick_median', 'degrees_from_calibration_mid', 'moving')}), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
