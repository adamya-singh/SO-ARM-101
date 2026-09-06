"""Capture physical reset evidence without writing motor registers."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from so_arm101_v2.contracts.bench import BenchConfig
from so_arm101_v2.contracts.physical_io import connect_read_only, disconnect_read_only
from so_arm101_v2.contracts.physical import physical_normalized_to_act
from so_arm101_v2.contracts.coordinates import act_to_mujoco_qpos, JOINT_NAMES
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default='/dev/ttyACM0')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(id='None', port=args.port, cameras={}, use_degrees=False))
    pinned = Path(__file__).resolve().parents[1] / 'src/so_arm101_v2/data/resources/physical_inference_calibration_20260620.json'
    if json.loads(robot.calibration_fpath.read_text()) != json.loads(pinned.read_text()):
        raise RuntimeError('live calibration differs from pinned calibration')
    rows, raw_rows = [], []
    try:
        connect_read_only(robot)
        for _ in range(20):
            norm = robot.bus.sync_read('Present_Position')
            raw = robot.bus.sync_read('Present_Position', normalize=False)
            rows.append([norm[n] for n in JOINT_NAMES])
            raw_rows.append([raw[n] for n in JOINT_NAMES])
            time.sleep(0.05)
    finally:
        disconnect_read_only(robot)
    positions = np.median(rows, axis=0)
    spread = np.ptp(rows, axis=0)
    qpos = act_to_mujoco_qpos(physical_normalized_to_act(positions))
    failure = None
    try:
        if np.any(spread > 0.25):
            raise ValueError('arm moved during capture')
        BenchConfig(reset_physical=tuple(positions))
    except ValueError as exc:
        failure = str(exc)
    device_links = [str(v) for v in Path('/dev/serial/by-id').glob('*') if v.resolve() == Path(args.port).resolve()]
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), port=args.port,
                  device_links=device_links, joint_names=list(JOINT_NAMES),
                  calibration_sha256=hashlib.sha256(pinned.read_bytes()).hexdigest(),
                  normalized_samples=rows, raw_tick_samples=raw_rows,
                  normalized_median=positions.tolist(), spread=spread.tolist(),
                  mujoco_qpos=qpos.tolist(), valid=failure is None, failure=failure)
    digest = content_sha256(report)
    dest = args.output_dir / digest[:16]
    write_immutable_json(dest / 'reset_evidence.json', report)
    if failure is None:
        config = BenchConfig(reset_physical=tuple(positions), reset_evidence_sha256=digest)
        write_immutable_json(dest / 'bench_config.json', asdict(config))
    print(json.dumps(dict(directory=str(dest), **report), indent=2))
    return 0 if failure is None else 2

if __name__ == '__main__':
    raise SystemExit(main())
