"""Capture one physical wrist-camera frame together with a read-only joint reading.

The pair (frame, joints at that instant) is what a camera-model fit needs; the
arm can be anywhere, posed by hand with torque off. Writes the PNG and an
immutable JSON record under artifacts/.../camera_references/. Never writes a
motor register or changes torque. Camera: MJPEG 1920x1080 (uncompressed 1080p
over USB forwarding produced corrupt frames).
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


def grab_frame(device: int, warmup: int, width: int, height: int):
    import cv2
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height); cap.set(cv2.CAP_PROP_FPS, 30)
        frame = None
        for _ in range(warmup):  # let auto-exposure settle
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('camera read failed')
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f'unexpected frame shape {None if frame is None else frame.shape}')
        props = dict(fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace'),
                     width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=cap.get(cv2.CAP_PROP_FPS))
        return frame, props
    finally:
        cap.release()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True); p.add_argument('--note', required=True)
    p.add_argument('--port', default='/dev/ttyACM0'); p.add_argument('--device', type=int, default=0)
    p.add_argument('--warmup-frames', type=int, default=20); p.add_argument('--samples', type=int, default=10)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_references')
    args = p.parse_args(argv)
    import cv2
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(id='None', port=args.port, cameras={}, use_degrees=False))
    assert_pinned_calibration(robot)
    connect_read_only(robot)
    try:
        before = [[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(args.samples)]
        frame, props = grab_frame(args.device, args.warmup_frames, 1920, 1080)
        captured_at = datetime.now(timezone.utc).isoformat()
        after_norm = [[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(args.samples)]
        raw = robot.bus.sync_read('Present_Position', normalize=False)
        torque = {n: int(robot.bus.read('Torque_Enable', n, normalize=False)) for n in JOINT_NAMES}
    finally:
        disconnect_read_only(robot)
    samples = np.array(before + after_norm); spread = np.ptp(samples, axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = captured_at[:19].replace(':', '')
    png = args.output_dir / f'{stamp}_{args.label}.png'
    if png.exists():
        raise FileExistsError(png)
    cv2.imwrite(str(png), frame)
    record = dict(timestamp=captured_at, label=args.label, note=args.note, frame=str(png.relative_to(ROOT)),
                  frame_sha256=hashlib.sha256(png.read_bytes()).hexdigest(), camera=props, camera_device=f'/dev/video{args.device}',
                  joint_names=list(JOINT_NAMES), normalized_median=np.median(samples, axis=0).tolist(), normalized_spread=spread.tolist(),
                  raw_ticks_after=[int(raw[n]) for n in JOINT_NAMES], torque_enable=torque, moving=bool(np.any(spread > 0.25)))
    digest = content_sha256(record)
    record_path = args.output_dir / f'{stamp}_{args.label}_{digest[:8]}.json'
    write_immutable_json(record_path, record)
    print(json.dumps(dict(record=str(record_path), **{k: record[k] for k in ('frame', 'frame_sha256', 'normalized_median', 'moving', 'camera')}), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
