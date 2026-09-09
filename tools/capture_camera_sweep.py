"""Continuously capture wrist frames + read-only joints while the arm is moved by hand; keep the good ones.

A frame is kept when the arm is still (joint spread below a threshold across the
capture), the white square is fully inside the view (four convex-hull corners
off the image border), and the pose differs from every kept pose. Records use
the same schema as tools/capture_camera_reference.py. Torque is never touched.
A live preview window shows the stream; its top-right dot flashes on every kept
frame (``--no-preview`` to disable; headless shells degrade automatically).
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def main(argv=None) -> int:
    from fit_bench_camera import detect_square_corners
    import cv2
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only, disconnect_read_only
    from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
    from camera_preview import FrameGrabber, PreviewWindow, _HeadlessPreview, run_with_preview
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=90); p.add_argument('--max-keep', type=int, default=8)
    p.add_argument('--still-threshold', type=float, default=0.6, help='max normalized-unit spread across a capture')
    p.add_argument('--distinct-threshold', type=float, default=6.0, help='min max-joint difference (normalized units) from every kept pose')
    p.add_argument('--port', default='/dev/ttyACM0'); p.add_argument('--device', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_references')
    p.add_argument('--no-preview', action='store_true', help='do not open the live preview window')
    args = p.parse_args(argv)
    robot = SO101Follower(SO101FollowerConfig(id='None', port=args.port, cameras={}, use_degrees=False))
    assert_pinned_calibration(robot)
    grabber = FrameGrabber(args.device)
    preview = _HeadlessPreview() if args.no_preview else PreviewWindow('wrist camera: square sweep', grabber)
    kept = []; attempts = 0; reasons = {}; last_seq = -1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connect_read_only(robot)
    def loop():
        nonlocal attempts, last_seq
        deadline = time.time() + args.seconds
        while time.time() < deadline and len(kept) < args.max_keep and not preview.closed:
            attempts += 1
            before = [[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(4)]
            seq, _stamp, frame = grabber.wait_for_new(last_seq)
            ok = seq != last_seq; last_seq = seq
            captured_at = datetime.now(timezone.utc).isoformat()
            after = [[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(4)]
            samples = np.array(before + after); spread = float(np.ptp(samples, axis=0).max()); median = np.median(samples, axis=0)
            if not ok or frame is None:
                reasons['camera'] = reasons.get('camera', 0) + 1; continue
            if spread > args.still_threshold:
                reasons['moving'] = reasons.get('moving', 0) + 1; preview.set_status(f'kept {len(kept)} | arm moving -> hold still'); continue
            tmp = args.output_dir / '_sweep_candidate.png'; cv2.imwrite(str(tmp), frame)
            corners, status = detect_square_corners(tmp)
            if status != 'full' or len(corners) != 4:
                reasons[f'square_{status}_{len(corners)}'] = reasons.get(f'square_{status}_{len(corners)}', 0) + 1; preview.set_status(f'kept {len(kept)} | square {status} ({len(corners)} corners) -> bring the whole square into view'); continue
            if any(np.max(np.abs(median - k)) < args.distinct_threshold for k in kept):
                reasons['same_pose'] = reasons.get('same_pose', 0) + 1; continue
            stamp = captured_at[:19].replace(':', ''); label = f'sweep_{len(kept) + 1:02d}'
            png = args.output_dir / f'{stamp}_{label}.png'; tmp.rename(png)
            raw = robot.bus.sync_read('Present_Position', normalize=False)
            record = dict(timestamp=captured_at, label=label, note='Hand-posed sweep frame (torque off); kept because the arm was still and the whole square was in view.',
                          frame=str(png.relative_to(ROOT)), frame_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                          camera=dict(fourcc='MJPG', width=1920, height=1080), camera_device=f'/dev/video{args.device}', joint_names=list(JOINT_NAMES),
                          normalized_median=median.tolist(), normalized_spread=np.ptp(samples, axis=0).tolist(), raw_ticks_after=[int(raw[n]) for n in JOINT_NAMES],
                          square_corners_px=[list(c) for c in corners], moving=False)
            digest = content_sha256(record)
            write_immutable_json(args.output_dir / f'{stamp}_{label}_{digest[:8]}.json', record)
            kept.append(median); preview.flash(); preview.set_status(f'kept {len(kept)} | saved {label} -> move to a different pose')
            print(f"KEPT {label}: joints {np.round(median, 1).tolist()} corners {[tuple(round(v) for v in c) for c in corners]}", flush=True)
    try:
        run_with_preview(loop, preview)
    finally:
        disconnect_read_only(robot); grabber.close()
        if (args.output_dir / '_sweep_candidate.png').exists():
            (args.output_dir / '_sweep_candidate.png').unlink()
    print(json.dumps(dict(kept=len(kept), attempts=attempts, rejected=reasons)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
