"""Capture wrist-camera frames in which the phone checkerboard is fully detected.

Two modes. ``intrinsics``: the phone is moved by hand in front of the camera
(arm still); frames are kept when the full board is found and the pose differs
from the kept ones. ``extrinsics``: the phone lies flat on the mousepad and the
ARM is moved by hand (torque off); frames are kept when the board is found AND
the arm is still, and the joints are recorded with each frame. Every kept frame
gets a JSON record with the refined corner pixels. Torque is never touched.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/checkerboard_iphone15plus_v3_7x14.json'


def find_board(gray, inner):
    """Full-grid detection; falls back to searching inside the bright phone-screen region, which the
    SB detector handles far better than a mostly dark full frame."""
    import cv2
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    for size in (tuple(inner), tuple(inner[::-1])):
        ok, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
        if ok:
            return corners.reshape(-1, 2)
    white = (gray > 200).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 3000:
        return None
    x, y, w, h = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    m = 40; x0, y0 = max(0, x - m), max(0, y - m); roi = gray[y0:y + h + m, x0:x + w + m]
    for size in (tuple(inner), tuple(inner[::-1])):
        ok, corners = cv2.findChessboardCornersSB(roi, size, flags=flags)
        if ok:
            return corners.reshape(-1, 2) + np.array([x0, y0], dtype=np.float64)
    return None


def main(argv=None) -> int:
    import cv2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['intrinsics', 'extrinsics'], required=True)
    p.add_argument('--board', type=Path, default=DEFAULT_BOARD, help='board JSON with inner_corners and square_mm')
    p.add_argument('--seconds', type=float, default=120); p.add_argument('--max-keep', type=int, default=25)
    p.add_argument('--min-move-px', type=float, default=60, help='intrinsics: board must have moved this much (mean corner shift) since the last kept frame')
    p.add_argument('--still-threshold', type=float, default=0.6); p.add_argument('--distinct-threshold', type=float, default=5.0)
    p.add_argument('--port', default='/dev/ttyACM0'); p.add_argument('--device', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration')
    args = p.parse_args(argv)
    board = json.loads(args.board.read_text()); inner = board['inner_corners']
    out = args.output_dir / args.mode; out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(20):
        cap.read()
    robot = None
    if args.mode == 'extrinsics':
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from so_arm101_v2.contracts.coordinates import JOINT_NAMES
        from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only
        robot = SO101Follower(SO101FollowerConfig(id='None', port=args.port, cameras={}, use_degrees=False)); assert_pinned_calibration(robot); connect_read_only(robot)
    kept = []; last_corners = None; attempts = 0; reasons = {}
    try:
        deadline = time.time() + args.seconds
        while time.time() < deadline and len(kept) < args.max_keep:
            attempts += 1
            before = after = None
            if robot is not None:
                before = np.array([[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(3)])
            for _ in range(3):
                ok, frame = cap.read()
            captured_at = datetime.now(timezone.utc).isoformat()
            if robot is not None:
                after = np.array([[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(3)])
            if not ok or frame is None:
                reasons['camera'] = reasons.get('camera', 0) + 1; continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            corners = find_board(gray, inner)
            if corners is None:
                reasons['no_board'] = reasons.get('no_board', 0) + 1; continue
            if sharp < 40:
                reasons['blurry'] = reasons.get('blurry', 0) + 1; continue
            record = dict(timestamp=captured_at, mode=args.mode, board=board, sharpness=round(sharp, 1), corners_px=corners.round(2).tolist())
            if robot is not None:
                samples = np.vstack([before, after]); spread = float(np.ptp(samples, axis=0).max()); median = np.median(samples, axis=0)
                if spread > args.still_threshold:
                    reasons['arm_moving'] = reasons.get('arm_moving', 0) + 1; continue
                if any(np.max(np.abs(median - k)) < args.distinct_threshold for k in kept):
                    reasons['same_pose'] = reasons.get('same_pose', 0) + 1; continue
                record.update(joint_names=list(JOINT_NAMES), normalized_median=median.tolist(), normalized_spread=np.ptp(samples, axis=0).tolist(), label=f'ext_{len(kept) + 1:02d}')
                kept.append(median)
            else:
                if last_corners is not None and float(np.mean(np.linalg.norm(corners - last_corners, axis=1))) < args.min_move_px:
                    reasons['board_not_moved'] = reasons.get('board_not_moved', 0) + 1; continue
                last_corners = corners; record['label'] = f'int_{len(kept) + 1:02d}'; kept.append(corners)
            stamp = captured_at[:19].replace(':', ''); png = out / f"{stamp}_{record['label']}.png"; cv2.imwrite(str(png), frame)
            record['frame'] = str(png.relative_to(ROOT)); record['frame_sha256'] = hashlib.sha256(png.read_bytes()).hexdigest()
            (out / f"{stamp}_{record['label']}.json").write_text(json.dumps(record, indent=1) + '\n')
            print(f"KEPT {record['label']} sharpness {sharp:.0f} board centre {corners.mean(axis=0).round().astype(int).tolist()}" + (f" joints {np.round(median, 1).tolist()}" if robot is not None else ''), flush=True)
    finally:
        cap.release()
        if robot is not None:
            from so_arm101_v2.contracts.physical_io import disconnect_read_only; disconnect_read_only(robot)
    print(json.dumps(dict(kept=len(kept), attempts=attempts, rejected=reasons)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
