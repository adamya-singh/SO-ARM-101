"""Capture wrist-camera frames in which the phone checkerboard is fully detected.

Two modes. ``intrinsics``: the phone is moved by hand in front of the camera
(arm still); frames are kept when the full board is found and the pose differs
from the kept ones. ``extrinsics``: the phone lies flat on the mousepad and the
ARM is moved by hand (torque off); frames are kept when the board is found AND
the arm is still, and the joints are recorded with each frame. Every kept frame
gets a JSON record with the refined corner pixels. Torque is never touched.

Intrinsics mode also tracks *coverage*: the frame is divided into a 4x3 grid and
a view counts for every cell that contains board corners. Existing records in the
output directory seed the grid, so a second session can target the cells the
first one missed (the lens model is only trustworthy where corners were seen).
With ``--assist`` a hint line is printed whenever the situation changes: which
cells still need the board, where the board currently is, and whether it is cut
off at a frame edge.

A live preview window (tools/camera_preview.py) shows the stream while the tool
runs; a red dot flashes in its top-right corner every time a frame is saved.
``--no-preview`` disables it; without a display it degrades to headless.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
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


GRID_COLS, GRID_ROWS = 4, 3
CELL_NAMES = [[f"{r}{c}" for c in 'LlrR'] for r in 'TMB']  # T/M/B rows x L,l,r,R columns (outer-left, inner-left, inner-right, outer-right)


def coverage_cells(corners, width=1920, height=1080):
    """Set of (row, col) grid cells that contain at least one detected corner."""
    c = np.asarray(corners, dtype=np.float64)
    cols = np.clip((c[:, 0] * GRID_COLS / width).astype(int), 0, GRID_COLS - 1)
    rows = np.clip((c[:, 1] * GRID_ROWS / height).astype(int), 0, GRID_ROWS - 1)
    return set(zip(rows.tolist(), cols.tolist()))


def coverage_grid(records):
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for rec in records:
        for r, c in coverage_cells(rec['corners_px']):
            grid[r, c] += 1
    return grid


def coverage_hint(grid, target):
    missing = [CELL_NAMES[r][c] for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r, c] < target]
    return missing


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
    p.add_argument('--assist', action='store_true', help='intrinsics: print live hints (coverage cells still needed, board position, cut-off edges)')
    p.add_argument('--coverage-target', type=int, default=4, help='intrinsics: views wanted per grid cell before a cell counts as covered')
    p.add_argument('--label-prefix', default=None, help='label prefix for kept records (default int_/ext_); numbering continues after existing records')
    p.add_argument('--no-preview', action='store_true', help='do not open the live preview window')
    args = p.parse_args(argv)
    from camera_preview import FrameGrabber, PreviewWindow, _HeadlessPreview, run_with_preview
    board = json.loads(args.board.read_text()); inner = board['inner_corners']
    out = args.output_dir / args.mode; out.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(f.read_text()) for f in sorted(out.glob('*.json'))]
    prefix = args.label_prefix or ('int' if args.mode == 'intrinsics' else 'ext')
    grid = coverage_grid(existing) if args.mode == 'intrinsics' else None
    if grid is not None:
        print(f"coverage from {len(existing)} existing views (rows top->bottom, cols left->right):\n{grid}\nstill needed (<{args.coverage_target} views): {coverage_hint(grid, args.coverage_target)}", flush=True)
    grabber = FrameGrabber(args.device)
    preview = _HeadlessPreview() if args.no_preview else PreviewWindow(f'wrist camera: checkerboard {args.mode}', grabber)
    robot = None
    if args.mode == 'extrinsics':
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from so_arm101_v2.contracts.coordinates import JOINT_NAMES
        from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only
        robot = SO101Follower(SO101FollowerConfig(id='None', port=args.port, cameras={}, use_degrees=False)); assert_pinned_calibration(robot); connect_read_only(robot)
    kept = []; last_corners = None; attempts = 0; reasons = {}; last_hint = ''; last_seq = -1
    def status(text):
        preview.set_status(f"kept {len(existing)} ({len(kept)} this run) | {text}")
    def hint_line(gray, corners):
        white = (gray > 200).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(white)
        blobs = sorted([(int(st[i, 4]), int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])) for i in range(1, n) if st[i, 4] > 3000], reverse=True)
        parts = []
        if blobs:
            a, x, y, w, h = blobs[0]
            edges = [name for cond, name in ((y <= 2, 'TOP'), (y + h >= 1078, 'BOTTOM'), (x <= 2, 'LEFT'), (x + w >= 1918, 'RIGHT')) if cond]
            where = f"screen at ({x + w // 2},{y + h // 2}) {w}x{h}px"
            if edges:
                parts.append(f"{where}, cut off at {'/'.join(edges)} -> move the phone back in so the whole pattern is visible")
            elif corners is None:
                parts.append(f"{where}, pattern NOT detected -> finger/glare over the squares, too oblique, or too far (squares too small)")
            else:
                cells = sorted(CELL_NAMES[r][c] for r, c in coverage_cells(corners))
                parts.append(f"{where}, detected in cells {cells}")
        else:
            parts.append('no bright screen in view -> bring the phone in front of the camera')
        parts.append(f"needed: {coverage_hint(grid, args.coverage_target)}")
        return ' | '.join(parts)
    def loop():
        nonlocal attempts, last_corners, last_hint, last_seq
        deadline = time.time() + args.seconds
        status('starting')
        while time.time() < deadline and len(kept) < args.max_keep and not preview.closed:
            attempts += 1
            before = after = None
            if robot is not None:
                before = np.array([[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(3)])
            seq, _stamp, frame = grabber.wait_for_new(last_seq)
            captured_at = datetime.now(timezone.utc).isoformat()
            if robot is not None:
                after = np.array([[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(3)])
            if seq == last_seq or frame is None:
                reasons['camera'] = reasons.get('camera', 0) + 1; continue
            last_seq = seq
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            corners = find_board(gray, inner)
            if args.assist and grid is not None:
                line = hint_line(gray, corners)
                if line != last_hint:
                    print(f"{time.time() - (deadline - args.seconds):5.1f}s {line}", flush=True); last_hint = line
                status(line)
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
                record.update(joint_names=list(JOINT_NAMES), normalized_median=median.tolist(), normalized_spread=np.ptp(samples, axis=0).tolist(), label=f'{prefix}_{len(existing) + len(kept) + 1:02d}')
                kept.append(median)
            else:
                if last_corners is not None and float(np.mean(np.linalg.norm(corners - last_corners, axis=1))) < args.min_move_px:
                    reasons['board_not_moved'] = reasons.get('board_not_moved', 0) + 1; continue
                last_corners = corners; record['label'] = f'{prefix}_{len(existing) + len(kept) + 1:02d}'; kept.append(corners)
                if grid is not None:
                    for r, c in coverage_cells(corners):
                        grid[r, c] += 1
            stamp = captured_at[:19].replace(':', ''); png = out / f"{stamp}_{record['label']}.png"; cv2.imwrite(str(png), frame)
            record['frame'] = str(png.relative_to(ROOT)); record['frame_sha256'] = hashlib.sha256(png.read_bytes()).hexdigest()
            (out / f"{stamp}_{record['label']}.json").write_text(json.dumps(record, indent=1) + '\n')
            print(f"KEPT {record['label']} sharpness {sharp:.0f} board centre {corners.mean(axis=0).round().astype(int).tolist()}" + (f" joints {np.round(median, 1).tolist()}" if robot is not None else '')
                  + (f" | still needed: {coverage_hint(grid, args.coverage_target)}" if grid is not None else ''), flush=True)
            existing.append(record); preview.flash()
            status(f"saved {record['label']}" + (f" | needed: {coverage_hint(grid, args.coverage_target)}" if grid is not None else ''))
    try:
        run_with_preview(loop, preview)
    finally:
        grabber.close()
        if robot is not None:
            from so_arm101_v2.contracts.physical_io import disconnect_read_only; disconnect_read_only(robot)
    summary = dict(kept=len(kept), attempts=attempts, rejected=reasons)
    if grid is not None:
        summary.update(coverage_grid=grid.tolist(), still_needed=coverage_hint(grid, args.coverage_target))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
