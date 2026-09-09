"""Live assist for the flat-board extrinsics capture: samples the camera every ~1.5 s, prints what to change
(board cut off on which side, not found, arm moving), and KEEPS a frame with joints whenever the whole board is
detected and the arm is still. Same record format as tools/capture_checkerboard.py --mode extrinsics.
Opens the live preview window (tools/camera_preview.py); its top-right dot flashes on every kept frame."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def main(argv=None) -> int:
    import cv2
    from capture_checkerboard import find_board, DEFAULT_BOARD
    from camera_preview import FrameGrabber, PreviewWindow, _HeadlessPreview, run_with_preview
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    from so_arm101_v2.contracts.physical_io import assert_pinned_calibration, connect_read_only, disconnect_read_only
    p = argparse.ArgumentParser(); p.add_argument('--seconds', type=float, default=45); p.add_argument('--max-keep', type=int, default=12)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/extrinsics')
    p.add_argument('--no-preview', action='store_true'); p.add_argument('--device', type=int, default=0)
    args = p.parse_args(argv)
    board = json.loads(DEFAULT_BOARD.read_text()); inner = board['inner_corners']; args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(args.output_dir.glob('*.json')))
    grabber = FrameGrabber(args.device)
    preview = _HeadlessPreview() if args.no_preview else PreviewWindow('wrist camera: flat-board extrinsics', grabber)
    robot = SO101Follower(SO101FollowerConfig(id='None', port='/dev/ttyACM0', cameras={}, use_degrees=False)); assert_pinned_calibration(robot); connect_read_only(robot)
    kept = []; t0 = time.time(); last = ''; last_seq = -1
    def loop():
        nonlocal last, last_seq
        while time.time() - t0 < args.seconds and len(kept) < args.max_keep and not preview.closed:
            before = np.array([[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(2)])
            seq, _st, f = grabber.wait_for_new(last_seq); last_seq = seq
            stamp = datetime.now(timezone.utc).isoformat()
            after = np.array([[float(robot.bus.sync_read('Present_Position')[n]) for n in JOINT_NAMES] for _ in range(2)])
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); corners = find_board(g, inner)
            white = (g > 200).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(white)
            blobs = sorted([(int(st[i, 4]), int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])) for i in range(1, n) if st[i, 4] > 3000], reverse=True)
            hint = []
            if blobs:
                a, x, y, w, h = blobs[0]
                if y <= 2: hint.append('board cut off at TOP -> tilt camera DOWN a little')
                if y + h >= 1078: hint.append('board cut off at BOTTOM -> tilt camera UP / move arm back')
                if x <= 2: hint.append('cut off at LEFT -> pan RIGHT')
                if x + w >= 1918: hint.append('cut off at RIGHT -> pan LEFT')
                if not hint: hint.append(f'screen fully inside view at ({x+w//2},{y+h//2}), {w}x{h}px' + ('' if corners is not None else ' -> something covers part of the pattern (finger/hand/glare?) or too oblique'))
            else:
                hint.append('no bright screen in view -> point the camera at the phone')
            samples = np.vstack([before, after]); spread = float(np.ptp(samples, axis=0).max()); median = np.median(samples, axis=0)
            status = ('DETECTED' if corners is not None else 'not detected') + (' but ARM MOVING -> hold still' if corners is not None and spread > 0.6 else '')
            line = f"{time.time()-t0:5.1f}s {status} | " + '; '.join(hint)
            if line != last: print(line, flush=True); last = line
            preview.set_status(f'kept {existing + len(kept)} | {status} | ' + '; '.join(hint))
            if corners is not None and spread <= 0.6 and not any(np.max(np.abs(median - k)) < 5.0 for k in kept):
                label = f'ext_{existing + len(kept) + 1:02d}'; s = stamp[:19].replace(':', ''); png = args.output_dir / f'{s}_{label}.png'; cv2.imwrite(str(png), f)
                rec = dict(timestamp=stamp, mode='extrinsics', board=board, label=label, corners_px=corners.round(2).tolist(), joint_names=list(JOINT_NAMES),
                           normalized_median=median.tolist(), normalized_spread=np.ptp(samples, axis=0).tolist(), frame=str(png.relative_to(ROOT)), frame_sha256=hashlib.sha256(png.read_bytes()).hexdigest())
                (args.output_dir / f'{s}_{label}.json').write_text(json.dumps(rec, indent=1) + '\n'); kept.append(median); preview.flash()
                print(f"   >>> KEPT {label} (total {existing + len(kept)}) joints {np.round(median,1).tolist()} -> now move to a DIFFERENT pose", flush=True)
            time.sleep(0.8)
    try:
        run_with_preview(loop, preview)
    finally:
        grabber.close(); disconnect_read_only(robot)
    print(json.dumps(dict(kept_this_run=len(kept), total=existing + len(kept)))); return 0


if __name__ == '__main__':
    raise SystemExit(main())
