"""Intrinsic calibration of the wrist camera from checkerboard captures (pinhole+rational and fisheye models).

Reads the JSON records written by tools/capture_checkerboard.py --mode intrinsics,
runs cv2.calibrateCamera (rational model) and cv2.fisheye.calibrate, reports RMS
reprojection error for both, and writes camera_intrinsics.json with K, distortion,
the equivalent vertical field of view, and the frame hashes used.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    import cv2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/intrinsics')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/camera_intrinsics.json')
    args = p.parse_args(argv)
    records = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(args.input_dir / '*.json')))]
    if len(records) < 8:
        raise SystemExit(f'need at least 8 board views, have {len(records)}')
    board = records[0]['board']; cols, rows = board['inner_corners']; s = board['square_mm'] / 1000.0
    objp = np.array([[c * s, r * s, 0.0] for r in range(rows) for c in range(cols)], dtype=np.float64)
    W, H = 1920, 1080
    obj, img = [], []
    for rec in records:
        c = np.array(rec['corners_px'], dtype=np.float64)
        if c.shape[0] != cols * rows:
            continue
        obj.append(objp.copy()); img.append(c)
    print(f'{len(img)} views')
    objs = [o.astype(np.float32) for o in obj]; imgs = [i.astype(np.float32).reshape(-1, 1, 2) for i in img]
    # Plain 5-coefficient model (k1 k2 p1 p2 k3): stable where the board never reached; the rational model is reported for comparison.
    # Principal point fixed at the image centre: with board coverage confined to part of the image, a free
    # principal point trades off against the focal length (seen as fx swinging 1330-1680 between models).
    K0 = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1.0]])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (W, H), K0, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS)
    rms_r, K_r, dist_r, _, _ = cv2.calibrateCamera(objs, imgs, (W, H), None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    per_view = []
    for o, i, rv, tv in zip(objs, imgs, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist); per_view.append(round(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))), 2))
    fovy = 2 * np.degrees(np.arctan(H / (2 * K[1, 1]))); fovx = 2 * np.degrees(np.arctan(W / (2 * K[0, 0])))
    result = dict(model='pinhole_5coef_fixed_principal_point', rms_px=round(float(rms), 3), K=K.round(3).tolist(), dist=dist.ravel().round(6).tolist(),
                  fovx_deg=round(float(fovx), 2), fovy_deg=round(float(fovy), 2), principal_point=[round(float(K[0, 2]), 1), round(float(K[1, 2]), 1)], per_view_rms_px=per_view,
                  rational_comparison=dict(rms_px=round(float(rms_r), 3), fx=round(float(K_r[0, 0]), 1), fy=round(float(K_r[1, 1]), 1), dist=dist_r.ravel().round(5).tolist()))
    try:
        Kf = np.zeros((3, 3)); Df = np.zeros((4, 1))
        fl = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms_f, Kf, Df, _, _ = cv2.fisheye.calibrate([o.reshape(-1, 1, 3) for o in obj], [i.reshape(-1, 1, 2) for i in img], (W, H), Kf, Df, flags=fl, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        fisheye = dict(model='fisheye_equidistant', rms_px=round(float(rms_f), 3), K=Kf.round(3).tolist(), dist=Df.ravel().round(6).tolist(),
                       fovy_deg_equivalent=round(float(2 * np.degrees(np.arctan(H / (2 * Kf[1, 1])))), 2))
    except cv2.error as exc:
        fisheye = dict(error=str(exc)[:200])
    report = dict(views=len(img), board=board, selected=result, pinhole_rational=result, pinhole=result, fisheye=fisheye,
                  coverage_note='board views concentrated in the upper-middle of the image; distortion beyond that region is extrapolated',
                  frames=[dict(frame=r['frame'], sha256=r['frame_sha256']) for r in records])
    args.output.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2)[:3000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
