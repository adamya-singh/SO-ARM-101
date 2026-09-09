"""Intrinsic calibration of the wrist camera from checkerboard captures, with a validity check.

Reads the JSON records written by tools/capture_checkerboard.py --mode intrinsics
(any mix of board versions: object points are built per view from each record's
own ``board``), fits several OpenCV lens models, and writes camera_intrinsics.json.

What makes this fit trustworthy for a *full-frame* simulator lens model:

* **Valid radius.** A polynomial distortion model is only invertible where the
  distorted radius r_d(r) still increases with the undistorted radius r; beyond
  the peak of r_d(r) it describes nothing. The 2026-09-08 centre-only fit peaked
  at 929 px while the frame corners lie 1101 px from the centre, so the outer
  ~11 % of the frame had no model. Every candidate reports its valid radius
  and is only selectable if that radius exceeds the frame-corner radius by a
  margin.
* **Coverage.** The 4x3 coverage grid of tools/capture_checkerboard.py is
  recorded with the fit; edges and corners must actually contain corners.
* **Per-region residuals.** RMS reprojection error is reported separately for
  the centre, mid and outer annuli so extrapolation cannot hide behind a good
  overall RMS.

Candidates: 5-coefficient (k1 k2 p1 p2 k3) with the principal point fixed at the
image centre or free, the rational 8-coefficient model (fixed / free principal
point), and the equidistant fisheye model. ``selected`` is the lowest-RMS
candidate whose valid radius exceeds the corner radius by ``--valid-margin``.
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H = 1920, 1080
REGIONS = ((0.0, 400.0, 'centre'), (400.0, 800.0, 'mid'), (800.0, 1e9, 'outer'))


def object_points(board):
    cols, rows = board['inner_corners']
    s = board['square_mm'] / 1000.0
    return np.array([[c * s, r * s, 0.0] for r in range(rows) for c in range(cols)], dtype=np.float64)


def polynomial_valid_radius(K, dist):
    """Distorted radius (normalised, and px along fx) at the first peak of r_d(r) for the OpenCV pinhole models.

    Radial part only: r_d(r) = r * (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6).
    Returns (valid_r_d_normalised, valid_px, undistorted_r_at_peak). A monotonic model up to 80 deg
    returns the value at 80 deg.
    """
    flat = np.asarray(dist, dtype=np.float64).ravel()[:8]
    d = np.zeros(8); d[:len(flat)] = flat
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r = np.linspace(0.0, np.tan(np.deg2rad(80.0)), 20001)
    r2 = r * r
    rd = r * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    drop = np.nonzero(np.diff(rd) <= 0)[0]
    i = int(drop[0]) if len(drop) else len(rd) - 1
    return float(rd[i]), float(rd[i] * min(K[0][0], K[1][1])), float(r[i])


def fisheye_valid_radius(K, dist):
    """theta_d at the first peak of theta_d(theta) for the equidistant model, and the px equivalent."""
    d = np.asarray(dist, dtype=np.float64).ravel()
    th = np.linspace(0.0, np.pi / 2 - 1e-3, 20001)
    thd = th * (1 + d[0] * th ** 2 + d[1] * th ** 4 + d[2] * th ** 6 + d[3] * th ** 8)
    drop = np.nonzero(np.diff(thd) <= 0)[0]
    i = int(drop[0]) if len(drop) else len(thd) - 1
    return float(thd[i]), float(thd[i] * min(K[0][0], K[1][1])), float(th[i])


def corner_radius(K):
    """Largest distance from the principal point to a frame corner, normalised (along the smaller focal) and in px."""
    cx, cy = K[0][2], K[1][2]
    px = max(np.hypot(x - cx, y - cy) for x in (0.0, W - 1.0) for y in (0.0, H - 1.0))
    return float(px / min(K[0][0], K[1][1])), float(px)


def region_rms(errors_px, radii_px):
    out = {}
    for lo, hi, name in REGIONS:
        m = (radii_px >= lo) & (radii_px < hi)
        out[name] = dict(corners=int(m.sum()), rms_px=(round(float(np.sqrt(np.mean(errors_px[m] ** 2))), 3) if m.any() else None))
    return out


def main(argv=None) -> int:
    import cv2
    from capture_checkerboard import coverage_grid, coverage_hint, CELL_NAMES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/intrinsics')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/camera_intrinsics.json')
    p.add_argument('--valid-margin', type=float, default=1.05, help='selected model must have valid radius >= margin x frame-corner radius')
    p.add_argument('--coverage-target', type=int, default=4, help='views per 4x3 cell wanted before the fit counts as full-frame evidence')
    p.add_argument('--min-outer-corners', type=int, default=200, help='corners beyond 800 px from the principal point required before a model may claim full-frame validity')
    p.add_argument('--ignore-cells', default='', help='comma-separated coverage cells that cannot be observed (e.g. Bl,Br: the fixed gripper jaw fills them) and are excluded from the coverage requirement')
    args = p.parse_args(argv)
    records = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(args.input_dir / '*.json')))]
    if len(records) < 8:
        raise SystemExit(f'need at least 8 board views, have {len(records)}')
    obj, img, used = [], [], []
    for rec in records:
        c = np.array(rec['corners_px'], dtype=np.float64)
        o = object_points(rec['board'])
        if c.shape[0] != o.shape[0]:
            continue
        obj.append(o.astype(np.float32)); img.append(c.astype(np.float32).reshape(-1, 1, 2)); used.append(rec)
    boards = sorted({f"{r['board']['inner_corners'][0]}x{r['board']['inner_corners'][1]}@{r['board']['square_mm']}mm" for r in used})
    grid = coverage_grid(used)
    ignored = {c.strip() for c in args.ignore_cells.split(',') if c.strip()}
    def missing_cells():
        return [c for c in coverage_hint(grid, args.coverage_target) if c not in ignored]
    print(f'{len(img)} views ({len(records) - len(img)} skipped for corner-count mismatch); boards {boards}')
    print(f'coverage grid (rows top->bottom, cols left->right):\n{grid}\ncells below target {args.coverage_target}: {coverage_hint(grid, args.coverage_target)} (ignored: {sorted(ignored)})')

    def evaluate(name, rms, K, dist, rvecs, tvecs, fisheye=False):
        errs, radii, per_view = [], [], []
        for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
            if fisheye:
                proj, _ = cv2.fisheye.projectPoints(o.reshape(-1, 1, 3).astype(np.float64), rv, tv, K, dist)
            else:
                proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
            e = np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
            errs.append(e); radii.append(np.hypot(i.reshape(-1, 2)[:, 0] - K[0, 2], i.reshape(-1, 2)[:, 1] - K[1, 2]))
            per_view.append(round(float(np.sqrt(np.mean(e ** 2))), 2))
        errs = np.concatenate(errs); radii = np.concatenate(radii)
        vr_norm, vr_px, r_peak = (fisheye_valid_radius(K, dist) if fisheye else polynomial_valid_radius(K, dist))
        c_norm, c_px = corner_radius(K)
        fovy = 2 * np.degrees(np.arctan(H / (2 * K[1, 1]))); fovx = 2 * np.degrees(np.arctan(W / (2 * K[0, 0])))
        return dict(model=name, rms_px=round(float(rms), 3), K=np.asarray(K).round(3).tolist(), dist=np.asarray(dist).ravel().round(6).tolist(),
                    fovx_deg=round(float(fovx), 2), fovy_deg=round(float(fovy), 2), principal_point=[round(float(K[0, 2]), 1), round(float(K[1, 2]), 1)],
                    image_size=[W, H], per_view_rms_px=per_view, region_rms_px=region_rms(errs, radii),
                    valid_radius_normalised=round(vr_norm, 4), valid_radius_px=round(vr_px, 1), undistorted_radius_at_peak=round(r_peak, 4),
                    corner_radius_normalised=round(c_norm, 4), corner_radius_px=round(c_px, 1),
                    # A model may claim the full frame only when (a) it is invertible out to the corners AND
                    # (b) the data actually reached the outer annulus and every coverage cell: a polynomial's
                    # valid radius is a property of the fit, not evidence, so extrapolation alone never qualifies.
                    model_reaches_corners=bool(vr_norm >= args.valid_margin * c_norm),
                    outer_corners=int((radii >= 800.0).sum()),
                    covers_full_frame=bool(vr_norm >= args.valid_margin * c_norm and (radii >= 800.0).sum() >= args.min_outer_corners and not missing_cells()))

    candidates = {}
    K0 = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1.0]])
    fits = [
        ('pinhole_5coef_fixed_principal_point', cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS, K0.copy()),
        ('pinhole_5coef_free_principal_point', cv2.CALIB_USE_INTRINSIC_GUESS, K0.copy()),
        ('pinhole_rational_8coef_fixed_principal_point', cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS, K0.copy()),
        ('pinhole_rational_8coef_free_principal_point', cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS, K0.copy()),
    ]
    for name, flags, Kinit in fits:
        try:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (W, H), Kinit, None, flags=flags)
            candidates[name] = evaluate(name, rms, K, dist, rvecs, tvecs)
        except cv2.error as exc:
            candidates[name] = dict(model=name, error=str(exc)[:200])
    try:
        Kf = np.zeros((3, 3)); Df = np.zeros((4, 1))
        fl = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms_f, Kf, Df, rv_f, tv_f = cv2.fisheye.calibrate([o.reshape(-1, 1, 3).astype(np.float64) for o in obj], [i.reshape(-1, 1, 2).astype(np.float64) for i in img], (W, H), Kf, Df, flags=fl,
                                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        candidates['fisheye_equidistant'] = evaluate('fisheye_equidistant', rms_f, Kf, Df, rv_f, tv_f, fisheye=True)
    except cv2.error as exc:
        candidates['fisheye_equidistant'] = dict(model='fisheye_equidistant', error=str(exc)[:200])

    for name, c in candidates.items():
        if 'error' in c:
            print(f'{name}: FAILED {c["error"]}'); continue
        regions = ', '.join(f"{k}: {v['rms_px']} ({v['corners']})" for k, v in c['region_rms_px'].items())
        verdict = 'covers frame' if c['covers_full_frame'] else ('reaches corners but coverage evidence is missing' if c['model_reaches_corners'] else 'DOES NOT reach the corners')
        print(f"{name}: rms {c['rms_px']} px | valid radius {c['valid_radius_px']} px vs corner {c['corner_radius_px']} px -> {verdict} | "
              f"regions {{{regions}}} | fovy {c['fovy_deg']} fovx {c['fovx_deg']} pp {c['principal_point']}")
    eligible = [c for c in candidates.values() if 'error' not in c and c['covers_full_frame']]
    if eligible:
        selected = min(eligible, key=lambda c: c['rms_px']); selection = 'lowest RMS among models whose valid radius covers the full frame'
    else:
        selected = candidates['pinhole_5coef_fixed_principal_point']; selection = 'FALLBACK: no candidate has full-frame validity AND coverage evidence; centre-region model kept, do not build a full-frame lens on it'
    report = dict(written_at=datetime.now(timezone.utc).isoformat(), views=len(img), boards=boards, coverage_grid=grid.tolist(), coverage_cell_names=CELL_NAMES,
                  coverage_below_target=coverage_hint(grid, args.coverage_target), coverage_ignored_cells=sorted(ignored), coverage_target=args.coverage_target,
                  selection_rule=selection, valid_margin=args.valid_margin, selected=selected, candidates=candidates,
                  fisheye=candidates.get('fisheye_equidistant'),
                  frames=[dict(frame=r['frame'], sha256=r['frame_sha256']) for r in used])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"selected: {selected['model']} ({selection}) -> {args.output}")
    return 0 if eligible else 2


if __name__ == '__main__':
    raise SystemExit(main())
