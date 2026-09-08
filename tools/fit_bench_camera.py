"""Fit the simulated wrist camera (pose in the gripper frame + vertical field of view) to physical frames.

Inputs are camera_references records: each pairs a physical wrist frame with a
read-only joint reading taken at the same instant. For every record the tool
converts the joints to the model pose through the bench joint map, detects the
white square's corners in the frame (convex hull of the largest bright, neutral
blob; corners on the image border are discarded), and associates them with the
square's world corners. It then solves for camera position (3), orientation (3)
and fovy (1) by Levenberg-Marquardt on the reprojection error, trying the
corner-ordering hypotheses and keeping the best. Output: a JSON report with the
fitted camera, per-frame residuals, and the MuJoCo <camera> attributes to paste
into the scene. Simulation plus images only; touches no hardware and edits no
scene by itself.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080


def rotvec_to_matrix(r):
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def matrix_to_quat(R):
    q = np.empty(4); t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; q[0] = 0.25 * s; q[1] = (R[2, 1] - R[1, 2]) / s; q[2] = (R[0, 2] - R[2, 0]) / s; q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        v = np.empty(3); v[i] = 0.25 * s; v[j] = (R[j, i] + R[i, j]) / s; v[k] = (R[k, i] + R[i, k]) / s
        q[0] = (R[k, j] - R[j, k]) / s; q[1:] = v
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def detect_square_corners(path: Path):
    """Corners of the white napkin: the bright, neutral blob that has the dark cube inside it.

    Rejects other white objects (a book cover, paper) by requiring a compact dark
    component well inside the blob's convex hull. Returns (corners, status) with
    status 'full' (four corners off the border), 'cropped', or 'none'.
    """
    import cv2
    img = cv2.imread(str(path)); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    dark = (hsv[:, :, 2] < 70).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4000:
            continue
        mask = (lab == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_mask = np.zeros_like(mask); cv2.fillConvexPoly(hull_mask, hull, 1)
        hull_area = float(hull_mask.sum())
        if hull_area <= 0 or area / hull_area < 0.6:
            continue
        # a compact dark component inside the hull, away from its boundary = the cube
        inner = cv2.erode(hull_mask, np.ones((15, 15), np.uint8))
        dn, dlab, dstats, _ = cv2.connectedComponentsWithStats((dark & inner).astype(np.uint8), connectivity=8)
        cube_like = False
        for j in range(1, dn):
            a = dstats[j, cv2.CC_STAT_AREA]; w = dstats[j, cv2.CC_STAT_WIDTH]; hh = dstats[j, cv2.CC_STAT_HEIGHT]
            if 0.01 * hull_area <= a <= 0.35 * hull_area and 0.5 <= w / max(hh, 1) <= 2.0 and a / max(w * hh, 1) > 0.5:
                cube_like = True; break
        if not cube_like:
            continue
        if best is None or area > best[0]:
            best = (area, hull)
    if best is None:
        return [], 'none'
    hull = best[1]
    poly = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(float)
    margin = 12
    inside = [tuple(p) for p in poly if margin < p[0] < W - margin and margin < p[1] < H - margin]
    touches = len(inside) < len(poly)
    merged = []
    for p in inside:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 25 for q in merged):
            merged.append(p)
    return merged, ('cropped' if touches else 'full')


def frame_pose_with_offsets(frame, model, data, offsets_rad, mount_body='gripper'):
    """Recompute (mount pos, mount rot, finger ends) for a frame with joint-zero corrections added to joints 0-4."""
    import mujoco
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    q = frame.qpos.copy(); q[:5] += np.asarray(offsets_rad)
    for name, value in zip(JOINT_NAMES, q):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(model, data)
    pos = data.body(mount_body).xpos.copy(); rot = data.body(mount_body).xmat.reshape(3, 3).copy(); ends = []
    for geom in ('fixed_jaw_pad_1', 'moving_jaw_pad_1'):
        gid = model.geom(geom).id; R = data.geom_xmat[gid].reshape(3, 3); half = float(model.geom_size[gid][1])
        ends.append(data.geom_xpos[gid] + R[:, 1] * (-half))
    return pos, rot, ends


def order_clockwise(points):
    c = np.mean(points, axis=0); ang = np.arctan2([p[1] - c[1] for p in points], [p[0] - c[0] for p in points])
    return [points[i] for i in np.argsort(ang)]


class Frame:
    def __init__(self, record, bench, model, data, mount_body='gripper'):
        import mujoco
        from so_arm101_v2.contracts.coordinates import JOINT_NAMES
        from so_arm101_v2.contracts.physical import physical_normalized_to_act
        self.label = record['label']; self.path = ROOT / record['frame']
        qpos = bench.joint_map_object.act_to_mujoco(physical_normalized_to_act(record['normalized_median']))
        self.qpos = np.asarray(qpos, dtype=np.float64).copy(); self.normalized = list(record['normalized_median'])
        for name, value in zip(JOINT_NAMES, qpos):
            data.qpos[model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(model, data)
        self.mount_body = mount_body
        self.gripper_pos = data.body(mount_body).xpos.copy(); self.gripper_rot = data.body(mount_body).xmat.reshape(3, 3).copy()
        self.tips = {'fixed_jaw_tip': data.site('fixed_jaw_tip').xpos.copy(), 'moving_jaw_tip': data.site('moving_jaw_tip').xpos.copy()}
        # Visible finger ends: outer end of pad 1 on each jaw (pad centre + half length along the jaw depth axis), world frame.
        self.finger_ends = []
        for geom in ('fixed_jaw_pad_1', 'moving_jaw_pad_1'):
            gid = model.geom(geom).id; R = data.geom_xmat[gid].reshape(3, 3); half = float(model.geom_size[gid][1])
            self.finger_ends.append(data.geom_xpos[gid] + R[:, 1] * (-half))
        self.corners_px, self.status = detect_square_corners(self.path)
        self.corners_px = order_clockwise(self.corners_px) if len(self.corners_px) >= 2 else self.corners_px


def project(cam_pos_g, cam_rot_g, fovy, frame, points_w, k1=0.0):
    """Pinhole projection with one radial distortion term (k1 on normalized image coordinates)."""
    f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
    Rw = frame.gripper_rot @ cam_rot_g; pw = frame.gripper_pos + frame.gripper_rot @ cam_pos_g
    out = []
    for p in points_w:
        pc = Rw.T @ (np.asarray(p) - pw)
        if pc[2] >= -1e-6:
            out.append((np.nan, np.nan)); continue
        xn, yn = pc[0] / -pc[2], pc[1] / -pc[2]; scale = 1.0 + k1 * (xn * xn + yn * yn)
        out.append((W / 2 + f * xn * scale, H / 2 - f * yn * scale))
    return np.array(out)


def pnp_initial_guess(full_frames, world_corners, fovy_grid):
    """Planar PnP per frame over a fovy grid; choose fovy + corner orderings making the camera-in-gripper pose agree.

    Returns (pos_g, rot_g, fovy, orderings, spread_mm) or None. OpenCV camera axes (z forward, y down) are
    converted to MuJoCo camera axes (-z forward, y up).
    """
    import cv2
    if not full_frames:
        return None
    obj = np.asarray(world_corners, dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0])
    best = None
    for fovy in fovy_grid:
        f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2); K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        per_frame = []  # list of lists of (ordering_index, pos_g, rot_g)
        for fr in full_frames:
            options = []
            c = fr.corners_px
            for idx, (rot, rev) in enumerate(itertools.product(range(4), (False, True))):
                seq = c[::-1] if rev else c; seq = seq[rot:] + seq[:rot]
                img = np.asarray(seq, dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
                if not ok:
                    continue
                Rcw, _ = cv2.Rodrigues(rvec); Rwc = Rcw.T; C = (-Rwc @ tvec).ravel()
                R_mj = Rwc @ flip
                # reject cameras looking away from the square (square must be in front: z_c < 0 in MuJoCo camera frame)
                if (R_mj.T @ (obj.mean(axis=0) - C))[2] >= 0:
                    continue
                pos_g = fr.gripper_rot.T @ (C - fr.gripper_pos); rot_g = fr.gripper_rot.T @ R_mj
                if np.linalg.norm(pos_g) > 0.25:  # a wrist camera sits within 25 cm of the gripper body
                    continue
                options.append((idx, pos_g, rot_g))
            if not options:
                per_frame = None; break
            per_frame.append(options)
        if per_frame is None:
            continue
        # Greedy association: seed with each option of the first frame, then for
        # every other frame take the option whose camera-in-gripper pose is
        # nearest the seed. Avoids the 8^n joint enumeration.
        def pose_distance(a, b):
            return np.linalg.norm(a[1] - b[1]) * 1000 + 2 * np.degrees(np.arccos(np.clip((np.trace(a[2].T @ b[2]) - 1) / 2, -1, 1)))
        for seed in per_frame[0]:
            combo = [seed] + [min(options, key=lambda o: pose_distance(o, seed)) for options in per_frame[1:]]
            P = np.array([o[1] for o in combo]); Rs = [o[2] for o in combo]
            spread = float(np.mean([np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))])) if len(P) > 1 else 0.0
            rot_spread = float(np.mean([np.degrees(np.arccos(np.clip((np.trace(Rs[i].T @ Rs[j]) - 1) / 2, -1, 1))) for i in range(len(Rs)) for j in range(i + 1, len(Rs))])) if len(Rs) > 1 else 0.0
            score = spread * 1000 + rot_spread * 2  # mm + 2*deg
            if best is None or score < best[0]:
                Rsum = sum(Rs); U, _, Vt = np.linalg.svd(Rsum); Ravg = U @ Vt
                if np.linalg.det(Ravg) < 0:
                    U[:, -1] *= -1; Ravg = U @ Vt
                best = (score, P.mean(axis=0), Ravg, fovy, [o[0] for o in combo], spread * 1000, rot_spread)
    if best is None:
        return None
    _, pos, rot, fovy, orderings, spread_mm, rot_spread = best
    return pos, rot, fovy, orderings, spread_mm, rot_spread


def main(argv=None) -> int:
    import mujoco
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    p.add_argument('--references', type=Path, nargs='+', required=True)
    p.add_argument('--fix-fovy', type=float, default=None, help='hold the vertical field of view fixed (deg) instead of fitting it')
    p.add_argument('--fingertips', default=None,
                   help='pixel positions of the two visible jaw tips, "x1,y1;x2,y2" (they are fixed in every frame because the camera is rigid to the gripper); both tip assignments are tried')
    p.add_argument('--fingertip-weight', type=float, default=0.5)
    p.add_argument('--fit-k1', action='store_true', help='also fit one radial distortion coefficient')
    p.add_argument('--multistart', type=int, default=0, help='number of random camera-in-gripper initialisations to try in addition to the PnP guess')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/camera_fit.json')
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model); model = mujoco.MjModel.from_xml_path(str(args.model)); data = mujoco.MjData(model)
    frames = [Frame(json.loads(r.read_text()), bench, model, data) for r in args.references]
    for fr in frames:
        print(f"{fr.label}: {len(fr.corners_px)} square corners detected ({fr.status})")
    sq = np.array([*bench.square_center_xy, bench.square_thickness_m]); h = bench.square_edge_m / 2
    world_corners = [sq + np.array([sx * h, sy * h, 0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]  # CCW from above
    near = [world_corners[0], world_corners[1]]  # edge nearest the robot base (smaller y)
    cid = model.camera('wrist_camera').id
    cam0_pos = model.cam_pos[cid].copy(); cam0_rot = quat_to_matrix(model.cam_quat[cid]); fovy0 = float(model.cam_fovy[cid])
    # Visible jaw tips in the GRIPPER frame: the outer end of pad 1 on each jaw (pad centre + half length along the depth axis).
    fixed_pad = model.geom('fixed_jaw_pad_1'); moving_pad = model.geom('moving_jaw_pad_1')
    tip_fixed_g = np.asarray(fixed_pad.pos) + np.array([0, -float(fixed_pad.size[1]), 0])
    # moving jaw body frame -> gripper frame using the pose at gripper qpos of the first frame (jaw angle matters little for the tip)
    mj_body = model.body('moving_jaw_so101_v1'); Rm = quat_to_matrix(mj_body.quat); tip_moving_g = np.asarray(mj_body.pos) + Rm @ (np.asarray(moving_pad.pos) + np.array([0, -float(moving_pad.size[1]), 0]))
    tips_px = None
    if args.fingertips:
        tips_px = [tuple(float(v) for v in pair.split(',')) for pair in args.fingertips.split(';')]
        assert len(tips_px) == 2

    def project_gripper_points(pos, rot, fovy, pts_g):
        f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2); out = []
        for pg in pts_g:
            pc = rot.T @ (np.asarray(pg) - pos)
            out.append((np.nan, np.nan) if pc[2] >= -1e-6 else (W / 2 + f * pc[0] / -pc[2], H / 2 - f * pc[1] / -pc[2]))
        return np.array(out)

    def residuals(params, assignments, tip_order=0):
        pos = params[:3]; rot = rotvec_to_matrix(params[3:6]) @ cam0_rot; fovy = args.fix_fovy if args.fix_fovy else params[6]
        k1 = params[7] if args.fit_k1 else 0.0
        res = []
        if tips_px is not None:
            order = [tip_fixed_g, tip_moving_g] if tip_order == 0 else [tip_moving_g, tip_fixed_g]
            for pr, obs in zip(project_gripper_points(pos, rot, fovy, order), tips_px):
                res.extend([2000.0, 2000.0] if np.any(np.isnan(pr)) else [args.fingertip_weight * (pr[0] - obs[0]), args.fingertip_weight * (pr[1] - obs[1])])
        for fr, assign in zip(frames, assignments):
            if not assign:
                continue
            pts_w = [w for w, _ in assign]; px = project(pos, rot, fovy, fr, pts_w, k1)
            for (w, obs), pr in zip(assign, px):
                if np.any(np.isnan(pr)):
                    res.extend([2000.0, 2000.0])
                else:
                    res.extend([pr[0] - obs[0], pr[1] - obs[1]])
        return np.array(res)

    init_pos, init_rot, init_fovy = cam0_pos, cam0_rot, fovy0
    full_frames = [fr for fr in frames if len(fr.corners_px) == 4]
    pnp = pnp_initial_guess(full_frames, world_corners, np.arange(20.0, 111.0, 2.0))
    if pnp is not None:
        init_pos, init_rot, init_fovy, pnp_orderings, pnp_spread_mm, pnp_rot_spread = pnp
        print(f'PnP initial guess: fovy {init_fovy:.0f} deg, camera-in-gripper spread across frames {pnp_spread_mm:.1f} mm / {pnp_rot_spread:.1f} deg')
    cam0_rot = init_rot  # rotation increments are applied on top of the initial rotation

    def angular_residuals(params, assignments):
        """Smooth everywhere: difference between the unit ray to each world corner and the observed pixel ray."""
        pos = params[:3]; rot = rotvec_to_matrix(params[3:6]) @ cam0_rot; fovy = args.fix_fovy if args.fix_fovy else params[6]
        f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2); res = []
        for fr, assign in zip(frames, assignments):
            if not assign:
                continue
            Rw = fr.gripper_rot @ rot; pw = fr.gripper_pos + fr.gripper_rot @ pos
            for w, obs in assign:
                d = Rw.T @ (np.asarray(w) - pw); d = d / np.linalg.norm(d)
                o = np.array([(obs[0] - W / 2) / f, -(obs[1] - H / 2) / f, -1.0]); o = o / np.linalg.norm(o)
                res.extend((d - o).tolist())
        return np.array(res)

    def lm(fn, params, mask, iters=200):
        lam = 1e-2; r = fn(params); cost = float(r @ r)
        for _ in range(iters):
            J = np.zeros((len(r), len(params)))
            for k in np.flatnonzero(mask):
                dp = np.zeros(len(params)); dp[k] = 1e-4 if k < 6 else (1e-2 if k == 6 else 1e-4)
                J[:, k] = (fn(params + dp) - r) / dp[k]
            A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12); g = J.T @ r
            step = np.zeros(len(params)); idx = np.flatnonzero(mask)
            step[idx] = -np.linalg.solve(A[np.ix_(idx, idx)], g[idx])
            trial = params + step; rt = fn(trial); ct = float(rt @ rt)
            if ct < cost:
                params, r, cost, lam = trial, rt, ct, max(lam / 3, 1e-6)
                if np.abs(step).max() < 1e-8:
                    break
            else:
                lam *= 4
                if lam > 1e8:
                    break
        return params, cost, r

    def solve(assignments, tip_order=0):
        params = np.concatenate([init_pos, np.zeros(3), [init_fovy], [0.0]]); lam = 1e-2
        r = residuals(params, assignments, tip_order); cost = float(r @ r)
        for _ in range(200):
            J = np.empty((len(r), 8))
            for k in range(8):
                dp = np.zeros(8); dp[k] = 1e-4 if k < 3 else (1e-4 if k < 6 else (1e-2 if k == 6 else 1e-4))
                J[:, k] = (residuals(params + dp, assignments, tip_order) - r) / dp[k]
            if args.fix_fovy:
                J[:, 6] = 0
            if not args.fit_k1:
                J[:, 7] = 0
            A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9); g = J.T @ r
            step = -np.linalg.solve(A, g)
            if args.fix_fovy:
                step[6] = 0
            if not args.fit_k1:
                step[7] = 0
            trial = params + step; rt = residuals(trial, assignments, tip_order); ct = float(rt @ rt)
            if ct < cost:
                params, r, cost, lam = trial, rt, ct, max(lam / 3, 1e-6)
                if abs(step).max() < 1e-7:
                    break
            else:
                lam *= 4
                if lam > 1e8:
                    break
        return params, cost, r

    # Hypotheses: per frame, corner association. Full squares: 4 rotations x 2 orientations.
    # Cropped (2 corners): they are the near edge; 2 orientations.
    per_frame_options = []
    for fr in frames:
        c = fr.corners_px
        if len(c) == 4:
            opts = []
            for rot in range(4):
                for rev in (False, True):
                    seq = c[::-1] if rev else c
                    seq = seq[rot:] + seq[:rot]
                    opts.append(list(zip(world_corners, seq)))
            per_frame_options.append(opts)
        elif len(c) == 2:
            per_frame_options.append([list(zip(near, c)), list(zip(near, c[::-1]))])
        else:
            per_frame_options.append([[]])
    # Choose each frame's corner ordering independently (fit that frame alone,
    # with the fingertip anchor when given), then refit all frames jointly with
    # the chosen orderings. This avoids the combinatorial joint search.
    chosen = []
    tip_orders = (0, 1) if tips_px is not None else (0,)
    full_iter = iter(pnp_orderings) if pnp is not None else None
    for fi, opts in enumerate(per_frame_options):
        if opts == [[]]:
            chosen.append([]); continue
        if len(frames[fi].corners_px) == 4 and full_iter is not None:
            chosen.append(opts[next(full_iter)]); continue
        best_local = None
        for opt in opts:
            for tip_order in tip_orders:
                combo = [opt if k == fi else [] for k in range(len(frames))]
                params, cost, r = solve(combo, tip_order)
                if best_local is None or cost < best_local[0]:
                    best_local = (cost, opt)
        chosen.append(best_local[1])
    def solve_from(init_p, init_R, assignments, tip_order=0, iters=200):
        nonlocal init_pos, cam0_rot
        saved = (init_pos, cam0_rot); init_pos, cam0_rot = init_p, init_R
        try:
            return solve(assignments, tip_order)
        finally:
            init_pos, cam0_rot = saved

    best = None
    for tip_order in tip_orders:
        params, cost, r = solve(chosen, tip_order)
        if best is None or cost < best[1]:
            best = (params, cost, r, tuple(chosen), tip_order, init_pos, cam0_rot)
    if args.multistart:
        rng = np.random.default_rng(args.seed)
        gripper_depth = np.array([0.0, -1.0, 0.0])  # palm -> tips in the gripper frame
        for k in range(args.multistart):
            # position: within 15 cm of the gripper body; orientation: random, but biased so the camera looks
            # somewhere between straight along the fingers and 90 deg off it (a wrist camera always sees the tips).
            pos_k = rng.uniform(-0.15, 0.15, size=3)
            q = rng.normal(size=4); q /= np.linalg.norm(q); R_k = quat_to_matrix(q)
            forward = -R_k[:, 2]
            if np.dot(forward, gripper_depth) < 0.0:
                continue
            # Stage 1: angular LM from the random start (smooth); stage 2: pixel LM from where it lands.
            saved = (init_pos, cam0_rot); init_pos, cam0_rot = pos_k, R_k
            try:
                mask = np.array([1, 1, 1, 1, 1, 1, 0 if args.fix_fovy else 1, 0], dtype=bool)
                p0 = np.concatenate([pos_k, np.zeros(3), [args.fix_fovy or init_fovy], [0.0]])
                p1, c1, _ = lm(lambda q: angular_residuals(q, chosen), p0, mask, iters=80)
                start_pos = p1[:3]; start_rot = rotvec_to_matrix(p1[3:6]) @ R_k
            finally:
                init_pos, cam0_rot = saved
            for tip_order in tip_orders:
                params, cost, r = solve_from(start_pos, start_rot, chosen, tip_order)
                if cost < best[1]:
                    best = (params, cost, r, tuple(chosen), tip_order, start_pos, start_rot)
    params, cost, r, combo, tip_order, init_pos, cam0_rot = best
    pos = params[:3]; rot = rotvec_to_matrix(params[3:6]) @ cam0_rot; fovy = args.fix_fovy if args.fix_fovy else float(params[6])
    k1 = float(params[7]) if args.fit_k1 else 0.0
    quat = matrix_to_quat(rot)
    men_rot = quat_to_matrix(model.cam_quat[cid]); men_pos = model.cam_pos[cid].copy()
    rel = men_rot.T @ rot; rel_angle = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
    n_obs = len(r) // 2; rms = float(np.sqrt(cost / max(n_obs, 1)))
    report = dict(scene_dependencies_sha256=scene_dependency_hash(args.model), joint_map=bench.joint_map,
                  references=[str(r_) for r_ in args.references], observations=n_obs, rms_px=round(rms, 2),
                  fitted=dict(pos=[round(float(v), 5) for v in pos], quat_wxyz=[round(float(v), 6) for v in quat], fovy_deg=round(fovy, 2), k1=round(k1, 4)),
                  menagerie=dict(pos=[round(float(v), 5) for v in men_pos], quat_wxyz=[round(float(v), 6) for v in model.cam_quat[cid]], fovy_deg=fovy0),
                  camera_offset_from_menagerie_mm=[round(float(v) * 1000, 1) for v in (pos - men_pos)],
                  pnp_initial=(dict(fovy_deg=float(init_fovy), spread_mm=round(pnp_spread_mm, 1), rot_spread_deg=round(pnp_rot_spread, 1)) if pnp is not None else None),
                  fingertips=dict(observed_px=tips_px, order='fixed,moving' if tip_order == 0 else 'moving,fixed', weight=args.fingertip_weight) if tips_px is not None else None,
                  rotation_from_menagerie_deg=round(rel_angle, 1),
                  mujoco_camera_xml=f'<camera name="wrist_camera" mode="fixed" pos="{pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f}" quat="{quat[0]:.7f} {quat[1]:.7f} {quat[2]:.7f} {quat[3]:.7f}" fovy="{fovy:.2f}" />',
                  per_frame=[])
    for fr, assign in zip(frames, combo):
        if not assign:
            report['per_frame'].append(dict(label=fr.label, status=fr.status, used=0)); continue
        px = project(pos, rot, fovy, fr, [w for w, _ in assign], k1)
        errs = [float(np.hypot(pr[0] - o[0], pr[1] - o[1])) for (w, o), pr in zip(assign, px)]
        tips = project(pos, rot, fovy, fr, list(fr.tips.values()), k1)
        report['per_frame'].append(dict(label=fr.label, status=fr.status, used=len(assign), corner_errors_px=[round(e, 1) for e in errs],
                                        detected_px=[[round(v, 1) for v in o] for _, o in assign],
                                        projected_jaw_tips_px={k: [round(float(v), 1) for v in t] for k, t in zip(fr.tips, tips)}))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
