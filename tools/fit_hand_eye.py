"""Hand-eye plus joint-zero calibration from flat-checkerboard frames.

Inputs: calibrated intrinsics (camera_intrinsics.json) and the extrinsics
captures from tools/capture_checkerboard.py --mode extrinsics (board flat on the
table, arm posed by hand, joints recorded per frame). Corners are undistorted
with the calibrated model, then a single Levenberg-Marquardt solve estimates the
camera pose in the gripper frame (6), the board pose in the world (6) and small
zero corrections for the five arm joints (5, Gaussian prior), by minimising the
reprojection error of every corner in every frame through the arm's forward
kinematics under the bench joint map. Reports RMS, per-frame errors, the joint
corrections in degrees, and the MuJoCo <camera> line. Simulation plus images.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotvec_to_matrix(r):
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_to_rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)); ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ax / (np.linalg.norm(ax) + 1e-12) * ang


def matrix_to_quat(R):
    q = np.empty(4); t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; q[0] = 0.25 * s; q[1] = (R[2, 1] - R[1, 2]) / s; q[2] = (R[0, 2] - R[2, 0]) / s; q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        v = np.empty(3); v[i] = 0.25 * s; v[j] = (R[j, i] + R[i, j]) / s; v[k] = (R[k, i] + R[i, k]) / s; q[0] = (R[k, j] - R[j, k]) / s; q[1:] = v
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main(argv=None) -> int:
    import cv2
    import mujoco
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    from so_arm101_v2.contracts.physical import physical_normalized_to_act
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    p.add_argument('--intrinsics', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/camera_intrinsics.json')
    p.add_argument('--input-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/extrinsics')
    p.add_argument('--joint-prior-deg', type=float, default=5.0); p.add_argument('--no-joint-offsets', action='store_true')
    p.add_argument('--board-z', type=float, default=0.0085, help='height of the phone screen above the world floor (m); phone thickness ~8.5 mm on the mousepad')
    p.add_argument('--free-board', action='store_true', help='fit the full 6-DoF board pose instead of a flat board at --board-z (x, y, yaw only)')
    p.add_argument('--mount-prior-mm', type=float, default=15.0, help='soft prior width tying the camera position to the Menagerie mount camera (0 = off)')
    p.add_argument('--mount-prior-weight-px', type=float, default=20.0)
    p.add_argument('--no-closed-form-init', action='store_true', help='skip the Park-Martin hand-eye initialisation from per-frame PnP')
    p.add_argument('--fix-camera-pos', action='store_true', help='hold the camera position at the Menagerie mount value (official mount); fit orientation only')
    p.add_argument('--camera-rot-prior-deg', type=float, default=0.0, help='soft prior width (deg) on the camera orientation about the rolled-180 Menagerie orientation (0 = off)')
    p.add_argument('--board-init-xy', type=float, nargs=2, default=None, help='initial board CENTRE x y (m); default = bench square centre')
    p.add_argument('--init-joint-offsets-deg', type=float, nargs=5, default=None, help='initial zero corrections (deg) for pan, lift, elbow, wrist flex, roll')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/camera_calibration/hand_eye_fit.json')
    args = p.parse_args(argv)
    intr = json.loads(args.intrinsics.read_text())['selected']; K = np.array(intr['K']); dist = np.array(intr['dist'])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    bench = scene_bench_config(args.model); model = mujoco.MjModel.from_xml_path(str(args.model)); data = mujoco.MjData(model); cid = model.camera('wrist_camera').id
    records = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(args.input_dir / '*.json')))]
    if len(records) < 3:
        raise SystemExit(f'need at least 3 extrinsic frames, have {len(records)}')
    board = records[0]['board']; cols, rows = board['inner_corners']; s = board['square_mm'] / 1000.0
    board_pts = np.array([[c * s, r * s, 0.0] for r in range(rows) for c in range(cols)])  # board frame, z up out of the screen
    Rm = quat_to_matrix(model.cam_quat[cid]); pm = model.cam_pos[cid].copy()
    frames = []
    for rec in records:
        corners = np.array(rec['corners_px'], dtype=np.float64)
        if corners.shape[0] != cols * rows:
            continue
        und = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)  # pixel coords in the ideal pinhole with the same K
        qpos = bench.joint_map_object.act_to_mujoco(physical_normalized_to_act(rec['normalized_median']))
        ok, rvec, tvec = cv2.solvePnP(board_pts, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        Rcw, _ = cv2.Rodrigues(rvec); R_cb = Rcw.T @ np.diag([1.0, -1.0, -1.0]); p_cb = (-Rcw.T @ tvec).ravel()  # camera (MuJoCo axes) in the board frame
        frames.append(dict(label=rec['label'], qpos=np.asarray(qpos, float), px=und, R_cb=R_cb, p_cb=p_cb))
    print(f'{len(frames)} frames, {len(frames) * cols * rows} corners')

    def fk(qpos, offsets):
        q = qpos.copy(); q[:5] += offsets
        for name, value in zip(JOINT_NAMES, q):
            data.qpos[model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(model, data)
        return data.body('gripper').xpos.copy(), data.body('gripper').xmat.reshape(3, 3).copy()

    # Parameters: camera pos (3), camera rotvec (3), board pos (3), board rotvec (3), joint offsets (5)
    def board_pose(params):
        if args.free_board:
            return params[6:9], rotvec_to_matrix(params[9:12])
        # flat board: params[6:8] = x, y of the board origin; params[9] = yaw; z fixed at --board-z; normal +z
        return np.array([params[6], params[7], args.board_z]), rotvec_to_matrix(np.array([0.0, 0.0, params[9]]))

    def residuals(params):
        cam_p = params[0:3]; cam_R = rotvec_to_matrix(params[3:6]); b_p, b_R = board_pose(params); off = np.zeros(5) if args.no_joint_offsets else params[12:17]
        world_pts = board_pts @ b_R.T + b_p
        res = []
        for fr in frames:
            gp, gR = fk(fr['qpos'], off); Rw = gR @ cam_R; pw = gp + gR @ cam_p
            pc = (world_pts - pw) @ Rw; z = -pc[:, 2]
            if np.any(z <= 1e-6):
                res.extend([3000.0] * (2 * len(world_pts))); continue
            # MuJoCo camera: x right, y up, -z forward; pixel y grows downward
            px = np.stack([cx + fx * pc[:, 0] / z, cy - fy * pc[:, 1] / z], axis=1)
            res.extend((px - fr['px']).ravel().tolist())
        if not args.no_joint_offsets:
            centre = np.deg2rad(args.init_joint_offsets_deg) if args.init_joint_offsets_deg else np.zeros(5)
            res.extend((20.0 * (off - centre) / np.deg2rad(args.joint_prior_deg)).tolist())  # prior centred on the initial offsets
        if args.mount_prior_mm > 0:
            res.extend((args.mount_prior_weight_px * (cam_p - pm) / (args.mount_prior_mm / 1000.0)).tolist())
        if args.camera_rot_prior_deg > 0:
            rel = matrix_to_rotvec((Rm @ rotvec_to_matrix(np.array([0, 0, np.pi]))).T @ cam_R)
            res.extend((20.0 * rel / np.deg2rad(args.camera_rot_prior_deg)).tolist())
        return np.array(res)

    active = np.ones(17, dtype=bool)
    if args.fix_camera_pos:
        active[0:3] = False
    if not args.free_board:
        active[8] = False; active[10] = False; active[11] = False
    if args.no_joint_offsets:
        active[12:17] = False

    def lm(fn, params, iters=200):
        lam = 1e-2; r = fn(params); cost = float(r @ r); idx = np.flatnonzero(active)
        for _ in range(iters):
            J = np.zeros((len(r), len(idx)))
            for j, k in enumerate(idx):
                d = np.zeros(len(params)); d[k] = 1e-6; J[:, j] = (fn(params + d) - r) / 1e-6
            A = J.T @ J; step = np.zeros(len(params)); step[idx] = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), J.T @ r)
            t = params + step; rt = fn(t); ct = float(rt @ rt)
            if ct < cost:
                params, r, cost, lam = t, rt, ct, max(lam / 3, 1e-8)
                if np.abs(step).max() < 1e-10:
                    break
            else:
                lam *= 4
                if lam > 1e10:
                    break
        return params, cost, r

    def closed_form_init():
        """Park-Martin AX=XB on relative motions: X = camera in gripper. Then the board pose from the frame average."""
        G = []
        for fr in frames:
            gp, gR = fk(fr['qpos'], np.zeros(5)); T = np.eye(4); T[:3, :3] = gR; T[:3, 3] = gp; G.append(T)
        C = []
        for fr in frames:
            T = np.eye(4); T[:3, :3] = fr['R_cb']; T[:3, 3] = fr['p_cb']; C.append(T)
        A_list, B_list = [], []
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                A_list.append(np.linalg.inv(G[i]) @ G[j]); B_list.append(np.linalg.inv(C[i]) @ C[j])
        M = np.zeros((3, 3))
        for A, B in zip(A_list, B_list):
            a = matrix_to_rotvec(A[:3, :3]); b = matrix_to_rotvec(B[:3, :3]); M += np.outer(b, a)
        U, S, Vt = np.linalg.svd(M); Rx = Vt.T @ U.T
        if np.linalg.det(Rx) < 0:
            Vt[-1] *= -1; Rx = Vt.T @ U.T
        rows_, rhs = [], []
        for A, B in zip(A_list, B_list):
            rows_.append(A[:3, :3] - np.eye(3)); rhs.append(Rx @ B[:3, 3] - A[:3, 3])
        tx, *_ = np.linalg.lstsq(np.vstack(rows_), np.concatenate(rhs), rcond=None)
        X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx
        Bs = [G[i] @ X @ np.linalg.inv(C[i]) for i in range(len(frames))]  # board in world per frame
        Bp = np.mean([b[:3, 3] for b in Bs], axis=0); Rsum = sum(b[:3, :3] for b in Bs); U, _, Vt = np.linalg.svd(Rsum); BR = U @ Vt
        spread = float(np.mean([np.linalg.norm(b[:3, 3] - Bp) for b in Bs]))
        return X, Bp, BR, spread, S

    # Initialisation: camera = Menagerie pose (as-is and rolled 180 deg), board = flat at the bench square centre with a few yaws.
    board_centre = np.array([*(args.board_init_xy if args.board_init_xy else bench.square_center_xy), args.board_z]); board_offset = np.array([(cols - 1) * s / 2, (rows - 1) * s / 2, 0.0])
    best = None
    if not args.no_closed_form_init:
        X, Bp, BR, spread, S = closed_form_init()
        print(f'closed-form init: camera-in-gripper pos mm {np.round(X[:3, 3] * 1000).astype(int).tolist()}, board centre spread across frames {spread * 1000:.0f} mm, rotation-axis singular values {np.round(S, 3).tolist()} (need >= 2 non-tiny for a well-posed rotation)')
        yaw0 = float(np.arctan2(BR[1, 0], BR[0, 0]))
        p0 = np.concatenate([X[:3, 3], matrix_to_rotvec(X[:3, :3]), np.array([Bp[0], Bp[1], args.board_z]), (matrix_to_rotvec(BR) if args.free_board else np.array([0.0, 0.0, yaw0])), np.zeros(5)])
        params, cost, r = lm(residuals, p0, iters=150); best = (params, cost, r)
    for roll in (0.0, np.pi):
        cam_R0 = Rm @ rotvec_to_matrix(np.array([0, 0, roll]))
        for yaw in np.deg2rad([0, 90, 180, 270]):
            for flip in (False, True):
                b_R0 = rotvec_to_matrix(np.array([0, 0, yaw])) @ (np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3))
                if not args.free_board and flip:
                    continue
                b_p0 = board_centre - b_R0 @ board_offset
                j0 = np.deg2rad(args.init_joint_offsets_deg) if args.init_joint_offsets_deg else np.zeros(5)
                p0 = np.concatenate([pm, matrix_to_rotvec(cam_R0), b_p0, (matrix_to_rotvec(b_R0) if args.free_board else np.array([0.0, 0.0, yaw])), j0])
                params, cost, r = lm(residuals, p0, iters=120)
                if best is None or cost < best[1]:
                    best = (params, cost, r)
    params, cost, r = lm(residuals, best[0], iters=300)
    n_pts = len(frames) * cols * rows; corner_res = r[:2 * n_pts].reshape(len(frames), -1)
    cam_p = params[0:3]; cam_R = rotvec_to_matrix(params[3:6]); off = np.zeros(5) if args.no_joint_offsets else params[12:17]
    fovy = 2 * np.degrees(np.arctan(1080 / (2 * fy)))
    rel = Rm.T @ cam_R
    report = dict(scene_dependencies_sha256=scene_dependency_hash(args.model), joint_map=bench.joint_map, frames=len(frames), corners=n_pts,
                  rms_px=round(float(np.sqrt(np.mean(corner_res ** 2))), 2),
                  per_frame_rms_px={fr['label']: round(float(np.sqrt(np.mean(cr ** 2))), 1) for fr, cr in zip(frames, corner_res)},
                  camera=dict(pos_m=[round(float(v), 5) for v in cam_p], quat_wxyz=[round(float(v), 6) for v in matrix_to_quat(cam_R)], fovy_deg=round(float(fovy), 2),
                              forward_in_gripper=[round(float(v), 3) for v in -cam_R[:, 2]], offset_from_menagerie_mm=[round(float(v) * 1000, 1) for v in cam_p - pm],
                              rotation_from_menagerie_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))), 1),
                              rolled_180_relative_to_menagerie=bool(rel[0, 0] < 0)),
                  board=dict(pos_m=[round(float(v), 4) for v in board_pose(params)[0]], rotvec=[round(float(v), 4) for v in matrix_to_rotvec(board_pose(params)[1])],
                             centre_m=[round(float(v), 4) for v in (board_pose(params)[0] + board_pose(params)[1] @ board_offset)], flat=not args.free_board),
                  joint_zero_offsets_deg=(None if args.no_joint_offsets else dict(zip(JOINT_NAMES[:5], [round(float(np.degrees(v)), 2) for v in off]))),
                  intrinsics=dict(fx=fx, fy=fy, cx=cx, cy=cy, source=str(args.intrinsics)),
                  mujoco_camera_xml=f'<camera name="wrist_camera" mode="fixed" pos="{cam_p[0]:.7f} {cam_p[1]:.7f} {cam_p[2]:.7f}" quat="{" ".join(f"{v:.7f}" for v in matrix_to_quat(cam_R))}" fovy="{fovy:.2f}" />',
                  note='MuJoCo has no principal-point offset or distortion; physical frames must be undistorted (and ideally re-centred) before the policy sees them.')
    args.output.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
