"""Fit the simulated wrist camera (pose in the gripper frame, optional fovy and k1) to physical frames.

Each camera_references record pairs a physical wrist frame with a read-only joint
reading taken at the same instant. The joints give the gripper pose through the
bench joint map; the white square's four corners are detected in the frame. The
corner-to-corner association is NOT fixed up front: for every candidate camera the
objective takes, per frame, the best of the eight cyclic/mirrored orderings. The
camera is found by multi-start Levenberg-Marquardt on a smooth angular residual
followed by pixel refinement. Simulation plus images only; touches no hardware.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fit_bench_camera import Frame, detect_square_corners, frame_pose_with_offsets, matrix_to_quat, quat_to_matrix, rotvec_to_matrix  # noqa: E402

W, H = 1920, 1080


def orderings(c):
    out = []
    for rot in range(4):
        for rev in (False, True):
            seq = c[::-1] if rev else c
            out.append(seq[rot:] + seq[:rot])
    return out


def main(argv=None) -> int:
    import mujoco
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    p.add_argument('--references', type=Path, nargs='+', required=True)
    p.add_argument('--fovy', type=float, default=70.5, help='vertical field of view (deg); fixed unless --fit-fovy')
    p.add_argument('--fit-fovy', action='store_true'); p.add_argument('--fit-k1', action='store_true')
    p.add_argument('--starts', type=int, default=300); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--huber', type=float, default=0.0, help='Huber threshold in px for the pixel stage (0 = plain least squares)')
    p.add_argument('--fingertips', default=None, help='observed pixel positions of the two finger ends, "x1,y1;x2,y2"; both assignments tried per frame')
    p.add_argument('--tip-weight', type=float, default=1.0)
    p.add_argument('--fit-joint-offsets', action='store_true', help='also fit zero corrections (deg) for pan, lift, elbow, wrist flex, roll; Gaussian prior --joint-prior-deg')
    p.add_argument('--joint-prior-deg', type=float, default=5.0)
    p.add_argument('--joint-prior-weight-px', type=float, default=30.0, help='pixel-equivalent penalty when an offset equals the prior width')
    p.add_argument('--init-from', type=Path, default=None, help='previous fit JSON to seed the camera')
    p.add_argument('--mount-prior-mm', type=float, default=0.0, help='Gaussian prior width (mm) tying the camera position to the Menagerie mount camera position (0 = off)')
    p.add_argument('--mount-prior-weight-px', type=float, default=50.0)
    p.add_argument('--menagerie-starts', action='store_true', help='start from the Menagerie camera pose and from it rolled 180 deg about the optical axis')
    p.add_argument('--mount-body', default='gripper', help='model body the camera is assumed rigid to (gripper, Wrist_Pitch_Roll, Lower_Arm, ...)')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/camera_fit.json')
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model); model = mujoco.MjModel.from_xml_path(str(args.model)); data = mujoco.MjData(model)
    frames = [Frame(json.loads(r.read_text()), bench, model, data, mount_body=args.mount_body) for r in args.references]
    frames = [f for f in frames if len(f.corners_px) == 4]
    print(f'{len(frames)} frames with four square corners: ' + ', '.join(f.label for f in frames))
    sq = np.array([*bench.square_center_xy, bench.square_thickness_m]); h = bench.square_edge_m / 2
    world = np.array([sq + np.array([sx * h, sy * h, 0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    frame_orderings = [[np.array(o) for o in orderings(f.corners_px)] for f in frames]

    NP = 13

    def unpack(params):
        return params[:3], rotvec_to_matrix(params[3:6]), (params[6] if args.fit_fovy else args.fovy), (params[7] if args.fit_k1 else 0.0)

    def apply_offsets(params):
        if not args.fit_joint_offsets:
            return
        for fr in frames:
            fr.gripper_pos, fr.gripper_rot, fr.finger_ends = frame_pose_with_offsets(fr, model, data, params[8:13], args.mount_body)

    def prior_residuals(params):
        res = []
        if args.fit_joint_offsets:
            res.extend((args.joint_prior_weight_px * params[8:13] / np.deg2rad(args.joint_prior_deg)).tolist())
        if args.mount_prior_mm > 0:
            res.extend((args.mount_prior_weight_px * (params[:3] - model.cam_pos[model.camera('wrist_camera').id]) / (args.mount_prior_mm / 1000.0)).tolist())
        return res

    def per_frame_pixels(params, fr):
        pos, rot, fovy, k1 = unpack(params); f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
        Rw = fr.gripper_rot @ rot; pw = fr.gripper_pos + fr.gripper_rot @ pos
        pc = (world - pw) @ Rw  # rows: points in camera frame
        z = -pc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = pc[:, 0] / z; yn = pc[:, 1] / z; scale = 1 + k1 * (xn * xn + yn * yn)
            px = np.stack([W / 2 + f * xn * scale, H / 2 - f * yn * scale], axis=1)
        return px, z

    def pixel_residuals(params):
        apply_offsets(params); res = []
        for fr, opts in zip(frames, frame_orderings):
            px, z = per_frame_pixels(params, fr)
            if np.any(z <= 1e-6):
                res.extend([2000.0] * 8); continue
            best = min(opts, key=lambda o: float(np.sum((px - o) ** 2)))
            diff = (px - best).ravel()
            if args.huber > 0:
                a = np.abs(diff); diff = np.where(a <= args.huber, diff, np.sign(diff) * np.sqrt(args.huber * (2 * a - args.huber)))
            res.extend(diff.tolist())
        res.extend(tip_residuals(params)); res.extend(prior_residuals(params))
        return np.array(res)

    def angular_residuals(params):
        apply_offsets(params)
        pos, rot, fovy, _ = unpack(params); f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2); res = []
        for fr, opts in zip(frames, frame_orderings):
            Rw = fr.gripper_rot @ rot; pw = fr.gripper_pos + fr.gripper_rot @ pos
            d = (world - pw) @ Rw; d = d / np.linalg.norm(d, axis=1, keepdims=True)
            best = None
            for o in opts:
                r = np.stack([(o[:, 0] - W / 2) / f, -(o[:, 1] - H / 2) / f, -np.ones(4)], axis=1); r = r / np.linalg.norm(r, axis=1, keepdims=True)
                diff = (d - r).ravel(); c = float(diff @ diff)
                if best is None or c < best[0]:
                    best = (c, diff)
            res.extend(best[1].tolist())
        if tips_px is not None:
            for fr in frames:
                Rw = fr.gripper_rot @ rot; pw = fr.gripper_pos + fr.gripper_rot @ pos
                d = (np.asarray(fr.finger_ends) - pw) @ Rw; d = d / np.linalg.norm(d, axis=1, keepdims=True)
                r = np.stack([(tips_px[:, 0] - W / 2) / f, -(tips_px[:, 1] - H / 2) / f, -np.ones(2)], axis=1); r = r / np.linalg.norm(r, axis=1, keepdims=True)
                a = (d - r).ravel(); b = (d - r[::-1]).ravel(); res.extend((a if a @ a <= b @ b else b).tolist())
        return np.array(res)

    tips_px = None
    if args.fingertips:
        tips_px = np.array([[float(v) for v in pair.split(',')] for pair in args.fingertips.split(';')])

    def project_points(params, fr, pts_w):
        pos, rot, fovy, k1 = unpack(params); f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
        Rw = fr.gripper_rot @ rot; pw = fr.gripper_pos + fr.gripper_rot @ pos
        pc = (np.asarray(pts_w) - pw) @ Rw; z = -pc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = pc[:, 0] / z; yn = pc[:, 1] / z; scale = 1 + k1 * (xn * xn + yn * yn)
            return np.stack([W / 2 + f * xn * scale, H / 2 - f * yn * scale], axis=1), z

    def tip_residuals(params):
        if tips_px is None:
            return []
        res = []
        for fr in frames:
            px, z = project_points(params, fr, fr.finger_ends)
            if np.any(z <= 1e-6):
                res.extend([2000.0] * 4); continue
            a = (px - tips_px).ravel(); b = (px - tips_px[::-1]).ravel()
            res.extend((args.tip_weight * (a if a @ a <= b @ b else b)).tolist())
        return res

    mask = np.array([1, 1, 1, 1, 1, 1, 1 if args.fit_fovy else 0, 1 if args.fit_k1 else 0] + [1 if args.fit_joint_offsets else 0] * 5, dtype=bool)

    def lm(fn, params, iters=150):
        lam = 1e-2; r = fn(params); cost = float(r @ r); idx = np.flatnonzero(mask)
        for _ in range(iters):
            J = np.zeros((len(r), len(idx)))
            for j, k in enumerate(idx):
                dp = np.zeros(NP); dp[k] = 1e-5 if k < 6 else (1e-2 if k == 6 else (1e-4 if k == 7 else 1e-5))
                J[:, j] = (fn(params + dp) - r) / dp[k]
            A = J.T @ J; g = J.T @ r
            step = np.zeros(NP); step[idx] = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            trial = params + step; rt = fn(trial); ct = float(rt @ rt)
            if ct < cost:
                params, r, cost, lam = trial, rt, ct, max(lam / 3, 1e-7)
                if np.abs(step).max() < 1e-9:
                    break
            else:
                lam *= 4
                if lam > 1e9:
                    break
        return params, cost

    rng = np.random.default_rng(args.seed)
    cid = model.camera('wrist_camera').id
    def rotvec_of(R):
        ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)); ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return ax / (np.linalg.norm(ax) + 1e-12) * ang
    starts = [np.concatenate([model.cam_pos[cid], np.zeros(3), [args.fovy], [0.0], np.zeros(5)])]
    if args.menagerie_starts:
        Rm = quat_to_matrix(model.cam_quat[cid]); Rflip = Rm @ np.diag([-1.0, -1.0, 1.0])  # 180 deg about the optical (z) axis
        starts = [np.concatenate([model.cam_pos[cid], rotvec_of(Rm), [args.fovy], [0.0], np.zeros(5)]),
                  np.concatenate([model.cam_pos[cid], rotvec_of(Rflip), [args.fovy], [0.0], np.zeros(5)])]
    if args.init_from is not None:
        prev = json.loads(args.init_from.read_text())['fitted']; Rp = quat_to_matrix(np.array(prev['quat_wxyz']))
        ang = np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1)); ax = np.array([Rp[2, 1] - Rp[1, 2], Rp[0, 2] - Rp[2, 0], Rp[1, 0] - Rp[0, 1]])
        starts.insert(0, np.concatenate([prev['pos_m'], ax / (np.linalg.norm(ax) + 1e-12) * ang, [args.fovy], [0.0], np.zeros(5)]))
    # express Menagerie rotation as a rotvec start? rotvec_to_matrix(0)=I means camera axes = gripper axes; sample rotations instead
    for _ in range(args.starts):
        q = rng.normal(size=4); q /= np.linalg.norm(q); R = quat_to_matrix(q)
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)); axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        rv = axis / (np.linalg.norm(axis) + 1e-12) * angle
        starts.append(np.concatenate([rng.uniform(-0.15, 0.15, 3), rv, [args.fovy], [0.0], np.zeros(5)]))
    best = None
    for i, s in enumerate(starts):
        p1, c1 = lm(angular_residuals, s, iters=60)
        p2, c2 = lm(pixel_residuals, p1, iters=150)
        if best is None or c2 < best[1]:
            best = (p2, c2)
    params, cost = best
    apply_offsets(params)
    pos, rot, fovy, k1 = unpack(params); n_obs = len(frames) * 4
    per_frame = []
    for fr, opts in zip(frames, frame_orderings):
        px, z = per_frame_pixels(params, fr)
        o = min(opts, key=lambda o: float(np.sum((px - o) ** 2)))
        tips_proj, _ = project_points(params, fr, fr.finger_ends)
        per_frame.append(dict(label=fr.label, corner_errors_px=[round(float(v), 1) for v in np.linalg.norm(px - o, axis=1)],
                              projected_finger_ends_px=[[round(float(v), 1) for v in t] for t in tips_proj]))
    quat = matrix_to_quat(rot); men_rot = quat_to_matrix(model.cam_quat[cid]); rel = men_rot.T @ rot
    report = dict(scene_dependencies_sha256=scene_dependency_hash(args.model), joint_map=bench.joint_map, mount_body=args.mount_body, frames=len(frames), observations=n_obs,
                  rms_px=round(float(np.sqrt(cost / (n_obs + (2 * len(frames) if tips_px is not None else 0)))), 2), starts=len(starts), fingertips_px=(tips_px.tolist() if tips_px is not None else None),
                  joint_offsets_deg=(dict(zip(['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'], [round(float(np.degrees(v)), 2) for v in params[8:13]])) if args.fit_joint_offsets else None),
                  fitted=dict(pos_m=[round(float(v), 5) for v in pos], quat_wxyz=[round(float(v), 6) for v in quat], fovy_deg=round(float(fovy), 2), k1=round(float(k1), 4),
                              forward_in_gripper=[round(float(v), 3) for v in -rot[:, 2]]),
                  menagerie=dict(pos_m=[round(float(v), 5) for v in model.cam_pos[cid]], quat_wxyz=[round(float(v), 6) for v in model.cam_quat[cid]], fovy_deg=float(model.cam_fovy[cid]),
                                 forward_in_gripper=[round(float(v), 3) for v in -men_rot[:, 2]]),
                  offset_from_menagerie_mm=[round(float(v) * 1000, 1) for v in (pos - model.cam_pos[cid])],
                  rotation_from_menagerie_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))), 1),
                  mujoco_camera_xml=f'<camera name="wrist_camera" mode="fixed" pos="{pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f}" quat="{quat[0]:.7f} {quat[1]:.7f} {quat[2]:.7f} {quat[3]:.7f}" fovy="{fovy:.2f}" />',
                  per_frame=per_frame)
    args.output.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
