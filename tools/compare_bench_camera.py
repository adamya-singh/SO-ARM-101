"""Quantify the sim-vs-physical wrist camera mismatch at the recorded reset pose (simulation + one image, no hardware).

Detects the white square in the physical wrist frame, projects the sim square
and cube through the sim wrist camera at the same recorded pose under the bench
config's joint map, writes an overlay (red = physical square, green = sim
square, cyan = sim cube top) and prints the pixel-space comparison. A reviewer
decides what it means; this tool asserts nothing about alignment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    import cv2
    import mujoco
    from PIL import Image, ImageDraw
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    from so_arm101_v2.contracts.coordinates import JOINT_NAMES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    p.add_argument('--physical', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/physical_prepared_wrist.png')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/inspection')
    p.add_argument('--reference', type=Path, default=None,
                   help='camera_references JSON (frame + joints at capture); overrides --physical and the reset pose')
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model); tag = scene_dependency_hash(args.model)[:8]
    qpos = bench.reset_qpos; pose_label = 'recorded reset'
    if args.reference is not None:
        from so_arm101_v2.contracts.physical import physical_normalized_to_act
        record = json.loads(args.reference.read_text())
        qpos = bench.joint_map_object.act_to_mujoco(physical_normalized_to_act(record['normalized_median']))
        args.physical = ROOT / record['frame']; pose_label = record['label']; tag = f"{tag}_{record['label']}"
    m = mujoco.MjModel.from_xml_path(str(args.model)); d = mujoco.MjData(m)
    for name, value in zip(JOINT_NAMES, qpos):
        d.qpos[m.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(m, d)
    cid = m.camera('wrist_camera').id
    W, H = 1920, 1080; f = 0.5 * H / np.tan(np.deg2rad(float(m.cam_fovy[cid])) / 2)
    cpos = d.cam_xpos[cid].copy(); R = d.cam_xmat[cid].reshape(3, 3)

    def project(point):
        pc = R.T @ (np.asarray(point, float) - cpos)
        if pc[2] >= 0:
            return None
        return (W / 2 + f * pc[0] / -pc[2], H / 2 - f * pc[1] / -pc[2])

    square = np.array([*bench.square_center_xy, bench.square_thickness_m]); h = bench.square_edge_m / 2
    sim_square = [project(square + np.array([sx * h, sy * h, 0])) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    cube_top = np.array([*bench.square_center_xy, bench.square_thickness_m + bench.cube_edge_m])
    sim_cube = [project(cube_top + np.array([sx * bench.cube_edge_m / 2, sy * bench.cube_edge_m / 2, 0])) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    report = dict(scene_dependencies_sha256=scene_dependency_hash(args.model), joint_map=bench.joint_map, pose=pose_label,
                  pose_model_degrees=np.degrees(np.asarray(qpos, float)).round(1).tolist(), physical_frame=str(args.physical),
                  sim_camera=dict(fovy_deg=float(m.cam_fovy[cid]), position=cpos.round(4).tolist(), forward=(-R[:, 2]).round(3).tolist(),
                                  distance_to_square_center_m=round(float(np.linalg.norm(square - cpos)), 4)),
                  sim_square_px=[None if q is None else [round(v, 1) for v in q] for q in sim_square],
                  sim_cube_top_px=[None if q is None else [round(v, 1) for v in q] for q in sim_cube])
    visible = [q for q in sim_square if q is not None]
    if len(visible) == 4:
        xs = [q[0] for q in visible]; ys = [q[1] for q in visible]
        report['sim_square_bbox_px'] = dict(width=round(max(xs) - min(xs), 1), height=round(max(ys) - min(ys), 1), center=[round(np.mean(xs), 1), round(np.mean(ys), 1)])
    img = cv2.imread(str(args.physical)); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(white, connectivity=8)
    candidates = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if stats[i, cv2.CC_STAT_LEFT] > 5 and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] < W - 5]
    overlay = Image.open(args.physical).convert('RGB'); draw = ImageDraw.Draw(overlay)
    if candidates:
        area, i = max(candidates)
        x, y, w, hh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        report['physical_square_bbox_px'] = dict(width=int(w), height=int(hh), center=[round(float(cent[i][0]), 1), round(float(cent[i][1]), 1)], touches_top=bool(y == 0))
        mask = (lab == i).astype(np.uint8); contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quad = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.02 * cv2.arcLength(max(contours, key=cv2.contourArea), True), True).reshape(-1, 2)
        draw.polygon([tuple(map(float, q)) for q in quad], outline=(255, 0, 0), width=4)
        if 'sim_square_bbox_px' in report:
            report['apparent_width_ratio_physical_over_sim'] = round(w / report['sim_square_bbox_px']['width'], 2)
    if len(visible) == 4:
        draw.polygon([tuple(q) for q in visible], outline=(0, 255, 0), width=4)
    if all(q is not None for q in sim_cube):
        draw.polygon([tuple(q) for q in sim_cube], outline=(0, 200, 255), width=3)
    draw.text((20, 20), f"red: physical white square | green: sim square projected at pose '{pose_label}' ({bench.joint_map}) | cyan: sim cube top", fill=(255, 255, 0))
    out = args.output_dir / f'compare_reset_sim_projection_over_physical_{tag}.png'; overlay.save(out); report['overlay'] = str(out)
    (args.output_dir / f'camera_comparison_{tag}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
