"""Render the simulated wrist camera at the recorded reset and candidate viewing poses.

Simulation only. Writes full-resolution PNGs (matching the physical 1920x1080
MJPEG frame) plus side-by-side and blended composites against a physical frame,
and prints SHA-256 digests for the camera review record. Never marks anything
as aligned: the comparison is the reviewer's.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(model, data, camera: str, width: int, height: int):
    import mujoco
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()
    finally:
        renderer.close()


def main(argv=None) -> int:
    from PIL import Image
    import mujoco
    from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
    from so_arm101_v2.simulation.contact import GRASP_DETECTOR_VERSION
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    p.add_argument('--physical', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/inspection/physical_prepared_wrist.png',
                   help='physical wrist frame taken at the recorded reset pose')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/inspection')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    args = p.parse_args(argv)
    bench = scene_bench_config(args.model)
    if bench is None or bench.viewing_qpos is None:
        raise RuntimeError('bench scene with a recorded reset and candidate viewing pose required')
    scene_hash = scene_dependency_hash(args.model)
    tag = f'{scene_hash[:8]}'
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    outputs = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, qpos in (('reset', bench.reset_qpos), ('viewing', bench.viewing_qpos)):
        mujoco.mj_resetData(model, data)
        for joint, value in zip(joints, qpos):
            data.qpos[model.joint(joint).qposadr[0]] = float(value)
        mujoco.mj_forward(model, data)
        for camera in ('wrist_camera', 'camera_side'):
            image = render(model, data, camera, args.width, args.height)
            path = args.output_dir / f'sim_{name}_{camera}_{tag}.png'
            Image.fromarray(image).save(path)
            outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        wrist = Image.open(args.output_dir / f'sim_{name}_wrist_camera_{tag}.png').convert('RGB')
        if args.physical.is_file():
            physical = Image.open(args.physical).convert('RGB').resize(wrist.size)
            side = Image.new('RGB', (wrist.width * 2, wrist.height))
            side.paste(physical, (0, 0)); side.paste(wrist, (wrist.width, 0))
            side_path = args.output_dir / f'compare_{name}_physical_left_sim_right_{tag}.png'
            side.save(side_path); outputs[side_path.name] = hashlib.sha256(side_path.read_bytes()).hexdigest()
            blend = Image.blend(physical, wrist, 0.5)
            blend_path = args.output_dir / f'compare_{name}_blend50_{tag}.png'
            blend.save(blend_path); outputs[blend_path.name] = hashlib.sha256(blend_path.read_bytes()).hexdigest()
    record = dict(scene_dependencies_sha256=scene_hash, grasp_detector=GRASP_DETECTOR_VERSION,
                  reset_evidence_sha256=bench.reset_evidence_sha256, physical_frame=str(args.physical),
                  physical_frame_sha256=hashlib.sha256(args.physical.read_bytes()).hexdigest() if args.physical.is_file() else None,
                  note='The physical frame was taken at the recorded RESET pose; no physical frame exists for the candidate viewing pose.',
                  outputs=outputs)
    (args.output_dir / f'render_manifest_{tag}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
