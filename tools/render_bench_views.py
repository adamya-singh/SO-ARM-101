"""Render the simulated wrist camera at the recorded reset and candidate viewing poses.

Simulation only. Writes full-resolution PNGs (matching the physical 1920x1080
MJPEG frame) plus side-by-side and blended composites against a physical frame,
and prints SHA-256 digests for the camera review record. Never marks anything
as aligned: the comparison is the reviewer's.

With a lens block in the bench config the review material is the *observation
pair*: the sim's 256x256 observation (wide pinhole render resampled through the
calibrated lens) next to the physical frame passed through the same area filter
(what a deployed policy would see), plus a synthetic 1920x1080 "raw" frame made
from the render with the lens applied, next to the real raw frame.
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


def synthetic_raw_frame(lens, render):
    """Full-resolution raw-camera frame synthesised from the wide pinhole render through the lens (review only)."""
    from so_arm101_v2.contracts.lens import _bilinear_entries, _coalesce, Resampler
    w, h = lens.image_size
    v, u = np.mgrid[0:h, 0:w]
    xd, yd = lens.raw_to_normalised(u.ravel().astype(np.float64), v.ravel().astype(np.float64))
    x, y = lens.undistort(xd, yd)
    ru, rv = lens.render_pixel(x, y)
    rows = np.arange(w * h)
    r, c, wt = _bilinear_entries(rows, ru, rv, lens.render_size[0], lens.render_size[1], "synthetic raw frame")
    row_ptr, col, weight = _coalesce(r, c, wt, w * h)
    op = Resampler((lens.render_size[1], lens.render_size[0]), (h, w), row_ptr, col, weight)
    return op.apply(render)


def render_appearance_sheet(model_path, bench, output_dir, tag, count, *, physical_observation=None):
    """Review sheet: the real reset observation (if present), the pristine sim observation, then ``count`` appearance draws."""
    from dataclasses import replace
    from PIL import Image
    from so_arm101_v2.contracts.appearance import resolve_appearance
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    if bench.appearance_regime is None:
        raise RuntimeError('--appearance-samples needs an appearance regime in bench_config.json')
    adapter = MujocoTaskAdapter(model_path)
    tiles, seeds = [], []
    try:
        nominal = bench_suite(adapter.bench, [(0, 0)], label='review', repeats=1).scenarios[0]
        if physical_observation is not None and Path(physical_observation).is_file():
            tiles.append(np.asarray(Image.open(physical_observation).convert('RGB')))
        adapter.reset(nominal)
        tiles.append(adapter.render_wrist_observation())
        for seed in range(count):
            adapter.reset(replace(nominal, scenario_id=f'review_a{seed}', fixed_appearance=False, appearance_seed=seed))
            tiles.append(adapter.render_wrist_observation()); seeds.append(seed)
    finally:
        adapter.close()
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * 256, rows * 256))
    for index, tile in enumerate(tiles):
        sheet.paste(Image.fromarray(tile), ((index % columns) * 256, (index // columns) * 256))
    path = Path(output_dir) / f'sim_reset_appearance_samples_{tag}.png'
    sheet.save(path)
    return path, dict(seeds=seeds, order='real reset observation (if present), pristine sim, then draws in seed order, row-major',
                      params={seed: resolve_appearance(bench.appearance_regime, seed).as_record() for seed in seeds})


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
    p.add_argument('--appearance-samples', type=int, default=0,
                   help='also render this many random appearance draws of the reset observation (needs an appearance regime in the config) '
                        'into a review sheet next to the real reset frame')
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
        lens = bench.lens_model
        for camera in ('wrist_camera', 'camera_side'):
            if camera == 'wrist_camera' and lens is not None:
                # Wide pinhole render -> synthetic raw frame through the lens (full resolution) and the
                # 256x256 observation the policy trains on.
                wide = render(model, data, camera, *lens.render_size)
                image = synthetic_raw_frame(lens, wide)
                obs = lens.sim_operator().apply(wide)
                obs_path = args.output_dir / f'sim_{name}_observation_{tag}.png'
                Image.fromarray(obs).save(obs_path); outputs[obs_path.name] = hashlib.sha256(obs_path.read_bytes()).hexdigest()
                wide_path = args.output_dir / f'sim_{name}_wide_render_{tag}.png'
                Image.fromarray(wide).save(wide_path); outputs[wide_path.name] = hashlib.sha256(wide_path.read_bytes()).hexdigest()
            else:
                image = render(model, data, camera, args.width, args.height)
            path = args.output_dir / f'sim_{name}_{camera}_{tag}.png'
            Image.fromarray(image).save(path)
            outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        wrist = Image.open(args.output_dir / f'sim_{name}_wrist_camera_{tag}.png').convert('RGB')
        if args.physical.is_file() and lens is not None:
            raw = np.asarray(Image.open(args.physical).convert('RGB'))
            if raw.shape[:2] == (lens.image_size[1], lens.image_size[0]):
                real_obs = lens.real_operator().apply(np.ascontiguousarray(raw))
                sim_obs = np.asarray(Image.open(args.output_dir / f'sim_{name}_observation_{tag}.png').convert('RGB'))
                pair = Image.new('RGB', (256 * 4 + 8, 256 * 2))
                pair.paste(Image.fromarray(real_obs).resize((512, 512), Image.NEAREST), (0, 0))
                pair.paste(Image.fromarray(sim_obs).resize((512, 512), Image.NEAREST), (512 + 8, 0))
                pair_path = args.output_dir / f'compare_{name}_observation_physical_left_sim_right_{tag}.png'
                pair.save(pair_path); outputs[pair_path.name] = hashlib.sha256(pair_path.read_bytes()).hexdigest()
                real_obs_path = args.output_dir / f'physical_{name}_observation_{tag}.png'
                Image.fromarray(real_obs).save(real_obs_path); outputs[real_obs_path.name] = hashlib.sha256(real_obs_path.read_bytes()).hexdigest()
        if args.physical.is_file():
            physical = Image.open(args.physical).convert('RGB').resize(wrist.size)
            side = Image.new('RGB', (wrist.width * 2, wrist.height))
            side.paste(physical, (0, 0)); side.paste(wrist, (wrist.width, 0))
            side_path = args.output_dir / f'compare_{name}_physical_left_sim_right_{tag}.png'
            side.save(side_path); outputs[side_path.name] = hashlib.sha256(side_path.read_bytes()).hexdigest()
            blend = Image.blend(physical, wrist, 0.5)
            blend_path = args.output_dir / f'compare_{name}_blend50_{tag}.png'
            blend.save(blend_path); outputs[blend_path.name] = hashlib.sha256(blend_path.read_bytes()).hexdigest()
    if args.appearance_samples > 0:
        sheet_path, sheet_record = render_appearance_sheet(args.model, bench, args.output_dir, tag, args.appearance_samples,
                                                           physical_observation=args.output_dir / f'physical_reset_observation_{tag}.png')
        outputs[sheet_path.name] = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
    record = dict(scene_dependencies_sha256=scene_hash, grasp_detector=GRASP_DETECTOR_VERSION, lens=bench.lens,
                  reset_evidence_sha256=bench.reset_evidence_sha256, physical_frame=str(args.physical),
                  physical_frame_sha256=hashlib.sha256(args.physical.read_bytes()).hexdigest() if args.physical.is_file() else None,
                  note='The physical frame was taken at the recorded RESET pose; no physical frame exists for the candidate viewing pose.',
                  outputs=outputs)
    (args.output_dir / f'render_manifest_{tag}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
