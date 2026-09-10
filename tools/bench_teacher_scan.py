"""Simulation-only teacher parameter scan on the active bench scene.

Runs the privileged teacher for every combination of grasp pad, approach
pitch, pocket height offset and depth lead, on the nominal pose or on the five
certification poses, and prints one JSON line per configuration: passes,
strict-grasp frames, safety events, corner rejections and the minimum jaw-axis
cosine. Teacher tuning only; it promotes nothing and writes no artifacts.

Example:
  PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYTHONPATH=src python tools/bench_teacher_scan.py \
      --pads 1 --pitches 10,20,30 --offsets 0,0.004 --leads 0 --poses certification
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.bench import bench_suite
from so_arm101_v2.simulation.contact import check_block_face_gripped
from so_arm101_v2.simulation.privileged import PrivilegedStagedController

ROOT = Path(__file__).resolve().parents[1]
from so_arm101_v2.simulation.bench import CERTIFICATION_OFFSETS  # noqa: E402


def run_episode(model, bench, pose, lead, offset):
    adapter = MujocoTaskAdapter(model)
    adapter.bench = bench
    adapter.joint_map = bench.joint_map_object
    try:
        adapter.reset(bench_suite(bench, [pose], label='scan', repeats=1).scenarios[0])
        controller = PrivilegedStagedController()
        controller.reset(adapter)
        contract = load_pick_place_contract('bench_pick_replace_v1', bench_config=bench)
        state = PickPlaceEvaluationState()
        strict = corner = safety = 0; min_jaw = 1.0; z0 = None; lift = 0.0
        for step in range(contract.max_actions):
            command = adapter.apply_policy_command(controller.predict(None, adapter.current_act(), adapter))
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(command)
            state, result = evaluate_pick_place_step(contract, measurement, state)
            z = float(adapter.data.body('red_block').xpos[2]); z0 = z if z0 is None else z0; lift = max(lift, z - z0)
            pickup = measurement.pickup
            strict += pickup.strict_bilateral_grasp
            safety += pickup.unsafe_contact + pickup.command_bound_violation + pickup.delta_limiter_activated + pickup.nonfinite_command
            _, _, diagnostics = check_block_face_gripped(adapter.model, adapter.data)
            if diagnostics['bilateral_interior_face_contact']:
                min_jaw = min(min_jaw, diagnostics['face_jaw_axis_alignment'])
            corner += diagnostics['face_corner_rejection_count']
            if result.terminated or result.truncated or result.invalidated:
                break
        return dict(pose=pose, ok=bool(result.success and not result.invalidated), strict=strict, lift_mm=round(lift * 1000, 1),
                    safety=safety, corner=corner, min_jaw=round(min_jaw, 3), steps=step + 1)
    except (RuntimeError, ValueError) as exc:
        return dict(pose=pose, ok=False, error=str(exc)[:100])
    finally:
        adapter.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    p.add_argument('--pads', default='1'); p.add_argument('--pitches', default='78'); p.add_argument('--offsets', default='0'); p.add_argument('--leads', default='0')
    p.add_argument('--poses', choices=['nominal', 'certification'], default='nominal')
    args = p.parse_args(argv)
    poses = CERTIFICATION_OFFSETS if args.poses == 'certification' else [(0, 0)]
    base = MujocoTaskAdapter(args.model).bench
    grid = itertools.product([int(v) for v in args.pads.split(',')], [float(v) for v in args.pitches.split(',')],
                             [float(v) for v in args.offsets.split(',')], [float(v) for v in args.leads.split(',')])
    for pad, pitch, offset, lead in grid:
        started = time.time()
        bench = replace(base, grasp_pad=pad, approach_pitch_deg=pitch, grasp_offset_m=offset, depth_lead_m=lead)
        per = [run_episode(args.model, bench, pose, lead, offset) for pose in poses]
        summary = dict(pad=pad, pitch=pitch, offset_mm=offset * 1000, lead_mm=lead * 1000, passes=f"{sum(r['ok'] for r in per)}/{len(per)}",
                       min_strict=min(r.get('strict', 0) for r in per), safety=sum(r.get('safety', 0) for r in per),
                       corner=sum(r.get('corner', 0) for r in per), min_jaw=min(r.get('min_jaw', 0) for r in per),
                       lift_mm=min(r.get('lift_mm', 0) for r in per), fails=[r for r in per if not r['ok']], sec=round(time.time() - started, 1))
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
