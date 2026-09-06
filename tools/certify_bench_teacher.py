"""Certify the bench teacher: nominal plus four +-10 mm axis offsets, three repeats each.

Simulation only; never touches hardware. Runs the privileged staged controller
through the existing preflight machinery on the active bench scene and writes
an immutable evaluation under artifacts/. The gate passes only when every
rollout completes the full lift-and-replace contract deterministically with
zero safety invalidation (`environment_proven` and `deterministic` both true).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash
from so_arm101_v2.simulation.bench import bench_suite
from so_arm101_v2.simulation.contact import GRASP_DETECTOR_VERSION
from so_arm101_v2.simulation.rollout import run_simulation_preflight

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATION_OFFSETS = [(0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'artifacts/so_arm101_v2/bench_pick_replace_v1/teacher_certification')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    import mujoco
    if mujoco.__version__ != '3.9.0':
        raise RuntimeError('bench certification requires MuJoCo 3.9.0')
    bench = scene_bench_config(args.model)
    if bench is None:
        raise RuntimeError('not a bench scene')
    bench.reset_qpos  # raises when the reset has not been measured
    scene_hash = scene_dependency_hash(args.model)
    suite = bench_suite(bench, CERTIFICATION_OFFSETS, label='certification', repeats=3)
    # Keyed by scene AND detector version: per-step telemetry carries detector
    # measurements, so a detector change must land in a fresh immutable tree.
    destination = args.output_dir / f'{scene_hash[:16]}-{GRASP_DETECTOR_VERSION}'
    result = run_simulation_preflight(args.model, destination, suite=suite,
                                      record_video=args.record_video, workers=args.workers)
    report = json.loads(Path(result.report_json).read_text())
    rollouts = report['rollouts']
    passed = bool(report.get('environment_proven')) and bool(report.get('deterministic'))
    summary = dict(
        scene_dependencies_sha256=scene_hash, grasp_detector=GRASP_DETECTOR_VERSION,
        suite_id=suite.suite_id, report=str(result.report_json),
        rollouts=len(rollouts), successes=sum(1 for r in rollouts if r.get('success') and not r.get('invalidated')),
        invalidated=sum(1 for r in rollouts if r.get('invalidated')),
        environment_proven=bool(report.get('environment_proven')), deterministic=bool(report.get('deterministic')),
        teacher=dict(grasp_pad=bench.grasp_pad, approach_pitch_deg=bench.approach_pitch_deg,
                     depth_lead_m=bench.depth_lead_m, grasp_offset_m=bench.grasp_offset_m),
        gate='passed' if passed else 'failed',
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
