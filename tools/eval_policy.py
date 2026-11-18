"""Evaluate one checkpoint on one (scene, physics) cell of the held-out suite (2026-11-05, handoff step 2).

The repo's training tools evaluate only at the end of a run on the scene they trained on. This runs any finished
ACT/SmolVLA/pi0.5 residual checkpoint (``--kind vla``) or CNN (``--kind vision_chunked``) on a chosen scene, optionally
under a physics regime (contracts/physics.py). Under a regime every rollout gets its own draw, so the suite is
expanded to one scenario per (pose, repeat), seeds from a stream no training suite uses. Writes ``result.json``
(successes, failure categories, closure and correction in mm) under ``--output-dir``.

  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot/bin/python tools/eval_policy.py --kind vla \\
      --checkpoint <act.pt> --scene <scene.xml> --physics real_like_v1 --output-dir <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DEFAULT_SUITE = EXPERIMENTS / "scaling_ladder_20260910/heldout_suite.path"
EVAL_PHYSICS_SEED, EVAL_PHYSICS_STREAM = 8, 7    # held-out physics draws: a stream no training suite or RL pool uses


def physics_suite(suite, regime: str):
    """One scenario per (pose, repeat), each with its own draw of ``regime``; repeats 1."""
    from so_arm101_v2.contracts.physics import physics_regime, physics_seeds
    physics_regime(regime)
    seeds = physics_seeds(EVAL_PHYSICS_SEED, len(suite.scenarios) * suite.repeats, stream=EVAL_PHYSICS_STREAM)
    scenarios = tuple(dataclasses.replace(s, scenario_id=f"{s.scenario_id}_r{r}", physics_regime=regime, physics_seed=seeds[i * suite.repeats + r])
                      for i, s in enumerate(suite.scenarios) for r in range(suite.repeats))
    return dataclasses.replace(suite, suite_id=f"{suite.suite_id}_{regime}", scenarios=scenarios, repeats=1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=("vla", "vision_chunked"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE, help="suite.json, or a *.path pointer file naming one")
    p.add_argument("--physics", default="none", help="none, or a regime name from contracts/physics.py (bench_physics_v1, real_like_v1)")
    p.add_argument("--execute", type=int, default=90)
    p.add_argument("--workers", type=int, default=8, help="vision_chunked only (the VLA path evaluates in one process on the GPU)")
    p.add_argument("--black-image", action="store_true",
                   help="pixel-reliance ablation: every camera frame the policy sees is replaced with zeros")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    from so_arm101_v2.contracts.bench import scene_dependency_hash
    from so_arm101_v2.simulation.closure import closure_summary, correction_summary
    from so_arm101_v2.simulation.suites import load_suite_from_path
    suite_path = Path(args.suite.read_text().strip()) if args.suite.suffix == ".path" else args.suite
    suite = load_suite_from_path(suite_path)
    if args.physics != "none":
        suite = physics_suite(suite, args.physics)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(tool="eval_policy", kind=args.kind, checkpoint=str(args.checkpoint.resolve()), scene=str(args.scene.resolve()),
                    scene_dependencies_sha256=scene_dependency_hash(args.scene), suite=suite.suite_id, source_suite=str(suite_path),
                    physics=args.physics, execute=args.execute, black_image=bool(args.black_image))
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    started = time.time()
    if args.kind == "vla":
        from so_arm101_v2.learning.vla_models import load_model
        from vla_finetune import closed_loop
        model = load_model(args.checkpoint)
        if args.black_image:
            import numpy as np
            sample = model.sample
            model.sample = lambda frames, state, **kw: sample(np.zeros_like(frames), state, **kw)
        result = closed_loop(args.scene, suite, model, args.execute, root / "evaluation")
        result.pop("closure_rows", None)
    else:
        from so_arm101_v2.simulation.policy_specs import PolicySpec
        from so_arm101_v2.simulation.rollout import evaluate_closed_loop
        options = (("black_image", bool(args.black_image)), ("clamp_channels", (5,)))
        if args.execute != 90:
            options = (("black_image", bool(args.black_image)), ("clamp_channels", (5,)), ("execute", int(args.execute)))
        spec = PolicySpec(kind="vision_chunked", checkpoint=str(args.checkpoint), options=options)
        report_path = evaluate_closed_loop(args.scene, suite, {"vision": spec}, root / "evaluation", environment_proven=True,
                                           record_video=False, workers=args.workers).report_json
        rows = [r for r in json.loads(Path(report_path).read_text())["rollouts"] if r["policy_id"] == "vision"]
        telemetry = [r["telemetry_path"] for r in rows if r.get("telemetry_path")]
        categories: dict[str, int] = {}
        for r in rows:
            categories[r["failure_category"]] = categories.get(r["failure_category"], 0) + 1
        result = dict(execute=args.execute, successes=int(sum(bool(r["success"] and not r["invalidated"]) for r in rows)), rollouts=len(rows),
                      pickups=int(sum(bool(r["pickup_completed"]) for r in rows)), categories=categories,
                      safety_frames=int(sum(sum(r[k] for k in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")) for r in rows)),
                      closure={k: v for k, v in closure_summary(args.scene, telemetry).items() if k != "rows"},
                      correction={k: v for k, v in correction_summary(args.scene, telemetry).items() if k != "rows"},
                      report=str(report_path))
    result.update(identity=identity, seconds=round(time.time() - started, 1))
    (root / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"EVAL {args.kind} {args.checkpoint.name} scene={args.scene.parent.name} physics={args.physics}: "
          f"{result['successes']}/{result['rollouts']} successes, categories {result.get('categories')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
