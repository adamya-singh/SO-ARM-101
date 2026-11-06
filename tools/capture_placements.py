"""Capture a screened placement suite with a row-strided frames sidecar, as a data directory for tools/augmentation_run.py (2026-09-12).

Generates ``--count`` placements (suite seed ``--suite-seed``, visibility +
teacher screening as the ladder did), runs the preflight, captures one
teacher episode per placement storing only every ``--frame-row-stride``-th
frame (a 4800-placement capture at stride 30 is ~15 GB instead of ~450 GB),
and writes the pointer files ``augmentation_run.py --ladder-dir`` expects:
``train_suite.path``, ``train_manifest.path``, plus ``heldout_suite.path``,
``heldout_manifest.path`` and ``ladder.json`` copied from ``--heldout-from``
(the ladder directory, so every run is scored on the same ten held-out poses).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash  # noqa: E402
from so_arm101_v2.data._serialization import write_immutable_json  # noqa: E402
from scaling_ladder import DEFAULT_MODEL, atomic_json, get_or_capture, get_or_generate_suite, preflight  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--suite-seed", type=int, required=True, help="placement/appearance draw seed (the ladder used 12; use a new one)")
    p.add_argument("--frame-row-stride", type=int, default=30)
    p.add_argument("--heldout-from", type=Path, default=None, help="directory with heldout_suite.path / heldout_manifest.path / ladder.json")
    p.add_argument("--heldout-seed", type=int, default=None,
                   help="instead of --heldout-from: generate, preflight and capture a held-out suite on this scene here (the ladder used seed 8). "
                        "Needed whenever the scene changed: a held-out capture is the teacher's commands on its own scene")
    p.add_argument("--heldout-count", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--perturb-teacher-mm", type=float, default=None,
                   help="perturbed-teacher capture (2026-11-05): shift the teacher's first descent by an offset uniform in a disk of this "
                        "radius (world xy, mm); it corrects onto the cube in 185-225 and the survey chunk keeps the clean label")
    p.add_argument("--teacher-descend-mm", type=float, default=None,
                   help="with --perturb-teacher-mm: height of the shifted 'descended' waypoint above the grasp point (default 8 mm, the teacher's own)")
    p.add_argument("--heldout-suite", type=Path, default=None,
                   help="re-capture this existing held-out suite.json on --model (e.g. the ladder's 10 poses on the measured-servo scene, "
                        "where the teacher's commands differ); --heldout-from then only supplies ladder.json")
    p.add_argument("--physics", default=None,
                   help="physics randomization regime for the training suite (contracts/physics.py, e.g. bench_physics_v1): every "
                        "candidate gets its own draw and is screened under it; the held-out capture keeps the scene's own physics")
    args = p.parse_args(argv)
    if args.teacher_descend_mm is not None and args.perturb_teacher_mm is None:
        p.error("--teacher-descend-mm needs --perturb-teacher-mm")
    suite_kwargs = {} if args.perturb_teacher_mm is None else dict(
        perturb_teacher_radius_m=args.perturb_teacher_mm / 1000.0,
        teacher_descend_height_m=None if args.teacher_descend_mm is None else args.teacher_descend_mm / 1000.0)
    if args.physics is not None:
        suite_kwargs["physics_regime"] = args.physics
    if args.heldout_suite is not None:
        if args.heldout_seed is not None:
            p.error("--heldout-suite and --heldout-seed are alternatives")
    elif (args.heldout_from is None) == (args.heldout_seed is None):
        p.error("give exactly one of --heldout-from, --heldout-seed and --heldout-suite")
    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    bench = scene_bench_config(args.model)
    if bench is None or bench.placement is None:
        raise RuntimeError("needs a bench scene with a placement regime")
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    identity = dict(scene_dependencies_sha256=scene_dependency_hash(args.model), bench=bench.__dict__, count=args.count, suite_seed=args.suite_seed,
                    frame_row_stride=args.frame_row_stride,
                    **(dict(teacher_perturbation=dict(radius_mm=args.perturb_teacher_mm, descend_mm=args.teacher_descend_mm))
                       if args.perturb_teacher_mm is not None else {}),
                    **(dict(physics=args.physics) if args.physics is not None else {}),
                    **(dict(heldout_suite=str(args.heldout_suite.resolve())) if args.heldout_suite is not None
                       else dict(heldout_from=str(args.heldout_from.resolve())) if args.heldout_from is not None
                       else dict(heldout=dict(seed=args.heldout_seed, count=args.heldout_count))))
    if not (root / "capture_identity.json").exists():
        write_immutable_json(root / "capture_identity.json", identity)
    copied = ("ladder.json",) if args.heldout_suite is not None else ("heldout_suite.path", "heldout_manifest.path", "ladder.json")
    for name in copied:
        if args.heldout_from is not None and not (root / name).exists() and ((args.heldout_from / name).exists() or name != "ladder.json"):
            shutil.copy2(args.heldout_from / name, root / name)     # ladder.json exists only in the ladder's own directory
    state = dict(status="running", started_at=time.time())
    try:
        if args.heldout_suite is not None:
            # The same held-out poses captured on this scene (every frame stored, as the ladder's held-out capture).
            state["phase"] = "heldout"; atomic_json(root / "progress.json", state)
            from so_arm101_v2.simulation.suites import load_suite_from_path
            heldout_suite = load_suite_from_path(args.heldout_suite)
            pointer = root / "heldout_suite.path"
            if not pointer.exists():
                pointer.write_text(str(args.heldout_suite.resolve()) + "\n")
            try:
                gate = preflight(args.model, root, heldout_suite, args.workers)
            except RuntimeError:
                # The teacher may fail a pose on this scene (2026-11-05: the compensated teacher misses held-out pose_000
                # on the measured-servo scene in all three repeats). Evaluation still uses every pose (heldout_suite.path
                # names the full suite); the capture, which only feeds the held-out loss curve, keeps the poses the
                # teacher passes in every repeat, and the dropped ones are recorded.
                import dataclasses
                report = json.loads((root / "simulation" / "preflight" / heldout_suite.suite_id / "evaluation.json").read_text())
                failed = sorted({r["scenario_id"] for r in report["rollouts"] if not r["success"]})
                kept = tuple(s for s in heldout_suite.scenarios if s.scenario_id not in failed)
                if not kept:
                    raise
                heldout_suite = dataclasses.replace(heldout_suite, suite_id=f"{heldout_suite.suite_id}_teacherok", scenarios=kept)
                atomic_json(root / "heldout_capture_dropped.json", dict(dropped=failed, kept=[s.scenario_id for s in kept],
                                                                         reason="teacher fails these poses on this scene"))
                gate = preflight(args.model, root, heldout_suite, args.workers)
            get_or_capture(args.model, root, "heldout", heldout_suite, gate, workers=args.workers)
        elif args.heldout_from is None:
            # Same held-out recipe as the ladder (seed 8, 10 poses, 3 repeats, every frame stored), on this scene.
            state["phase"] = "heldout"; atomic_json(root / "progress.json", state)
            heldout_suite, _ = get_or_generate_suite(args.model, bench, root, "heldout", seed=args.heldout_seed, count=args.heldout_count, repeats=3, workers=args.workers)
            get_or_capture(args.model, root, "heldout", heldout_suite, preflight(args.model, root, heldout_suite, args.workers), workers=args.workers)
        state["phase"] = "suite"; atomic_json(root / "progress.json", state)
        suite, _ = get_or_generate_suite(args.model, bench, root, "train", seed=args.suite_seed, count=args.count, repeats=1, workers=args.workers,
                                         **suite_kwargs)
        state["phase"] = "preflight"; atomic_json(root / "progress.json", state)
        pre = preflight(args.model, root, suite, args.workers)
        state["phase"] = "capture"; atomic_json(root / "progress.json", state)
        manifest = get_or_capture(args.model, root, "train", suite, pre, workers=args.workers, frame_row_stride=args.frame_row_stride)
        body = json.loads(Path(manifest).read_text())
        state.update(status="complete", phase="complete", manifest=str(manifest), episodes=len(body["episodes"]), frame_rows=body["frames"]["rows"],
                     seconds=round(time.time() - state["started_at"]))
        atomic_json(root / "progress.json", state)
        print(f"CAPTURE {len(body['episodes'])} episodes, {body['frames']['rows']} frame rows, {state['seconds']} s -> {manifest}", flush=True)
        return 0
    except BaseException as exc:
        import traceback
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc()); atomic_json(root / "progress.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
