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
    p.add_argument("--heldout-from", type=Path, required=True, help="directory with heldout_suite.path / heldout_manifest.path / ladder.json")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    bench = scene_bench_config(args.model)
    if bench is None or bench.placement is None:
        raise RuntimeError("needs a bench scene with a placement regime")
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    identity = dict(scene_dependencies_sha256=scene_dependency_hash(args.model), bench=bench.__dict__, count=args.count, suite_seed=args.suite_seed,
                    frame_row_stride=args.frame_row_stride, heldout_from=str(args.heldout_from.resolve()))
    if not (root / "capture_identity.json").exists():
        write_immutable_json(root / "capture_identity.json", identity)
    for name in ("heldout_suite.path", "heldout_manifest.path", "ladder.json"):
        if not (root / name).exists():
            shutil.copy2(args.heldout_from / name, root / name)
    state = dict(status="running", started_at=time.time())
    try:
        state["phase"] = "suite"; atomic_json(root / "progress.json", state)
        suite, _ = get_or_generate_suite(args.model, bench, root, "train", seed=args.suite_seed, count=args.count, repeats=1, workers=args.workers)
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
