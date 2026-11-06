"""Does the second look correct? Lateral error at step 179 vs 250 for any set of evaluations (2026-11-05).

Each ``--eval`` is ``label=path`` where path is an evaluation directory (searched for ``telemetry/*.json``). Prints
one row per evaluation and, with ``--pool label=prefix``, pooled rows over every evaluation whose label starts with
the prefix (the perturbed-teacher tranche judges pooled seeds). Pre-registered targets: step-250 lateral rms under
4 mm and below step 179; not-descended 0. Same definition as tranche_analysis_20260918 (closure.correction_summary).

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/correction_report.py \
      --eval vision_s202=<dir> --eval vision_s101=<dir> --pool vision=vision_ --out <correction.json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
TARGET_RMS_250_MM = 4.0


def _pair(text: str) -> tuple[str, str]:
    label, _, value = text.partition("=")
    if not label or not value:
        raise argparse.ArgumentTypeError(f"expected label=value, got {text!r}")
    return label, value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--eval", type=_pair, action="append", required=True, metavar="LABEL=DIR")
    p.add_argument("--pool", type=_pair, action="append", default=[], metavar="LABEL=PREFIX")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    from so_arm101_v2.simulation.closure import correction_summary
    paths = {}
    for label, directory in args.eval:
        found = sorted(glob.glob(str(Path(directory) / "**" / "telemetry" / "*.json"), recursive=True))
        if not found:
            print(f"{label}: no telemetry under {directory} (not finished?)", flush=True)
        paths[label] = found
    report = {"target_rms_250_mm": TARGET_RMS_250_MM, "evaluations": {}, "pooled": {}}
    def row(label, summary):
        if not summary.get("count"):
            return f"{label:32s}  no rollouts"
        ok = summary["lateral_rms_mm_250"] < TARGET_RMS_250_MM and summary["correction_mm"] > 0 and summary["not_descended"] == 0
        return (f"{label:32s}  n={summary['count']:3d}  rms179 {summary['lateral_rms_mm_179']:5.1f}  rms250 {summary['lateral_rms_mm_250']:5.1f}  "
                f"corrected {summary['correction_mm']:+5.1f} mm  not_descended {summary['not_descended']:2d}  {'PASS' if ok else 'fail'}")
    for label, found in paths.items():
        summary = correction_summary(args.model, found) if found else {"count": 0}
        report["evaluations"][label] = {**{k: v for k, v in summary.items() if k != "rows"}, "telemetry": len(found)}
        print(row(label, summary), flush=True)
    for label, prefix in args.pool:
        found = [x for name, items in paths.items() if name.startswith(prefix) for x in items]
        summary = correction_summary(args.model, found) if found else {"count": 0}
        report["pooled"][label] = {**{k: v for k, v in summary.items() if k != "rows"}, "members": [n for n in paths if n.startswith(prefix)]}
        print(row(f"POOLED {label}", summary), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
