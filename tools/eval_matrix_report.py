"""Collect tools/eval_policy.py results into a policy x cell table (2026-11-05, physics-randomization tranche).

Reads ``<evals>/<policy>__<cell>/result.json``, prints successes per cell (CNN seeds also pooled per family), the
step-250 lateral closure and the not-descended count, and writes ``--out``. Pre-registered judgement: a policy passes
if it beats its baseline in the servo and servo+real_like_v1 cells without losing more than 3/30 on the nominal scene.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/eval_matrix_report.py --evals <dir> --out <matrix.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

CELLS = ("nominal", "servo", "servo_real_like_v1", "servo_bench_physics_v1")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    table: dict[str, dict[str, dict]] = {}
    for result in sorted(args.evals.glob("*__*/result.json")):
        policy, cell = result.parent.name.split("__", 1)
        r = json.loads(result.read_text())
        closure, correction = r.get("closure") or {}, r.get("correction") or {}
        table.setdefault(policy, {})[cell] = dict(successes=r["successes"], rollouts=r["rollouts"], categories=r.get("categories"),
                                                 lateral_rms_250_mm=correction.get("lateral_rms_mm_250", closure.get("lateral_rms_mm")),
                                                 not_descended=closure.get("not_descended"), safety_frames=r.get("safety_frames"))
    pooled: dict[str, dict[str, dict]] = {}
    for policy, cells in table.items():
        family = re.sub(r"_s\d+$", "", policy)
        if family == policy:
            continue
        for cell, v in cells.items():
            agg = pooled.setdefault(family, {}).setdefault(cell, dict(successes=0, rollouts=0, members=0))
            agg["successes"] += v["successes"]; agg["rollouts"] += v["rollouts"]; agg["members"] += 1
    def fmt(v):
        return "      -" if v is None else f"{v['successes']:3d}/{v['rollouts']:<3d}"
    print(f"{'policy':28s} " + " ".join(f"{c[:20]:>20s}" for c in CELLS))
    for name, cells in list(table.items()) + [(f"POOLED {k}", v) for k, v in pooled.items()]:
        print(f"{name:28s} " + " ".join(f"{fmt(cells.get(c)):>20s}" for c in CELLS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(cells=CELLS, policies=table, pooled=pooled), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
