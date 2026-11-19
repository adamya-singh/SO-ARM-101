#!/usr/bin/env python
"""Read the recursive-generalization tranche (2026-11-19, notes/recursive-generalization-20261119.md).

Three tables from the probe cells (tools/feature_probes.py) of the tranche and of the feature probes it builds on:

  1 stacking     does a policy trained on both data changes show both features?
  2 combination  cube shift and elbow change in one episode: follow gain (against the elbow-change episode) and pickups
                 kept, next to what two independent features would keep (the product of the two single retentions)
  3 order        the target feature against the amount of new data, from a start that has the other feature and from
                 one that has neither (N = 0 is the start checkpoint itself)

  tools/recursive_generalization_report.py [--root DIR] [--probe-root DIR] [--predictions FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from feature_probes import verdict  # noqa: E402

EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
STACK = (("neither", "base_act"), ("look only", "pert_act"), ("motor only", "physv2_act"), ("both, 3600 + 3600", "both_split"), ("both, 7200 + 7143", "both_full"))
TARGET = dict(motor="swap_elbow", look="shift_placement")       # curriculum direction -> the probe of the feature being added
OTHER = dict(motor="shift_placement", look="swap_elbow")
START_CELL = dict(base="base_act", pert="pert_act", physv2="physv2_act")
HAS_OTHER = dict(motor="pert", look="physv2")                  # the start that already carries the other feature
CURRICULUM = re.compile(r"^cur_(motor|look)_from_([a-z0-9]+)_n(\d+)_s(\d+)$")
ORDER_MARGIN = 0.10      # registered: the start with the other feature must lead by this much, mean over N > 0 and seeds


def value(probe: str, m: dict | None) -> float | None:
    """The number a probe is read by: follow gain for the shift, pickup retention for the elbow."""
    if m is None:
        return None
    return m.get("follow_gain_mean") if probe == "shift_placement" else m.get("pickup_retention")


def text(probe: str, m: dict | None) -> str:
    if m is None:
        return "-"
    if probe == "shift_placement":
        return f"{m['follow_gain_mean']:+.2f} {m['verdict']}"
    if probe == "swap_elbow":
        return f"{m['probe_pickups']}/{m['control_pickups']} {m['verdict']}"
    return f"{m['follow_gain_mean']:+.2f}, {m['probe_pickups']}/{m['control_pickups']} {m['verdict']}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=EXPERIMENTS / "recursive_generalization_20261119")
    p.add_argument("--probe-root", type=Path, default=EXPERIMENTS / "feature_probes_20261118")
    p.add_argument("--predictions", type=Path, default=ROOT / "notes/recursive-generalization-20261119-predictions.json")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    cells: dict[str, dict] = {}
    for root in (args.probe_root, args.root):
        for path in sorted((root / "cells").glob("*__*/result.json")):
            cells[path.parent.name] = json.loads(path.read_text())["metrics"]
    if not cells:
        raise SystemExit(f"no probe cells under {args.root} or {args.probe_root}")
    get = lambda cell, probe: cells.get(cell, {}).get(probe)

    # 1 and 2: one row per policy, execute and scene.
    stacking, combination = [], []
    for scene in ("so101", "so101_servo"):
        for label, name in STACK:
            for suffix in ("", "_exec30"):
                cell = f"{name}{suffix}__{scene}"
                if cell not in cells:
                    continue
                m = cells[cell]
                control = m["control"]
                stacking.append(dict(cell=cell, label=label, control=f"{control['successes']}/{control['scenarios']}",
                                     shift_placement=m.get("shift_placement"), swap_elbow=m.get("swap_elbow")))
                both, shift, swap = m.get("shift_and_swap"), m.get("shift_placement"), m.get("swap_elbow")
                if both is None or shift is None or swap is None:
                    continue
                keeps = [x.get("pickup_retention") for x in (shift, swap, both)]
                independent = None if None in keeps else keeps[0] * keeps[1]
                combination.append(dict(cell=cell, label=label, shift_and_swap=both, follow_gain=both["follow_gain_mean"],
                                        kept=keeps[2], kept_shift_alone=keeps[0], kept_swap_alone=keeps[1], kept_if_independent=independent))
    families: dict[str, list[dict]] = {}
    for cell, m in cells.items():           # CNN seeds pooled (pickups summed, gains averaged)
        policy, scene = cell.split("__", 1)
        family = re.sub(r"_s\d+$", "", policy)
        if family != policy and family.endswith("_cnn") and all(k in m for k in ("shift_placement", "swap_elbow", "shift_and_swap")):
            families.setdefault(f"{family}__{scene}", []).append(m)
    for family, members in families.items():
        pool = lambda probe: dict(probe_pickups=sum(m[probe]["probe_pickups"] for m in members), control_pickups=sum(m[probe]["control_pickups"] for m in members))
        keep = {probe: (lambda c: c["probe_pickups"] / c["control_pickups"] if c["control_pickups"] else None)(pool(probe))
                for probe in ("shift_placement", "swap_elbow", "shift_and_swap")}
        both = dict(pool("shift_and_swap"), follow_gain_mean=float(np.mean([m["shift_and_swap"]["follow_gain_mean"] for m in members])),
                    follow_gain_ci95=None, pickup_retention=keep["shift_and_swap"])
        both["verdict"] = verdict("shift_and_swap", both)
        independent = None if None in keep.values() else keep["shift_placement"] * keep["swap_elbow"]
        combination.append(dict(cell=f"{family} ({len(members)} seeds)", label="", shift_and_swap=both, follow_gain=both["follow_gain_mean"],
                                kept=keep["shift_and_swap"], kept_shift_alone=keep["shift_placement"], kept_swap_alone=keep["swap_elbow"],
                                kept_if_independent=independent))

    # 3: target feature against new data, per direction and start; N = 0 is the start checkpoint's own cell.
    order: dict[str, dict] = {}
    for cell, m in cells.items():
        policy, scene = cell.split("__", 1)
        match = CURRICULUM.match(policy)
        if not match or scene != "so101":
            continue
        direction, start, n, seed = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
        entry = order.setdefault(direction, {}).setdefault(start, {}).setdefault(n, [])
        entry.append(dict(seed=seed, target=value(TARGET[direction], m.get(TARGET[direction])), other=value(OTHER[direction], m.get(OTHER[direction])),
                          control_successes=m["control"]["successes"]))
    order_summary = {}
    for direction, starts in order.items():
        for start in starts:
            zero = cells.get(f"{START_CELL[start]}__so101", {})
            starts[start][0] = [dict(seed=None, target=value(TARGET[direction], zero.get(TARGET[direction])),
                                     other=value(OTHER[direction], zero.get(OTHER[direction])),
                                     control_successes=zero.get("control", {}).get("successes"))]
        mean = lambda start, n: (lambda v: float(np.mean(v)) if v else None)([r["target"] for r in starts.get(start, {}).get(n, []) if r["target"] is not None])
        sizes = sorted(n for n in set().union(*[set(s) for s in starts.values()]) if n > 0)
        has, lacks = HAS_OTHER[direction], "base"
        pairs = [(n, mean(has, n), mean(lacks, n)) for n in sizes]
        done = [(n, a, b) for n, a, b in pairs if a is not None and b is not None]
        lead = float(np.mean([a - b for _, a, b in done])) if done else None
        order_summary[direction] = dict(target_probe=TARGET[direction], start_with_other_feature=has, pairs=[dict(n=n, with_other=a, without=b) for n, a, b in pairs],
                                        mean_lead=lead, complete=len(done) == len(pairs) and len(done) >= 3,
                                        order_helps=None if lead is None else bool(lead >= ORDER_MARGIN))

    predictions = json.loads(args.predictions.read_text()) if args.predictions.exists() else {}
    checks = []
    for key, expected in predictions.get("verdicts", {}).items():          # "<cell>:<probe>" -> allowed verdicts
        cell, probe = key.rsplit(":", 1)
        got = (get(cell, probe) or {}).get("verdict")
        checks.append(dict(cell=cell, probe=probe, predicted=expected, observed=got, held=None if got is None else got in expected))
    out = dict(stacking=stacking, combination=combination, order=order, order_summary=order_summary, predictions=checks)
    target = args.out or args.root / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")

    print("1 STACKING (shift: follow gain; elbow: pickups kept / unprobed pickups)")
    for r in stacking:
        print(f"  {r['cell']:<34}{r['label']:<20}control {r['control']:>8}   shift {text('shift_placement', r['shift_placement']):<16} elbow {text('swap_elbow', r['swap_elbow'])}")
    print("2 COMBINATION (both changes in one episode; 'independent' = shift-alone kept x elbow-alone kept)")
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    for r in combination:
        print(f"  {r['cell']:<34}{r['label']:<20}{text('shift_and_swap', r['shift_and_swap']):<28} kept {fmt(r['kept'])} vs independent {fmt(r['kept_if_independent'])}")
    print("3 ORDER (target feature after 5000 steps on the first N placements of the new data; N = 0 is the start)")
    for direction, starts in order.items():
        for start, by_n in sorted(starts.items()):
            line = f"  {direction:<6} from {start:<7}"
            for n in sorted(by_n):
                rows = by_n[n]
                line += f"  N={n}: " + "/".join("-" if r["target"] is None else f"{r['target']:.2f}" for r in rows) + \
                        " (other " + "/".join("-" if r["other"] is None else f"{r['other']:.2f}" for r in rows) + ")"
            print(line)
        s = order_summary[direction]
        if s["mean_lead"] is not None:
            print(f"  {direction}: start with the other feature leads by {s['mean_lead']:+.2f} on {s['target_probe']} "
                  f"(registered margin {ORDER_MARGIN:.2f}) -> {'ORDER HELPS' if s['order_helps'] else 'no order effect'}{'' if s['complete'] else ' (incomplete)'}")
    for c in checks:
        print(f"PREDICTION {c['cell']}:{c['probe']} expected {c['predicted']} observed {c['observed']} -> "
              f"{'not run' if c['held'] is None else 'held' if c['held'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
