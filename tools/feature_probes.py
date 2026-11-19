"""Counterfactual feature probes for a finished policy (2026-11-18, notes/feature-probes-20261118.md).

Does a policy look, or does it remember? Each probe changes the world in the middle of an episode and compares the
episode with the same one left alone (so_arm101_v2/simulation/probes.py). Three subcommands:

  suite   screen a held-out suite of distinct placements with the teacher (the 30-rollout suites are 10 poses x 3)
  plan    per scene: the displacement each placement gets, admitted by the teacher (it must complete the displaced episode)
  run     one policy on one scene: the unprobed control and every probe on every placement -> <output-dir>/result.json
  report  collect <root>/cells/<policy>__<scene>/result.json into a table and check the pre-registered predictions

  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot/bin/python tools/feature_probes.py run --kind vla \\
      --checkpoint <act.pt> --execute 30 --scene <scene.xml> --suite <probe_suite.path> --output-dir <root>/cells/<policy>__<scene>

``--kind``: vla (ACT / SmolVLA / pi0.5 residual checkpoint, one process on the GPU), vision_chunked (CNN, ``--workers``
CPU processes), teacher and teacher_informed (the scripted teacher as the zero and one reference controls, no rendering).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_ROOT = EXPERIMENTS / "feature_probes_20261118"
SUITE_SEED = 31      # placement/appearance stream no training or held-out suite uses (8, 12-17 and 99 are taken)

# Pre-registered reading of each probe (notes/feature-probes-20261118.md, written before any learned policy was run).
GAIN_PRESENT, GAIN_PRESENT_CI_LOW, GAIN_ABSENT = 0.5, 0.25, 0.2
REGRASP_PRESENT, REGRASP_ABSENT = 0.5, 0.05
COPES, BREAKS = 0.8, 0.5
CONTROL_ZERO, CONTROL_ONE, CONTROL_ONE_SUCCESS = 0.15, (0.85, 1.15), 0.8


def verdict(name: str, m: dict) -> str:
    """present / partial / absent for the feature a probe tests; 'n/a' when the probe never fired or nothing compares."""
    if name in ("shift_placement", "shift_cube", "move_square"):
        gain, ci = m.get("follow_gain_mean"), m.get("follow_gain_ci95")
        if gain is None:
            return "n/a"
        if gain >= GAIN_PRESENT and (ci is None or ci[0] > GAIN_PRESENT_CI_LOW):
            return "present"
        return "absent" if gain <= GAIN_ABSENT else "partial"
    if name == "drop_cube":
        rate = m.get("regrasp_rate")
        return "n/a" if rate is None else "present" if rate >= REGRASP_PRESENT else "absent" if rate <= REGRASP_ABSENT else "partial"
    if name == "swap_elbow":
        keep = m.get("pickup_retention")
        return "n/a" if keep is None else "present" if keep >= COPES else "absent" if keep <= BREAKS else "partial"
    if name == "shift_and_swap":      # both features in one episode: the weaker of the two readings
        parts = [verdict("shift_placement", m), verdict("swap_elbow", m)]
        return "n/a" if "n/a" in parts else min(parts, key=("absent", "partial", "present").index)
    raise ValueError(name)


def headline(name: str, m: dict) -> str:
    if name in ("shift_placement", "shift_cube", "move_square"):
        gain = m.get("follow_gain_mean")
        return "-" if gain is None else f"{gain:+.2f}"
    if name == "shift_and_swap":
        gain = m.get("follow_gain_mean")
        return "-" if gain is None or m.get("pickup_retention") is None else f"{gain:+.2f} {m['probe_pickups']}/{m['control_pickups']}"
    if name == "drop_cube":
        return "-" if m.get("regrasp_rate") is None else f"{m['regrasps']}/{m['fired']}"
    return "-" if m.get("pickup_retention") is None else f"{m['probe_pickups']}/{m['control_pickups']}"


def load_suite(path: Path, limit: int | None):
    import dataclasses
    from so_arm101_v2.simulation.suites import load_suite_from_path
    suite_path = Path(path.read_text().strip()) if path.suffix == ".path" else path
    suite = load_suite_from_path(suite_path)
    if suite.repeats != 1:
        suite = dataclasses.replace(suite, repeats=1)       # the simulator is deterministic: repeats add nothing
    if limit is not None:
        suite = dataclasses.replace(suite, suite_id=f"{suite.suite_id}_first{limit}", scenarios=suite.scenarios[:limit])
    return suite, suite_path


def cmd_suite(args) -> int:
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import generate_bench_suite
    root = args.output_dir.resolve()
    pointer = root / "probe_suite.path"
    if pointer.exists():
        print(f"SUITE exists: {pointer.read_text().strip()}")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    suite, path = generate_bench_suite(args.scene, scene_bench_config(args.scene), seed=args.seed, count=args.count, repeats=1,
                                       output_dir=root / "suites", randomize_appearance=True, randomize_placement=True, workers=args.workers)
    pointer.write_text(str(Path(path).resolve()) + "\n")
    print(f"SUITE {suite.suite_id}: {len(suite.scenarios)} teacher-screened placements, seed {args.seed} -> {path}")
    return 0


def scene_name(scene: Path) -> str:
    return scene.resolve().parent.name.replace("bench_pick_replace_v1_", "")


def cmd_plan(args) -> int:
    from so_arm101_v2.contracts.bench import scene_dependency_hash
    from so_arm101_v2.simulation import probes as P
    suite, suite_path = load_suite(args.suite, args.limit)
    out = args.root / "plans" / f"{scene_name(args.scene)}.json"
    if out.exists():
        print(f"PLAN exists: {out}")
        return 0
    started = time.time()
    shifts = P.plan_shifts(args.scene, suite, seed=args.probe_seed, magnitude_m=args.magnitude_mm / 1000.0,
                           cube_magnitude_m=args.cube_magnitude_mm / 1000.0, at_action=args.at_action, max_draws=args.max_draws, workers=args.workers)
    payload = dict(scene=str(args.scene.resolve()), scene_dependencies_sha256=scene_dependency_hash(args.scene), suite=suite.suite_id,
                   source_suite=str(suite_path), probe_seed=args.probe_seed, at_action=args.at_action, magnitude_m=args.magnitude_mm / 1000.0,
                   cube_magnitude_m=args.cube_magnitude_mm / 1000.0, max_draws=args.max_draws, shifts=shifts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    count = lambda key: sum(v[key] is not None for v in shifts.values())
    print(f"PLAN {scene_name(args.scene)}: placement shift admitted for {count('placement')}/{len(shifts)} placements, "
          f"cube shift for {count('cube')}/{len(shifts)}, {time.time() - started:.0f} s -> {out}")
    return 0


def cmd_run(args) -> int:
    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    from so_arm101_v2.contracts.bench import scene_dependency_hash
    from so_arm101_v2.simulation import probes as P
    if args.kind in ("vla", "vision_chunked") and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for a learned policy")
    suite, suite_path = load_suite(args.suite, args.limit)
    names = ["control"] + [n for n in (args.probes or P.PROBE_NAMES[1:]) if n != "control"]
    for name in list(names):          # a probe read against another probe's episodes runs after it
        needs = P.GAIN_REFERENCE.get(name)
        if needs and needs not in names:
            names.insert(names.index(name), needs)
    names.sort(key=lambda n: n in P.GAIN_REFERENCE)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = args.plan or root.parent.parent / "plans" / f"{scene_name(args.scene)}.json"
    if not plan_path.exists():
        raise SystemExit(f"no shift plan at {plan_path}: run `feature_probes.py plan --scene {args.scene}` first")
    plan = json.loads(plan_path.read_text())
    if plan["scene_dependencies_sha256"] != scene_dependency_hash(args.scene) or plan["source_suite"] != str(suite_path):
        raise SystemExit(f"{plan_path} was made for another scene or suite")
    for name in ("probe_seed", "at_action"):
        if plan[name] != getattr(args, name):
            raise SystemExit(f"{plan_path} was made with {name}={plan[name]}")
    if abs(plan["magnitude_m"] * 1000 - args.magnitude_mm) > 1e-9 or abs(plan["cube_magnitude_m"] * 1000 - args.cube_magnitude_mm) > 1e-9:
        raise SystemExit(f"{plan_path} was made with other magnitudes")
    identity = dict(tool="feature_probes", kind=args.kind, checkpoint=None if args.checkpoint is None else str(args.checkpoint.resolve()),
                    scene=str(args.scene.resolve()), scene_dependencies_sha256=scene_dependency_hash(args.scene), suite=suite.suite_id,
                    source_suite=str(suite_path), scenarios=len(suite.scenarios), execute=args.execute, probe_seed=args.probe_seed,
                    at_action=args.at_action, magnitude_m=args.magnitude_mm / 1000.0, cube_magnitude_m=args.cube_magnitude_mm / 1000.0,
                    plan=str(plan_path.resolve()))
    stored = root / "identity.json"
    if stored.exists() and json.loads(stored.read_text()) != identity:
        raise SystemExit(f"{root} holds a different probe run (identity.json differs); use a new --output-dir")
    stored.write_text(json.dumps(identity, indent=2) + "\n")
    model = None
    if args.kind == "vla":
        from so_arm101_v2.learning.vla_models import load_model
        model = load_model(args.checkpoint)
    started = time.time()
    rows: dict[str, list] = {}
    traces: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    for name in names:
        spec = P.ProbeSpec(name, at_action=args.at_action, magnitude_m=args.magnitude_mm / 1000.0,
                           cube_magnitude_m=args.cube_magnitude_mm / 1000.0, seed=args.probe_seed)
        rows_path, trace_path = root / f"rows_{name}.json", root / f"trace_{name}.npz"
        if rows_path.exists() and trace_path.exists():
            rows[name] = json.loads(rows_path.read_text())
            traces[name] = dict(np.load(trace_path))
        else:
            t = time.time()
            rows[name], traces[name] = P.run_probe(args.scene, suite, args.kind, None if args.checkpoint is None else str(args.checkpoint),
                                                   args.execute, spec, workers=args.workers, model=model, plan=plan["shifts"])
            np.savez_compressed(trace_path, **traces[name])
            rows_path.write_text(json.dumps(rows[name]) + "\n")       # written last: its presence marks the probe complete
            print(f"PROBE {name}: {len(rows[name])} episodes in {time.time() - t:.0f} s", flush=True)
        if name == "control":
            picked = sum(bool(r["pickup_completed"]) for r in rows[name])
            metrics[name] = dict(scenarios=len(rows[name]), successes=sum(bool(r["success"]) for r in rows[name]), pickups=picked,
                                 limit_frames=sum(r["limit_frames"] for r in rows[name]))
        else:
            needs = P.GAIN_REFERENCE.get(name)
            metrics[name] = P.paired_metrics(spec, rows["control"], traces["control"], rows[name], traces[name],
                                             reference=(rows[needs], traces[needs]) if needs else None)
            metrics[name]["verdict"] = verdict(name, metrics[name])
    result = dict(identity=identity, metrics=metrics, seconds=round(time.time() - started, 1))
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    control = metrics["control"]
    print(f"PROBES {args.kind} {root.name}: control {control['successes']}/{control['scenarios']} successes, {control['pickups']} pickups; "
          + ", ".join(f"{n} {headline(n, m)} ({m['verdict']})" for n, m in metrics.items() if n != "control"), flush=True)
    return 0


def cmd_report(args) -> int:
    from so_arm101_v2.simulation.probes import PROBE_NAMES
    cells: dict[str, dict] = {}
    for path in sorted((args.root / "cells").glob("*__*/result.json")):
        cells[path.parent.name] = json.loads(path.read_text())
    if not cells:
        raise SystemExit(f"no result.json under {args.root / 'cells'}")
    probes = PROBE_NAMES[1:]
    # CNN seeds pooled per family: rates summed, gains averaged over seeds (each seed's gain is a mean over the same placements).
    families: dict[str, list[str]] = {}
    for cell in cells:
        policy, scene = cell.split("__", 1)
        family = re.sub(r"_s\d+(?=_exec\d+$|$)", "", policy)
        if family != policy:
            families.setdefault(f"{family}__{scene}", []).append(cell)
    table: dict[str, dict] = {}
    for cell, result in cells.items():
        m = result["metrics"]
        table[cell] = dict(control=m["control"], **{p: m[p] for p in probes if p in m})
    pooled: dict[str, dict] = {}
    for family, members in families.items():
        entry: dict[str, dict] = dict(members=members, control=dict(
            scenarios=sum(table[c]["control"]["scenarios"] for c in members), successes=sum(table[c]["control"]["successes"] for c in members),
            pickups=sum(table[c]["control"]["pickups"] for c in members)))
        for p in probes:
            have = [table[c][p] for c in members if p in table[c]]
            if not have:
                continue
            agg: dict = {k: sum(h[k] for h in have) for k in ("fired", "control_pickups", "probe_pickups", "probe_successes", "control_successes")}
            agg["pickup_retention"] = agg["probe_pickups"] / agg["control_pickups"] if agg["control_pickups"] else None
            gains = [h["follow_gain_mean"] for h in have if h.get("follow_gain_mean") is not None]
            if gains:
                agg["follow_gain_mean"] = float(np.mean(gains))
                agg["follow_gain_seeds"] = gains
                agg["follow_gain_ci95"] = None        # seeds share placements: no pooled interval, read the per-seed ones
            if "regrasps" in have[0]:
                agg["regrasps"] = sum(h["regrasps"] for h in have)
                agg["regrasp_rate"] = agg["regrasps"] / agg["fired"] if agg["fired"] else None
            agg["verdict"] = verdict(p, agg)
            entry[p] = agg
        pooled[family] = entry
    # The reference controls decide whether a probe may be read at all.
    controls: dict[str, dict] = {}
    for cell, entry in table.items():
        policy, scene = cell.split("__", 1)
        if policy not in ("teacher", "teacher_informed"):
            continue
        for p in ("shift_placement", "shift_cube", "move_square", "shift_and_swap"):
            if p not in entry:
                continue
            gain = entry[p].get("follow_gain_mean")
            if policy == "teacher":
                ok = gain is not None and abs(gain) <= CONTROL_ZERO
            else:
                # shift_and_swap: the teacher's plan does not survive the changed elbow, so only its gain is a control
                rate = 1.0 if p == "shift_and_swap" else entry[p]["probe_successes"] / max(entry[p]["scenarios"], 1)
                ok = gain is not None and CONTROL_ONE[0] <= gain <= CONTROL_ONE[1] and rate >= CONTROL_ONE_SUCCESS
            controls.setdefault(scene, {})[f"{policy}:{p}"] = dict(gain=gain, ok=bool(ok))
    predictions = json.loads(args.predictions.read_text()) if args.predictions and args.predictions.exists() else {}
    checks = []
    for key, expected in predictions.items():          # "<policy or family>__<scene>:<probe>" -> list of allowed verdicts
        cell, probe = key.rsplit(":", 1)
        got = (pooled.get(cell) or table.get(cell) or {}).get(probe, {}).get("verdict")
        checks.append(dict(cell=cell, probe=probe, predicted=expected, observed=got, held=None if got is None else got in expected))
    # Link 3 (the feature is what buys success under the shift): follow gain against pickup retention, one point per cell.
    points = [(c, e["shift_placement"]["follow_gain_mean"], e["shift_placement"]["pickup_retention"]) for c, e in table.items()
              if "shift_placement" in e and not c.startswith("teacher") and e["shift_placement"].get("pickup_retention") is not None]
    link3 = None
    if len(points) >= 4:
        gains, keeps = np.array([p[1] for p in points]), np.array([p[2] for p in points])
        rank = lambda v: np.argsort(np.argsort(v)).astype(float)
        link3 = dict(cells=len(points), spearman=float(np.corrcoef(rank(gains), rank(keeps))[0, 1]),
                     points=[dict(cell=c, follow_gain=g, pickup_retention=k) for c, g, k in points])
    out = dict(cells=table, pooled=pooled, controls=controls, predictions=checks, link3=link3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    width = max(len(c) for c in list(table) + list(pooled)) + 2
    print(f"{'cell':<{width}}{'control':>10}" + "".join(f"{p:>24}" for p in probes))
    for label, source in (("", table), ("pooled ", pooled)):
        for cell, entry in source.items():
            control = entry["control"]
            line = f"{label + cell:<{width}}{str(control['successes']) + '/' + str(control['scenarios']):>10}"
            for p in probes:
                line += f"{(headline(p, entry[p]) + ' ' + entry[p]['verdict']) if p in entry else '-':>24}"
            print(line)
    for scene, items in controls.items():
        print(f"CONTROLS {scene}: " + ", ".join(f"{k} {v['gain']:+.2f} {'ok' if v['ok'] else 'FAILED'}" for k, v in items.items()))
    for c in checks:
        print(f"PREDICTION {c['cell']}:{c['probe']} expected {c['predicted']} observed {c['observed']} -> "
              f"{'not run' if c['held'] is None else 'held' if c['held'] else 'FAILED'}")
    if link3:
        print(f"LINK 3: Spearman(follow gain, pickup retention) = {link3['spearman']:+.2f} over {link3['cells']} cells")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("suite")
    s.add_argument("--scene", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--seed", type=int, default=SUITE_SEED)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_suite)
    n = sub.add_parser("plan")
    n.add_argument("--scene", type=Path, required=True)
    n.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    n.add_argument("--suite", type=Path, default=DEFAULT_ROOT / "probe_suite.path")
    n.add_argument("--at-action", type=int, default=100)
    n.add_argument("--magnitude-mm", type=float, default=20.0)
    n.add_argument("--cube-magnitude-mm", type=float, default=10.0)
    n.add_argument("--probe-seed", type=int, default=0)
    n.add_argument("--max-draws", type=int, default=12)
    n.add_argument("--workers", type=int, default=8)
    n.add_argument("--limit", type=int, default=None)
    n.set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--kind", required=True, choices=("vla", "vision_chunked", "teacher", "teacher_informed"))
    r.add_argument("--checkpoint", type=Path, default=None)
    r.add_argument("--scene", type=Path, required=True)
    r.add_argument("--suite", type=Path, default=DEFAULT_ROOT / "probe_suite.path", help="suite.json, or a *.path pointer file naming one")
    r.add_argument("--execute", type=int, default=90)
    r.add_argument("--probes", nargs="*", default=None, help="default: every probe; the control always runs")
    r.add_argument("--at-action", type=int, default=100, help="shift_* and swap_elbow: after the survey frame (action 90)")
    r.add_argument("--magnitude-mm", type=float, default=20.0, help="shift_placement and move_square")
    r.add_argument("--cube-magnitude-mm", type=float, default=10.0, help="shift_cube (the cube stays on its square)")
    r.add_argument("--probe-seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=8, help="CPU kinds only; vla runs in one process on the GPU")
    r.add_argument("--limit", type=int, default=None, help="first N placements only (rehearsals)")
    r.add_argument("--plan", type=Path, default=None, help="shift plan (default: <output-dir>/../../plans/<scene>.json)")
    r.add_argument("--output-dir", type=Path, required=True)
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("report")
    t.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    t.add_argument("--predictions", type=Path, default=ROOT / "notes/feature-probes-20261118-predictions.json")
    t.add_argument("--out", type=Path, default=None)
    t.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    if args.command == "report" and args.out is None:
        args.out = args.root / "matrix.json"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
