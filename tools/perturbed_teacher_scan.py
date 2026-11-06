"""Feasibility scan for the perturbed teacher (2026-11-05, notes/handoff-20261105.md step 1). CPU only, no artifacts but --out.

Runs the scripted teacher on the certification placements with its first descent shifted sideways (``--radius-mm``
in ``--directions`` world directions) at each ``--descend-mm`` height, and reports per height: admitted (success and
zero safety frames, as the suite screen requires), the jaw's lateral offset from the clean teacher at row 180 (what
the policy's second look sees) and at row 250 (after the correction, should be ~0).

  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot/bin/python tools/perturbed_teacher_scan.py --out <file.json>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
ROWS = (179, 180, 250)


def _episode(job):
    model, scenario, kwargs = job
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    adapter = MujocoTaskAdapter(model)
    try:
        contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
        adapter.reset(scenario)
        controller = PrivilegedStagedController()
        try:
            controller.reset(adapter, **kwargs)
        except (RuntimeError, ValueError) as exc:
            return dict(admitted=False, reason=f"plan: {exc}"[:120])
        state = PickPlaceEvaluationState()
        safety = 0
        jaw = {}
        result = None
        for row in range(contract.max_actions):
            command = adapter.apply_policy_command(controller.predict(None, adapter.current_act(), adapter))
            adapter.advance_control_period()
            if row in ROWS:   # telemetry row ``row`` = the state after action row+1, as closure.py reads it
                jaw[row] = ((adapter.data.site("fixed_jaw_tip").xpos + adapter.data.site("moving_jaw_tip").xpos) / 2).tolist()
            measurement, _ = adapter.pick_place_measurement(command)
            state, result = evaluate_pick_place_step(contract, measurement, state)
            safety += int(measurement.pickup.unsafe_contact + measurement.pickup.command_bound_violation
                          + measurement.pickup.delta_limiter_activated + measurement.pickup.nonfinite_command)
            if result.invalidated or result.terminated or result.truncated:
                break
        ok = bool(result is not None and result.success and not result.invalidated and safety == 0)
        reason = "success" if ok else ("safety" if safety else "incomplete")
        return dict(admitted=ok, reason=reason, safety_frames=safety, jaw=jaw, last_row=row)
    finally:
        adapter.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--radius-mm", type=float, default=15.0)
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--descend-mm", type=float, nargs="+", default=[8.0, 20.0, 30.0, 40.0])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import certification_suite
    bench = scene_bench_config(args.model)
    scenarios = certification_suite(bench, repeats=1).scenarios
    angles = np.linspace(0, 2 * np.pi, args.directions, endpoint=False)
    jobs, keys = [], []
    for i, s in enumerate(scenarios):
        jobs.append((str(args.model), s, {})); keys.append((i, "clean", None))
        for h in args.descend_mm:
            for a in angles:
                off = (args.radius_mm / 1000 * float(np.cos(a)), args.radius_mm / 1000 * float(np.sin(a)))
                kwargs = dict(lateral_offset_xy_m=off, perturbed_descend_height_m=None if h == 8.0 else h / 1000)
                jobs.append((str(args.model), replace(s, teacher_offset_xy_m=off, teacher_descend_height_m=kwargs["perturbed_descend_height_m"]), kwargs))
                keys.append((i, h, round(float(np.degrees(a)))))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_episode, jobs))
    clean = {k[0]: r for k, r in zip(keys, results) if k[1] == "clean"}
    report = dict(radius_mm=args.radius_mm, directions=args.directions, clean_admitted=sum(r["admitted"] for r in clean.values()), heights={})
    for h in args.descend_mm:
        rows = []
        for k, r in zip(keys, results):
            if k[1] != h:
                continue
            entry = dict(placement=k[0], direction_deg=k[2], admitted=r["admitted"], reason=r["reason"])
            c = clean[k[0]]
            if r.get("jaw") and c.get("jaw"):
                for row in ROWS:
                    if row in r["jaw"] and row in c["jaw"]:
                        entry[f"lateral_vs_clean_mm_{row}"] = 1000 * float(np.hypot(*(np.array(r["jaw"][row][:2]) - np.array(c["jaw"][row][:2]))))
            rows.append(entry)
        admitted = [e for e in rows if e["admitted"]]
        def stat(key):
            v = [e[key] for e in admitted if key in e]
            return None if not v else dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
        report["heights"][str(h)] = dict(admitted=len(admitted), total=len(rows), rate=len(admitted) / max(len(rows), 1),
                                         reasons={r: sum(e["reason"] == r for e in rows) for r in {e["reason"] for e in rows}},
                                         lateral_vs_clean_180=stat("lateral_vs_clean_mm_180"), lateral_vs_clean_250=stat("lateral_vs_clean_mm_250"),
                                         rows=rows)
        print(f"descend {h} mm: admitted {len(admitted)}/{len(rows)}  row180 {stat('lateral_vs_clean_mm_180')}  row250 {stat('lateral_vs_clean_mm_250')}", flush=True)
    print(f"clean teacher admitted {report['clean_admitted']}/{len(clean)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
