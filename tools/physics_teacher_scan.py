"""Feasibility scan for physics randomization (2026-11-05, notes/handoff-20261105.md step 2). CPU only; writes --out only.

Runs the scripted teacher (servo-compensated on the measured-servo scene) on the certification placements under
``--draws`` physics draws each, exactly as the suite screen does (inference pause every 90 actions), and reports:
admitted (success with zero safety frames), the failure kinds (relative-limit holds, unsafe contact, incomplete), and
for each randomized parameter the failure rate in its top and bottom quintile, so a range that the teacher cannot
handle shows up before any capture is queued.

  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot/bin/python tools/physics_teacher_scan.py --out <file.json>
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
SERVO_SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1_servo/scene_bench_pick_replace_v1.xml"


def _episode(job):
    model, scenario = job
    from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    adapter = MujocoTaskAdapter(model)
    counts = dict(limit=0, unsafe=0, clip=0, nonfinite=0)
    try:
        contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=adapter.bench)
        try:
            adapter.reset(scenario)
            controller = PrivilegedStagedController()
            controller.reset(adapter)
        except (RuntimeError, ValueError) as exc:
            return dict(admitted=False, reason=f"reset/plan: {exc}"[:100], counts=counts)
        state, result, pause_total = PickPlaceEvaluationState(), None, 0.0
        for action in range(contract.max_actions):
            requested = controller.predict(None, adapter.current_act(), adapter)
            if action % 90 == 0:
                pause_total += adapter.inference_pause(action)
            command = adapter.apply_policy_command(requested)
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(command)
            p = measurement.pickup
            counts["limit"] += int(p.delta_limiter_activated); counts["unsafe"] += int(p.unsafe_contact)
            counts["clip"] += int(p.command_bound_violation); counts["nonfinite"] += int(p.nonfinite_command)
            state, result = evaluate_pick_place_step(contract, measurement, state)
            if result.invalidated or result.terminated or result.truncated:
                break
        ok = bool(result is not None and result.success and not result.invalidated and not any(counts.values()))
        if ok:
            reason = "success"
        elif counts["unsafe"]:
            reason = "unsafe_contact"
        elif counts["limit"]:
            reason = "relative_limit_hold"
        elif counts["clip"]:
            reason = "command_bound"
        else:
            reason = "incomplete"
        return dict(admitted=ok, reason=reason, counts=counts, pause_total_s=pause_total)
    finally:
        adapter.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=SERVO_SCENE)
    p.add_argument("--regime", default="bench_physics_v1")
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.contracts.physics import physics_regime, physics_seeds, resolve_physics
    from so_arm101_v2.simulation.bench import certification_suite
    regime = physics_regime(args.regime)
    scenarios = certification_suite(scene_bench_config(args.model), repeats=1).scenarios
    seeds = physics_seeds(args.seed, len(scenarios) * args.draws, stream=9)
    jobs, meta = [], []
    for i, s in enumerate(scenarios):
        jobs.append((str(args.model), s)); meta.append(dict(placement=i, seed=None))
        for k in range(args.draws):
            seed = seeds[i * args.draws + k]
            jobs.append((str(args.model), replace(s, physics_regime=regime.name, physics_seed=seed))); meta.append(dict(placement=i, seed=seed))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_episode, jobs, chunksize=2))
    clean = [r for m, r in zip(meta, results) if m["seed"] is None]
    rows = []
    for m, r in zip(meta, results):
        if m["seed"] is None:
            continue
        params = resolve_physics(regime, m["seed"])
        rows.append(dict(placement=m["placement"], seed=m["seed"], admitted=r["admitted"], reason=r["reason"], counts=r["counts"],
                         kp_scale=params.kp_scale, kv_scale=params.kv_scale, friction_scale=params.friction_scale,
                         arm_mass_scale=params.arm_mass_scale, latency_ms=1000 * params.latency_s, cube_mass_g=1000 * params.cube_mass_kg,
                         cube_friction=params.cube_friction, pause_total_s=r.get("pause_total_s")))
    failed = np.array([not r["admitted"] for r in rows])
    features = {"latency_ms": [r["latency_ms"] for r in rows], "arm_mass_scale": [r["arm_mass_scale"] for r in rows],
                "cube_mass_g": [r["cube_mass_g"] for r in rows], "cube_friction": [r["cube_friction"] for r in rows],
                "pause_total_s": [r["pause_total_s"] or 0.0 for r in rows]}
    for j, joint in enumerate(("pan", "lift", "elbow", "wflex", "wroll", "grip")):
        features[f"kp_{joint}"] = [r["kp_scale"][j] for r in rows]
        features[f"friction_{joint}"] = [r["friction_scale"][j] for r in rows]
    quintiles = {}
    for name, values in features.items():
        v = np.asarray(values, dtype=np.float64)
        low, high = np.quantile(v, 0.2), np.quantile(v, 0.8)
        quintiles[name] = dict(fail_rate_bottom=float(failed[v <= low].mean()), fail_rate_top=float(failed[v >= high].mean()))
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    report = dict(model=str(args.model), regime=regime.identity(), clean_admitted=sum(r["admitted"] for r in clean), clean_total=len(clean),
                  admitted=int((~failed).sum()), total=len(rows), rate=float((~failed).mean()), reasons=reasons,
                  quintiles=dict(sorted(quintiles.items(), key=lambda kv: -max(kv[1].values()))), rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"clean teacher {report['clean_admitted']}/{report['clean_total']}; randomized {report['admitted']}/{report['total']} "
          f"({100 * report['rate']:.0f}%); reasons {reasons}")
    for name, q in list(report["quintiles"].items())[:8]:
        print(f"  {name:16s} fail bottom quintile {q['fail_rate_bottom']:.2f}  top quintile {q['fail_rate_top']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
