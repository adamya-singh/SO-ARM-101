"""Teacher ceiling on every cell of the physics-v2 evaluation matrix: the scripted teacher on exactly the 30 rollouts
(same held-out poses, same per-rollout physics draws) each policy faced. CPU, 2026-11-09."""
import json, sys, multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, "src")
from physics_teacher_scan import _episode
from eval_policy import physics_suite
from so_arm101_v2.simulation.suites import load_suite_from_path
E = Path("artifacts/so_arm101_v2/bench_pick_replace_v1/experiments")
M = Path("simulation_code/model")
SCN = {"nominal": M / "bench_pick_replace_v1_so101", "servo": M / "bench_pick_replace_v1_so101_servo",
       "stiff_elbow": E / "next_round_20261108/scenes/so101_servo_stiff_elbow",
       "low_friction": E / "next_round_20261108/scenes/so101_servo_low_friction"}
CELLS = {"nominal": ("nominal", "none"), "servo": ("servo", "none"), "servo_real_like_v1": ("servo", "real_like_v1"),
         "servo_bench_physics_v2": ("servo", "bench_physics_v2"), "stiff_elbow": ("stiff_elbow", "none"), "low_friction": ("low_friction", "none")}
suite_path = Path((E / "so101_benchmark_20261106/so101/cap4800/heldout_suite.path").read_text().strip())
if __name__ == "__main__":
    out = {}
    for cell, (scene, physics) in CELLS.items():
        suite = load_suite_from_path(suite_path)
        if physics != "none":
            suite = physics_suite(suite, physics)
        model = str(SCN[scene] / "scene_bench_pick_replace_v1.xml")
        jobs = [(model, s) for s in suite.scenarios]   # repeats are identical without physics draws
        with ProcessPoolExecutor(14, mp_context=multiprocessing.get_context("spawn")) as pool:
            res = list(pool.map(_episode, jobs))
        reps = 1 if physics != "none" else suite.repeats
        ok = sum(r["admitted"] for r in res) * reps
        reasons = {}
        for r in res:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + reps
        out[cell] = dict(admitted=ok, total=len(res) * reps, reasons=reasons,
                         per_scenario={s.scenario_id: r["admitted"] for s, r in zip(suite.scenarios, res)})
        print(f"{cell:24s} teacher {ok}/{len(res) * reps}  {reasons}", flush=True)
    (E / "analysis_20261109/teacher_ceiling.json").write_text(json.dumps(out, indent=1) + "\n")
