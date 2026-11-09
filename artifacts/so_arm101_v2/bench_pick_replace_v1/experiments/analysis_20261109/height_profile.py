"""Jaw height over the cube (mm) and lateral error vs time, median across rollouts (2026-11-09 analysis)."""
import glob, json, sys
import numpy as np
from so_arm101_v2.simulation.closure import ClosureKinematics, closure_row
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
SO101 = "simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml"
SETS = {
    "pert_act_exec90": f"{E}/perturbed_so101_20261108/act_ft/evaluations/exec90",
    "pert_act_exec30": f"{E}/perturbed_so101_20261108/act_ft/evaluations/exec30",
    "base_act_exec90": f"{E}/so101_benchmark_20261106/so101/act_ft/evaluations/exec90",
    "base_act_exec30": f"{E}/so101_benchmark_20261106/so101/act_ft/evaluations/exec30",
}
STEPS = list(range(60, 331, 30))
kin = ClosureKinematics(SO101)
print("height of jaw midpoint over cube start (mm), median over 30 rollouts; step 250 ref height %.1f" % 32.6)
print("set".ljust(18) + "".join(str(s).rjust(7) for s in STEPS))
for name, root in SETS.items():
    files = sorted(glob.glob(root + "/policies/*/telemetry/*.json"))
    H = []
    for f in files:
        t = json.load(open(f))
        H.append([closure_row(kin, t, s)["height"] for s in STEPS])
    H = np.array(H)
    print(name.ljust(18) + "".join(f"{v:7.1f}" for v in np.median(H, 0)), f"(n={len(files)})")
