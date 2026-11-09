"""Per-pose closure (step 250) and step-179 lateral error, perturbed vs baseline ACT exec90 (2026-11-09)."""
import glob, json
from so_arm101_v2.simulation.closure import ClosureKinematics, closure_row
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
kin = ClosureKinematics("simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml")
for name, root in [("pert", f"{E}/perturbed_so101_20261108/act_ft/evaluations/exec90"),
                   ("base", f"{E}/so101_benchmark_20261106/so101/act_ft/evaluations/exec90")]:
    ev = json.load(open(glob.glob(root + "/policies/*/evaluation.json")[0]))
    cat = {r["scenario_id"]: r["failure_category"] for r in ev["rollouts"] if r["repeat"] == 0}
    print(name, "pose  lat179  lat250   dz250  min_height_200_260  category")
    for f in sorted(glob.glob(root + "/policies/*/telemetry/*repeat0.json")):
        t = json.load(open(f)); a = closure_row(kin, t, 179); b = closure_row(kin, t, 250)
        hmin = min(closure_row(kin, t, s)["height"] for s in range(180, 261, 5))
        print(f"  {t['scenario_id']}  {a['lat']:6.1f}  {b['lat']:6.1f}  {b['dz']:6.1f}  {hmin:8.1f}   {cat[t['scenario_id']]}")
