"""The fully random physics_v2 cell: which physics draws fail? Pools the 7 physics-v2 policies and the 3 baselines.

Each of the 30 rollouts in the cell draws its own physics (pose_k_r0..r2); the draw depends only on the scenario,
so every policy sees the same 30 draws. Success per draw vs each sampled parameter (2026-11-09).
"""
import glob, json, collections
import numpy as np
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/physics_v2_20261108/evals"
JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
draw, wins, cat = {}, collections.defaultdict(dict), collections.defaultdict(dict)
for cell in sorted(glob.glob(f"{E}/*__servo_bench_physics_v2")):
    pol = cell.split("/")[-1].split("__")[0]
    ev = json.load(open(glob.glob(cell + "/evaluation/policies/*/evaluation.json")[0]))
    for r in ev["rollouts"]:
        wins[r["scenario_id"]][pol] = bool(r["success"]); cat[r["scenario_id"]][pol] = r["failure_category"]
    for f in glob.glob(cell + "/evaluation/policies/*/telemetry/*.json"):
        t = json.load(open(f)); draw.setdefault(t["scenario_id"], t["physics"]["params"])
ids = sorted(draw)
pols = sorted({p for s in wins.values() for p in s})
v2 = [p for p in pols if p.startswith("physv2")]
print("policies:", pols)
# per-draw success count over the physics-v2 policies
succ = np.array([sum(wins[s][p] for p in v2) for s in ids])
feat = {
    **{f"kp_{j}": [draw[s]["kp_scale"][i] for s in ids] for i, j in enumerate(JOINTS[:5])},
    **{f"fric_{j}": [draw[s]["friction_scale"][i] for s in ids] for i, j in enumerate(JOINTS[:5])},
    **{f"kv_{j}": [draw[s]["kv_scale"][i] for s in ids] for i, j in enumerate(JOINTS[:5])},
    "latency_s": [draw[s]["latency_s"] for s in ids],
    "pause_s": [draw[s]["pause_ms"][1] / 1000.0 for s in ids],
    "arm_mass": [draw[s]["arm_mass_scale"] for s in ids],
    "cube_mass": [draw[s]["cube_mass_kg"] for s in ids],
    "cube_fric": [draw[s]["cube_friction"] for s in ids],
}
print(f"\nsuccess per draw over {len(v2)} physics-v2 policies: total {succ.sum()}/{len(v2)*len(ids)}")
print("Spearman correlation of per-draw success with each parameter (|rho| > 0.36 ~ p<0.05 at n=30):")
def rank(x): return np.argsort(np.argsort(x)).astype(float)
res = sorted(((float(np.corrcoef(rank(np.array(v)), rank(succ))[0, 1]), k) for k, v in feat.items()), key=lambda t: -abs(t[0]))
for rho, k in res: print(f"  {k:12s} {rho:+.2f}")
print("\nper draw: id  wins/7  latency  pause  kp_elbow fric_elbow kp_lift  baselines(won)")
for s, n in sorted(zip(ids, succ), key=lambda t: t[1]):
    d = draw[s]; base = "".join("1" if wins[s].get(p) else "0" for p in pols if not p.startswith("physv2"))
    print(f"  {s:12s} {n}  {d['latency_s']*1000:5.0f}ms  {d['pause_ms'][1]/1000:4.1f}s  {d['kp_scale'][2]:5.2f}  {d['friction_scale'][2]:5.2f}  {d['kp_scale'][1]:5.2f}   {base}")
c = collections.Counter(cat[s][p] for s in ids for p in v2)
print("\nfailure categories over physics-v2 policies:", dict(c))
json.dump({"ids": ids, "draws": draw, "wins": wins, "spearman": res}, open(f"{E}/../../analysis_20261109/v2_cell_draws.json", "w"), indent=1)
