"""Which safety mechanism invalidates the physics-v2 policies' rollouts, per cell (2026-11-09)."""
import glob, json, collections
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/physics_v2_20261108/evals"
KEYS = ("clipping_frames", "limiting_frames", "unsafe_contact_frames", "nonfinite_frames")
for pol in ["physv2_act_ft", "physv2_act_ft_exec30", "physv2_cnn_s303", "so101_act_ft"]:
    for cell in ["servo", "servo_real_like_v1", "servo_bench_physics_v2", "nominal"]:
        ev = json.load(open(glob.glob(f"{E}/{pol}__{cell}/evaluation/policies/*/evaluation.json")[0]))
        rs = ev["rollouts"]
        inval = [r for r in rs if r["failure_category"] == "safety_invalidation"]
        kinds = collections.Counter(max(KEYS, key=lambda k: r[k]) for r in inval)
        tot = {k: sum(r[k] for r in rs) for k in KEYS}
        print(f"{pol:22s} {cell:24s} wins {sum(r['success'] for r in rs):2d}  invalidated {len(inval):2d}  dominant {dict(kinds)}  frames {tot}")

# Relative-limit holds per draw in the v2 cell vs the draw's elbow damping and latency (pooled physics-v2 policies).
import numpy as np
draws = json.load(open("artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/analysis_20261109/v2_cell_draws.json"))["draws"]
lim = collections.Counter(); won = collections.Counter()
for cell in glob.glob(f"{E}/physv2_*__servo_bench_physics_v2"):
    ev = json.load(open(glob.glob(cell + "/evaluation/policies/*/evaluation.json")[0]))
    for r in ev["rollouts"]:
        lim[r["scenario_id"]] += r["limiting_frames"]; won[r["scenario_id"]] += int(r["success"])
ids = sorted(draws)
def rank(x): return np.argsort(np.argsort(np.asarray(x, float)))
L = [lim[s] for s in ids]
for name, f in [("kv_elbow", lambda d: d["kv_scale"][2]), ("latency", lambda d: d["latency_s"]), ("kp_elbow", lambda d: d["kp_scale"][2]),
                ("kp_lift", lambda d: d["kp_scale"][1]), ("kv/kp elbow", lambda d: d["kv_scale"][2] / d["kp_scale"][2])]:
    x = [f(draws[s]) for s in ids]
    print(f"limit frames vs {name:12s} rho {np.corrcoef(rank(x), rank(L))[0,1]:+.2f}   success vs it {np.corrcoef(rank(x), rank([won[s] for s in ids]))[0,1]:+.2f}")
tc = json.load(open("artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/analysis_20261109/teacher_ceiling.json"))["servo_bench_physics_v2"]["per_scenario"]
print("teacher-failed draws:", [s for s, ok in tc.items() if not ok], " policy wins on them:", [won[s] for s, ok in tc.items() if not ok])
print("policy wins on teacher-solved draws:", sum(won[s] for s, ok in tc.items() if ok), "/", 7 * sum(tc.values()))
