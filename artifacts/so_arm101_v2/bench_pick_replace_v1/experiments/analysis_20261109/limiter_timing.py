"""When do relative-limit holds start, and on which joint, in the physics-v2 cell? (2026-11-09)
The limiter fires when a command is more than 20 physical units from the measured joint."""
import glob, json, collections
import numpy as np
from so_arm101_v2.contracts.physical import act_to_physical_normalized
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/physics_v2_20261108/evals"
JN = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
for pol in ["physv2_act_ft", "physv2_act_ft_exec30", "physv2_cnn_s303"]:
    first, joint, phase = [], collections.Counter(), collections.Counter()
    for f in glob.glob(f"{E}/{pol}__servo_bench_physics_v2/evaluation/policies/*/telemetry/*.json"):
        rows = json.load(open(f))["rows"]
        hits = [i for i, r in enumerate(rows) if r["pickup_measurement"]["delta_limiter_activated"]]
        if not hits:
            continue
        i = hits[0]; first.append(i)
        cmd = act_to_physical_normalized(np.asarray(rows[i]["requested_act"], np.float32))
        cur = act_to_physical_normalized(np.asarray(rows[i]["current_act"], np.float32))
        joint[JN[int(np.argmax(np.abs(cmd - cur)))]] += 1
        phase["survey 0-89" if i < 90 else "descent 90-179" if i < 180 else "grasp 180-269" if i < 270 else "carry/place 270+"] += 1
    print(f"{pol:22s} rollouts with holds {len(first):2d}  first-hold step median {np.median(first):.0f}  phase {dict(phase)}  joint {dict(joint)}")
