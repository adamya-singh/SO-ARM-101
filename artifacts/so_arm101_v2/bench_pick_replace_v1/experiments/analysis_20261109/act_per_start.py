"""Held-out loss of each ACT at every exec30 chunk start (normalised MSE over the 90-step residual chunk).

The question: does the perturbed-only ACT fail at the chunk starts its training never contained (120, 150)?
Held-out = the clean SO-101 teacher on the 10 held-out poses (full frames). CPU, 2026-11-09.
"""
import glob, json, sys
import numpy as np
from so_arm101_v2.learning.pi05 import CaptureSource
from so_arm101_v2.learning.vla_models import load_model
E = "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
HELD = glob.glob(f"{E}/so101_benchmark_20261106/so101/cap4800/capture_heldout/oracle/*/*/manifest.json")[0]
MODELS = {
    "base_act (clean 4800)": f"{E}/so101_benchmark_20261106/so101/act_ft/model/act.pt",
    "pert_act (perturbed 7200)": f"{E}/perturbed_so101_20261108/act_ft/model/act.pt",
    "physv2_act": f"{E}/physics_v2_20261108/act_ft/model/act.pt",
}
STARTS = list(range(0, 331, 30))
src = CaptureSource.load(HELD, name="heldout")
out = {}
print("model".ljust(28) + "".join(str(s).rjust(9) for s in STARTS))
for name, path in MODELS.items():
    model = load_model(path, device="cpu")
    row = []
    for s in STARTS:
        rows = src.rows_at(s)
        pred = model.sample(src.images(rows), src.state(rows), seed=0)
        row.append(float(((pred - src.targets(rows, 90)) ** 2).mean()))
    out[name] = dict(zip(map(str, STARTS), row))
    print(name.ljust(28) + "".join(f"{v:9.1e}" for v in row), flush=True)
json.dump({"heldout_manifest": HELD, "starts": STARTS, "mse": out},
          open(f"{E}/analysis_20261109/act_per_start.json", "w"), indent=1)
