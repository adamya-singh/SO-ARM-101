"""Offline real-frame gate: would this policy hold on a recorded real reset-pose frame? Nothing is sent to the arm.

For each recorded physical episode directory the boundary-0 observation (the
256x256 frame the runner actually fed the network at the reset pose) and its
measured anchor pose are loaded, hash-verified, and run through the dry pass
and the gate (`so_arm101_v2.physical.dry_pass.check_reset_frame`). The same
dry pass on the simulator's nominal reset observation is the reference. With
--perturb N the real frame is also passed through N photometric draws of the
appearance regime (a robustness sweep, reported, not gated).

Exit 2 when any recorded frame fails the gate; the pipeline records the same
result without raising. Calibration (2026-09-09): the lens policy fails this
gate on physical/episode_02_20260909 (shoulder 26 units, elbow 23, 34 holds)
while its simulated reset chunk moves both by under one unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts import JOINT_NAMES  # noqa: E402
from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash  # noqa: E402
from so_arm101_v2.physical.dry_pass import check_reset_frame, load_boundary_frame, sim_reference_dry_pass  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DEFAULT_CHECKPOINT = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909/models/vision_h90/5e96018881d4140f/model.pt"
DEFAULT_EPISODES = (ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/physical/episode_02_20260909",)


def check_policy_on_real_frames(policy, model_path, bench, episode_dirs, *, perturb: int = 0, regime=None) -> dict:
    frames = []
    reference = None
    for episode_dir in episode_dirs:
        image, anchor, evidence = load_boundary_frame(episode_dir, step=0)
        if reference is None:
            reference = sim_reference_dry_pass(policy, model_path, anchor, bench)
            reference_check = check_reset_frame(policy, _sim_image(model_path, bench), anchor, bench, label="sim_reference")
        result = check_reset_frame(policy, image, anchor, bench, label=Path(episode_dir).name, reference=reference)
        result["frame"] = evidence
        if perturb > 0:
            from so_arm101_v2.contracts.appearance import AppearanceRegime, apply_photometric, resolve_appearance
            regime = regime or AppearanceRegime()
            passed = []
            for seed in range(int(perturb)):
                params = resolve_appearance(regime, seed).photometric
                perturbed = apply_photometric(image, params, seed=seed, frame_index=0)
                passed.append(bool(check_reset_frame(policy, perturbed, anchor, bench, label=f"perturb_{seed}")["passed"]))
            result["photometric_sweep"] = dict(draws=int(perturb), regime=regime.identity(), passed=int(sum(passed)), pass_fraction=round(sum(passed) / len(passed), 3))
        frames.append(result)
    return dict(
        gate=frames[0]["gate"] if frames else None,
        model=str(model_path), scene_dependencies_sha256=scene_dependency_hash(model_path),
        sim_reference=dict(dry_pass=reference, check=reference_check) if reference is not None else None,
        frames=frames, passed=bool(frames) and all(f["passed"] for f in frames),
    )


def _sim_image(model_path, bench):
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    adapter = MujocoTaskAdapter(model_path)
    try:
        adapter.reset(bench_suite(bench, [(0, 0)], label="reference", repeats=1).scenarios[0])
        return adapter.render_wrist_observation()
    finally:
        adapter.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--episode-dir", type=Path, action="append", default=None, help="recorded physical episode directory (repeatable); default: episode_02_20260909")
    p.add_argument("--perturb", type=int, default=0, help="photometric perturbation draws of the appearance regime on each real frame (report only)")
    p.add_argument("--output", type=Path, default=None, help="write the JSON record here (default: print)")
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    bench = scene_bench_config(args.model)
    if bench is None or bench.lens is None:
        raise SystemExit("the bench scene must carry a lens block")
    policy = VisionChunkedPolicy(args.checkpoint, black_image=False, clamp_channels=(5,))
    episodes = args.episode_dir or list(DEFAULT_EPISODES)
    record = check_policy_on_real_frames(policy, args.model, bench, episodes, perturb=args.perturb, regime=bench.appearance_regime)
    record.update(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), policy_id=policy.policy_id)
    text = json.dumps(record, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    for frame in record["frames"]:
        dry = frame["dry_pass"]
        print(f"{frame['label']}: {'PASS' if frame['passed'] else 'FAIL'} shoulder {dry['max_abs_delta_from_start_units']['shoulder_lift']} "
              f"elbow {dry['max_abs_delta_from_start_units']['elbow_flex']} holds {dry['holds_in_dry_chunk']} "
              f"min shoulder {dry['min_units']['shoulder_lift']} reasons={frame['reasons']}")
        if "photometric_sweep" in frame:
            print(f"  photometric sweep: {frame['photometric_sweep']['passed']}/{frame['photometric_sweep']['draws']} pass")
    ref = record["sim_reference"]["dry_pass"]["max_abs_delta_from_start_units"]
    print(f"sim reference: shoulder {ref['shoulder_lift']} elbow {ref['elbow_flex']} ({'PASS' if record['sim_reference']['check']['passed'] else 'FAIL'})")
    if args.output is None:
        print(text)
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
