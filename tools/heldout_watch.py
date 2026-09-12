"""Score a training run's scratch checkpoints on the held-out capture while it trains (2026-09-12).

For runs that started before the trainer's ``on_checkpoint`` observer existed
(or any run launched without it): polls ``--scratch`` (written atomically every
checkpoint_interval steps), scores each new checkpoint with
``HeldoutCurveScorer`` (held-out start-90 mean, per-pose median, poses under
2.5e-4, a fixed train sample and the ratio), appends to ``heldout_curve.jsonl``
next to the scratch file and logs to its own W&B run (same project and group,
name ``<run name>-heldout-curve``; a second process must not write into the
training process's run). Exits when ``--progress`` reports complete/failed or
the scratch file has not changed for ``--idle-minutes``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--heldout-manifest", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--episodes", type=int, required=True, help="training prefix (episode_limit) of the run")
    p.add_argument("--encoder", default="v2")
    p.add_argument("--hidden-width", type=int, default=512)
    p.add_argument("--shift", type=int, required=True, help="random shift of the run (metric suffix only)")
    p.add_argument("--progress", type=Path, default=None, help="progress.json of the run; exit on complete/failed")
    p.add_argument("--wandb-name", required=True)
    p.add_argument("--wandb-group", default="augmentation")
    p.add_argument("--idle-minutes", type=float, default=30.0)
    p.add_argument("--poll-seconds", type=float, default=20.0)
    args = p.parse_args(argv)
    import torch
    import wandb
    from heldout_fit import HeldoutCurveScorer
    from so_arm101_v2.learning.vision import build_vision_chunked_model
    scorer = HeldoutCurveScorer(args.heldout_manifest, args.train_manifest, episode_limit=args.episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vision_chunked_model(args.hidden_width, 90, encoder=args.encoder).to(device).eval()
    tracker = wandb.init(project="so-arm101-v2-scaling", group=args.wandb_group, name=args.wandb_name,
                         config=dict(scratch=str(args.scratch), episodes=args.episodes, encoder=args.encoder, shift=args.shift, watcher=True),
                         settings=wandb.Settings(init_timeout=120))
    tracker.define_metric("03_training/step")
    tracker.define_metric("02_generalisation/curve_*", step_metric="03_training/step")
    curve_path = args.scratch.with_name("heldout_curve.jsonl")
    last_mtime, last_step, last_change = None, None, time.time()
    print(f"watching {args.scratch} -> {curve_path}; W&B {tracker.url}", flush=True)
    while True:
        if args.progress is not None and args.progress.exists():
            status = json.loads(args.progress.read_text()).get("status")
            if status in ("complete", "failed"):
                print(f"run {status}; watcher exiting", flush=True); break
        if args.scratch.exists():
            mtime = os.path.getmtime(args.scratch)
            if mtime != last_mtime:
                try:
                    payload = torch.load(args.scratch, map_location="cpu", weights_only=False)
                except Exception as exc:   # mid-replace or partial read; try again next poll
                    print(f"load failed ({exc}); retrying", flush=True); time.sleep(args.poll_seconds); continue
                last_mtime, last_change = mtime, time.time()
                step = int(payload["step"])
                if step != last_step:
                    model.load_state_dict(payload["model"])
                    curve = scorer.score(model, device=device)
                    tracker.log({"03_training/step": step,
                                 f"02_generalisation/curve_heldout_start_90_shift{args.shift}": curve["heldout_start_90"],
                                 f"02_generalisation/curve_train_start_90_shift{args.shift}": curve["train_start_90"],
                                 f"02_generalisation/curve_ratio_start_90_shift{args.shift}": curve["ratio_start_90"],
                                 f"02_generalisation/curve_per_pose_median_shift{args.shift}": curve["median"],
                                 f"02_generalisation/curve_poses_under_2.5e-4_shift{args.shift}": curve["poses_under_threshold"]})
                    with open(curve_path, "a") as handle:
                        handle.write(json.dumps(dict(step=step, **curve)) + "\n")
                    print(f"step {step}: heldout90 {curve['heldout_start_90']:.2e} train90 {curve['train_start_90']:.2e} ratio {curve['ratio_start_90']:.0f}x "
                          f"median {curve['median']:.1e} under {curve['poses_under_threshold']}/{curve['poses']}", flush=True)
                    last_step = step
        if time.time() - last_change > args.idle_minutes * 60:
            print("scratch idle; watcher exiting", flush=True); break
        time.sleep(args.poll_seconds)
    tracker.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
