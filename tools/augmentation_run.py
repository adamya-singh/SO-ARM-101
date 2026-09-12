"""Random-shift augmentation run on the scaling ladder's captures (2026-09-12, lever 1 after the ladder).

Why (notes/vision-rung-notebook.md, "Scaling ladder"): placements, encoder size
and steps all left the held-out loss at chunk start 90 (survey frame -> first
descent chunk) at 2-7e-3 while a nearest-training-placement lookup scores
4e-4, so the network memorises pixel-exact images instead of reading where
the square is. Random shift (pad by ``s`` pixels, crop back at a per-sample
random offset; DrQ) removes the fixed pixel identity of every training frame,
so the only fit left is a function of the square's position relative to the
scene. This tool trains the ladder's sweep point (1200 placements, v2, 120k
steps, stride 18) with a few shift magnitudes, re-uses the ladder's train and
held-out captures (no new frames), and judges each point by what the ladder
said to watch: per-pose held-out start-90 loss (median, count of poses under
2.5e-4) and 30 closed-loop rollouts on the held-out suite. The ladder's
unaugmented point is scored the same way as the baseline.

W&B (project so-arm101-v2-scaling, group augmentation): sections in the
repository's order, x-axis = shift in pixels: 01_outcome (held-out success
rate, safety frames), 02_generalisation (held-out / train start-90, ratio,
per-pose median / max / count under 2.5e-4, the lookup baseline as a flat
line, one line per pose), 03_training (batch loss per shift, final train
mse), 04_throughput (train seconds, steps/s).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from so_arm101_v2.contracts.bench import scene_dependency_hash  # noqa: E402
from so_arm101_v2.data._serialization import write_immutable_json  # noqa: E402
from so_arm101_v2.simulation.policy_specs import PolicySpec  # noqa: E402
from so_arm101_v2.simulation.rollout import evaluate_closed_loop  # noqa: E402
from so_arm101_v2.simulation.suites import load_suite_from_path  # noqa: E402
from scaling_ladder import DEFAULT_MODEL, atomic_json  # noqa: E402

DEFAULT_LADDER = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/scaling_ladder_20260910"
# Nearest-training-placement lookup on the 1200-placement prefix (analysis_lookup_baseline.txt, 2026-09-11):
# the level a policy that merely retrieved its nearest training placement would reach.
LOOKUP_BASELINE_1200 = dict(mean=5.95e-4, median=3.67e-4)
POSE_THRESHOLD = 2.5e-4


def score(checkpoint_path: Path, encoder: str, train_manifest: Path, heldout_manifest: Path, episodes: int):
    import torch
    from heldout_fit import boundary_losses, per_pose_start_90, per_pose_summary
    from so_arm101_v2.learning.vision import build_vision_chunked_model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_vision_chunked_model(int(checkpoint["hidden_width"]), int(checkpoint["chunk_horizon"]), encoder=encoder)
    model.load_state_dict(checkpoint["state_dict"]); model.chunk_horizon = int(checkpoint["chunk_horizon"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval().to(device)
    train = boundary_losses(model, train_manifest, device=device, episode_limit=episodes)
    heldout = boundary_losses(model, heldout_manifest, device=device)
    per_pose = per_pose_start_90(model, heldout_manifest, device=device)
    parameters = int(sum(q.numel() for q in model.parameters()))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(parameters=parameters, train=train, heldout=heldout, heldout_start_90=heldout["start_90"],
                ratio_start_90=round(heldout["start_90"] / max(train["start_90"], 1e-12), 1),
                per_pose=per_pose, per_pose_summary=per_pose_summary(per_pose, POSE_THRESHOLD))


def rollouts(model_xml, suite, checkpoint_path: Path, directory: Path, workers: int):
    spec = PolicySpec(kind="vision_chunked", checkpoint=str(checkpoint_path), options=(("clamp_channels", (5,)), ("black_image", False)))
    result = evaluate_closed_loop(model_xml, suite, {"vision": spec}, directory, environment_proven=True, record_video=False, workers=workers)
    report = json.loads(Path(result.report_json).read_text())
    rows = [r for r in report["rollouts"] if r["policy_id"] == "vision"]
    per_scenario: dict[str, int] = {}
    for r in rows:
        per_scenario[r["scenario_id"]] = per_scenario.get(r["scenario_id"], 0) + int(r["success"] and not r["invalidated"])
    return dict(successes=sum(r["success"] and not r["invalidated"] for r in rows), rollouts=len(rows),
                safety_frames=sum(sum(r[k] for k in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")) for r in rows),
                per_scenario=per_scenario, report=str(result.report_json))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER, help="scaling ladder directory whose captures and baseline point are reused")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shifts", type=int, nargs="+", default=[4, 12], help="random-shift magnitudes in pixels (256-px frames)")
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--encoder", default="v2")
    p.add_argument("--hidden-width", type=int, default=512)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--frame-stride", type=int, default=18)
    p.add_argument("--baseline-label", default="d1200_v2_s120000", help="ladder point trained without augmentation at the same settings")
    p.add_argument("--no-rollouts", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rehearsal", action="store_true", help="toy settings under a rehearsal/ directory; nothing may be quoted from it")
    args = p.parse_args(argv)
    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    root = args.output_dir.resolve()
    if args.rehearsal and "rehearsal" not in root.parts:
        raise RuntimeError("--rehearsal output must live under a directory named 'rehearsal'")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["SO_ARM101_V2_FRAME_CACHE"] = "gpu"
    ladder = args.ladder_dir.resolve()
    train_manifest = Path((ladder / "train_manifest.path").read_text().strip())
    heldout_manifest = Path((ladder / "heldout_manifest.path").read_text().strip())
    heldout_suite = load_suite_from_path(Path((ladder / "heldout_suite.path").read_text().strip()))
    ladder_points = {pt["label"]: pt for pt in json.loads((ladder / "ladder.json").read_text())["points"]}
    baseline = ladder_points.get(args.baseline_label)
    if baseline is None and not args.rehearsal:
        raise RuntimeError(f"baseline point {args.baseline_label} not found in {ladder / 'ladder.json'}")
    scene_hash = scene_dependency_hash(args.model)
    identity = dict(scene_dependencies_sha256=scene_hash, ladder_dir=str(ladder), train_manifest=str(train_manifest), heldout_manifest=str(heldout_manifest),
                    shifts=args.shifts, episodes=args.episodes, encoder=args.encoder, hidden_width=args.hidden_width, steps=args.steps,
                    frame_stride=args.frame_stride, baseline_label=args.baseline_label, training_seed=202, rehearsal=bool(args.rehearsal),
                    lookup_baseline_1200=LOOKUP_BASELINE_1200, pose_threshold=POSE_THRESHOLD)
    write_immutable_json(root / "augmentation_identity.json", identity)
    state = dict(status="running", phase="start", started_at=time.time(), points=[])
    done = {}
    previous = root / "progress.json"
    if previous.exists():
        for point in json.loads(previous.read_text()).get("points", []):
            done[point["label"]] = point
        if done:
            print(f"resuming: {len(done)} completed point(s) kept", flush=True)
    import wandb
    tracker = wandb.init(project="so-arm101-v2-scaling", group="augmentation", name=("augmentation-rehearsal" if args.rehearsal else f"random-shift-{scene_hash[:8]}"),
                         mode=("offline" if args.rehearsal else "online"), config=identity, settings=wandb.Settings(init_timeout=120))
    # Panel order: numbered sections sort first-things-first in the workspace; every summary chart is plotted against the shift.
    tracker.define_metric("augmentation/shift_px")
    tracker.define_metric("03_training/step")
    for section in ("01_outcome/*", "02_generalisation/*", "04_throughput/*"):
        tracker.define_metric(section, step_metric="augmentation/shift_px")
    tracker.define_metric("03_training/batch_mse_*", step_metric="03_training/step")

    def progress(**values):
        state.update(values); state["updated_at"] = time.time(); atomic_json(root / "progress.json", state)

    def log_point(point):
        shift = point["shift"]
        payload = {"augmentation/shift_px": shift,
                   "02_generalisation/heldout_start_90": point["heldout_start_90"], "02_generalisation/train_start_90": point["train"]["start_90"],
                   "02_generalisation/ratio_start_90": point["ratio_start_90"],
                   "02_generalisation/per_pose_median": point["per_pose_summary"]["median"], "02_generalisation/per_pose_max": point["per_pose_summary"]["max"],
                   "02_generalisation/poses_under_2.5e-4": point["per_pose_summary"]["poses_under_threshold"],
                   "02_generalisation/lookup_baseline_mean": LOOKUP_BASELINE_1200["mean"], "02_generalisation/lookup_baseline_median": LOOKUP_BASELINE_1200["median"],
                   "02_generalisation/heldout_all_rows": point["heldout"]["all_rows"], "02_generalisation/heldout_start_0": point["heldout"]["start_0"],
                   "03_training/final_train_mse": point["normalized_mse"], "04_throughput/train_seconds": point["train_seconds"],
                   "04_throughput/steps_per_s": args.steps / max(point["train_seconds"], 1e-9)}
        payload.update({f"02_generalisation/pose/{pose}": value for pose, value in point["per_pose"].items()})
        if "heldout_success" in point:
            hs = point["heldout_success"]
            payload["01_outcome/heldout_success_rate"] = hs["successes"] / max(hs["rollouts"], 1)
            payload["01_outcome/heldout_successes"] = hs["successes"]; payload["01_outcome/safety_frames"] = hs["safety_frames"]
        try:
            tracker.log(payload)
        except Exception as exc:   # tracking must never stop the run
            print(f"wandb log failed: {exc}", flush=True)

    def announce(point):
        s = point["per_pose_summary"]
        print(f"AUG shift {point['shift']}: train90 {point['train']['start_90']:.2e} heldout90 {point['heldout_start_90']:.2e} ratio {point['ratio_start_90']}x "
              f"per-pose median {s['median']:.1e} max {s['max']:.1e} under {POSE_THRESHOLD:.1e}: {s['poses_under_threshold']}/{s['poses']}"
              + (f" success {point['heldout_success']['successes']}/{point['heldout_success']['rollouts']}" if "heldout_success" in point else ""), flush=True)

    try:
        progress()
        from so_arm101_v2.learning.vision import VisionChunkedConfig, train_vision_chunked
        results = []
        plan = ([("baseline", 0)] if baseline is not None else []) + [(f"shift{s}", s) for s in args.shifts]
        for index, (label, shift) in enumerate(plan):
            if label in done:
                results.append(done[label]); state["points"] = results; progress(phase=f"kept {label}"); continue
            if shift == 0:
                progress(phase="score baseline")
                checkpoint = Path(baseline["checkpoint"])
                point = dict(label=label, shift=0, run_digest=baseline["run_digest"], checkpoint=str(checkpoint), train_seconds=baseline["train_seconds"],
                             normalized_mse=baseline["normalized_mse"], source=f"ladder point {args.baseline_label}")
            else:
                progress(phase=f"train {label} ({index + 1}/{len(plan)})")
                config = VisionChunkedConfig(seed=202, max_steps=args.steps, hidden_width=args.hidden_width, frame_stride=args.frame_stride,
                                             encoder=args.encoder, episode_limit=args.episodes, random_shift=shift)

                def on_loss(step, value, _shift=shift):
                    try:
                        tracker.log({"03_training/step": step, f"03_training/batch_mse_shift{_shift}": value})
                    except Exception:
                        pass
                started = time.time()
                trained = train_vision_chunked(train_manifest, root / "runs" / label, config=config, scratch_checkpoint=root / "runs" / label / "scratch.pt",
                                               checkpoint_interval=5000, on_loss=on_loss)
                checkpoint = trained.checkpoint
                point = dict(label=label, shift=shift, run_digest=json.loads(trained.report_json.read_text())["run_digest"], checkpoint=str(checkpoint),
                             train_seconds=round(time.time() - started, 1), normalized_mse=trained.normalized_mse)
            progress(phase=f"score {label}")
            point.update(score(checkpoint, args.encoder, train_manifest, heldout_manifest, args.episodes))
            if not args.no_rollouts:
                progress(phase=f"evaluate {label}")
                if shift == 0 and "heldout_success" in baseline:
                    point["heldout_success"] = baseline["heldout_success"]
                else:
                    point["heldout_success"] = rollouts(args.model, heldout_suite, checkpoint, root / "evaluations" / label, args.workers)
            results.append(point); state["points"] = results
            log_point(point); announce(point)
            atomic_json(root / "augmentation.json", dict(identity=identity, points=results))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            shifts = [r["shift"] for r in results]
            axes[0].plot(shifts, [r["heldout_start_90"] for r in results], "o-", label="held-out start 90 (mean)")
            axes[0].plot(shifts, [r["per_pose_summary"]["median"] for r in results], "s-", label="held-out start 90 (per-pose median)")
            axes[0].plot(shifts, [r["train"]["start_90"] for r in results], "x--", alpha=.6, label="train start 90")
            axes[0].axhline(LOOKUP_BASELINE_1200["median"], color="k", ls=":", label="lookup baseline (median)")
            axes[0].axhline(POSE_THRESHOLD, color="g", ls=":", label="rollouts succeed below")
            axes[0].set_yscale("log"); axes[0].set_xlabel("random shift (px)"); axes[0].set_ylabel("chunk-target MSE"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
            poses = list(results[0]["per_pose"].keys()); x = np.arange(len(poses)); width = 0.8 / max(len(results), 1)
            for k, r in enumerate(results):
                axes[1].bar(x + k * width, [r["per_pose"][pose] for pose in poses], width, label=f"shift {r['shift']}" + (f" ({r['heldout_success']['successes']}/{r['heldout_success']['rollouts']})" if "heldout_success" in r else ""))
            axes[1].axhline(POSE_THRESHOLD, color="g", ls=":"); axes[1].set_yscale("log"); axes[1].set_xticks(x + 0.4 - width / 2); axes[1].set_xticklabels(poses, rotation=45, fontsize=7)
            axes[1].set_ylabel("held-out start-90 MSE per pose"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3, axis="y")
            fig.tight_layout(); fig.savefig(root / "augmentation.png", dpi=110)
            try:
                tracker.log({"01_outcome/summary_figure": wandb.Image(str(root / "augmentation.png"))})
            except Exception:
                pass
        except ImportError:
            pass
        best = min(results, key=lambda r: r["per_pose_summary"]["median"])
        tracker.summary.update({"best/shift": best["shift"], "best/per_pose_median": best["per_pose_summary"]["median"],
                                "best/poses_under_2.5e-4": best["per_pose_summary"]["poses_under_threshold"]})
        tracker.finish()
        progress(status="complete", phase="complete")
        return 0
    except BaseException as exc:
        import traceback
        progress(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
