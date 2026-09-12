"""Scaling ladder for the bench vision policy: unique placements x encoder x steps, scored by held-out loss (2026-09-10).

Why (see notes/vision-rung-notebook.md, "Analysis of run 4"): the placement
policy memorised its 400 training placements (held-out loss at the first
descent chunk 136x the training loss). Chinchilla-style compute-optimal
laws assume single-epoch, data-unlimited training; our regime is the
opposite (data is manufactured by a CPU-bound teacher, training is minutes
on the GPU, and we train for ~100 epochs), so the useful question is how the
held-out loss falls with unique scenes for each encoder, where repeated
epochs stop paying, and what the irreducible floor is. This tool answers
those with one capture and a grid of cheap trainings.

Design, kept GPU-frugal:
- ONE screened capture of ``--capture-count`` placements (CPU); data sizes are
  prefixes of it (``episode_limit``: accepted placements are in draw order,
  so a prefix is a random subset). No frames are captured twice.
- Every training uses the GPU frame store (SO_ARM101_V2_FRAME_CACHE=gpu) with
  ``--frame-stride`` (9 by default: only chunk starts at multiples of 90 occur
  at inference, all multiples of 9), so a run costs minutes.
- Every point is scored with ``tools/heldout_fit.py`` (seconds): train and
  held-out loss at chunk start 0, 90, later boundaries and all rows, on a
  held-out capture of the seed-8 suite made once.
- Closed-loop rollouts (the expensive ground truth) only for the frontier:
  the largest data size of each encoder, plus the steps sweep's endpoints.
- Fits loss = E + A/N^alpha + B/D^beta on the held-out start-90 loss when
  enough points exist, and writes ladder.json + a PNG.

Rehearsal: ``--rehearsal`` runs the whole chain at toy scale under a
rehearsal/ directory (no result may be quoted from it).
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

from so_arm101_v2.contracts.bench import scene_bench_config, scene_dependency_hash  # noqa: E402
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json  # noqa: E402
from so_arm101_v2.simulation.bench import generate_bench_suite  # noqa: E402
from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations  # noqa: E402
from so_arm101_v2.simulation.policy_specs import PolicySpec  # noqa: E402
from so_arm101_v2.simulation.rollout import evaluate_closed_loop, run_simulation_preflight  # noqa: E402
from so_arm101_v2.simulation.suites import load_suite_from_path  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temp, path)


def preflight(model, root: Path, suite, workers):
    path = root / "simulation" / "preflight" / suite.suite_id / "evaluation.json"
    if not path.exists():
        path = Path(run_simulation_preflight(model, root / "simulation", suite=suite, record_video=False, workers=workers).report_json)
    report = json.loads(path.read_text())
    if not report.get("environment_proven") or not report.get("deterministic"):
        raise RuntimeError(f"oracle gate failed: {path}")
    return path


def get_or_generate_suite(model, bench, root: Path, label: str, *, seed: int, count: int, repeats: int, workers):
    pointer = root / f"{label}_suite.path"
    if pointer.exists():
        path = Path(pointer.read_text().strip())
        return load_suite_from_path(path), path
    suite, path = generate_bench_suite(model, bench, seed=seed, count=count, repeats=repeats, output_dir=root / "suites",
                                       randomize_appearance=bench.appearance is not None, randomize_placement=True)
    pointer.write_text(str(path.resolve()) + "\n")
    return suite, path


def get_or_capture(model, root: Path, label: str, suite, preflight_path, *, workers, horizon=480, frame_row_stride=1):
    pointer = root / f"{label}_manifest.path"
    if pointer.exists():
        return Path(pointer.read_text().strip())
    capture = capture_oracle_demonstrations(model, suite, preflight_path, root / f"capture_{label}", scenario="all", record_video=False,
                                            teacher_horizon=horizon, store_frames=True, workers=workers, frame_row_stride=frame_row_stride)
    pointer.write_text(str(Path(capture.manifest).resolve()) + "\n")
    return Path(capture.manifest)


def fit_power_law(points):
    """loss = E + A * N^-alpha + B * D^-beta by least squares in log space over a coarse grid; returns dict or None."""
    if len(points) < 6:
        return None
    N = np.array([p["parameters"] for p in points], float); D = np.array([p["episodes"] for p in points], float); L = np.array([p["heldout_start_90"] for p in points], float)
    best = None
    for E in np.geomspace(1e-7, L.min() * 0.999, 25):
        for alpha in np.linspace(0.05, 1.5, 30):
            for beta in np.linspace(0.05, 1.5, 30):
                X = np.stack([N ** -alpha, D ** -beta], 1)
                coef, *_ = np.linalg.lstsq(X, L - E, rcond=None)
                if np.any(coef < 0):
                    continue
                pred = E + X @ coef
                err = float(np.mean((np.log(pred) - np.log(L)) ** 2))
                if best is None or err < best["log_error"]:
                    best = dict(E=float(E), A=float(coef[0]), alpha=float(alpha), B=float(coef[1]), beta=float(beta), log_error=err)
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--capture-count", type=int, default=2400)
    p.add_argument("--data-sizes", type=int, nargs="+", default=[300, 600, 1200, 2400])
    p.add_argument("--encoders", nargs="+", default=["v1", "v2", "v3"])
    p.add_argument("--hidden-width", type=int, default=512)
    p.add_argument("--steps", type=int, default=120000, help="optimizer steps for the data x encoder grid")
    p.add_argument("--steps-sweep", type=int, nargs="*", default=[60000, 240000, 480000], help="extra step budgets at the sweep point")
    p.add_argument("--sweep-point", nargs=2, metavar=("DATA", "ENCODER"), default=["1200", "v2"])
    p.add_argument("--frame-stride", type=int, default=9)
    p.add_argument("--heldout-seed", type=int, default=8)
    p.add_argument("--heldout-count", type=int, default=10)
    p.add_argument("--eval-frontier-only", action="store_true", default=True)
    p.add_argument("--eval-all", action="store_true", help="closed-loop rollouts for every point (expensive)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rehearsal", action="store_true")
    args = p.parse_args(argv)
    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    bench = scene_bench_config(args.model)
    if bench is None or bench.placement is None:
        raise RuntimeError("the scaling ladder needs a bench scene with a placement regime")
    root = args.output_dir.resolve()
    if args.rehearsal:
        if "rehearsal" not in root.parts:
            raise RuntimeError("--rehearsal output must live under a directory named 'rehearsal'")
        args.capture_count, args.data_sizes, args.encoders, args.steps, args.steps_sweep = 4, [2, 4], ["v1", "v2"], 6, [3, 12]
        args.sweep_point, args.heldout_count = ["4", "v2"], 2
    root.mkdir(parents=True, exist_ok=True)
    os.environ["SO_ARM101_V2_FRAME_CACHE"] = "gpu"
    scene_hash = scene_dependency_hash(args.model)
    identity = dict(scene_dependencies_sha256=scene_hash, bench=bench.__dict__, capture_count=args.capture_count, data_sizes=args.data_sizes,
                    encoders=args.encoders, hidden_width=args.hidden_width, steps=args.steps, steps_sweep=args.steps_sweep,
                    sweep_point=args.sweep_point, frame_stride=args.frame_stride, heldout=dict(seed=args.heldout_seed, count=args.heldout_count),
                    rehearsal=bool(args.rehearsal), training_seed=202)
    write_immutable_json(root / "ladder_identity.json", identity)
    state = dict(status="running", phase="suites", started_at=time.time(), points=[])
    # Resume (GPU-frugal): points already recorded by an earlier attempt in this directory are kept, not retrained
    # or re-evaluated (their evaluation directories are immutable; EGL raster jitter can make a repeat differ).
    previous = root / "progress.json"
    done_points = {}
    if previous.exists():
        for point in json.loads(previous.read_text()).get("points", []):
            done_points[point["label"]] = point
        if done_points:
            print(f"resuming: {len(done_points)} completed point(s) kept from {previous}", flush=True)
    import wandb
    wandb_mode = "offline" if args.rehearsal else "online"
    tracker = wandb.init(project="so-arm101-v2-scaling", group="scaling-ladder", name=("ladder-rehearsal" if args.rehearsal else f"scaling-ladder-{scene_hash[:8]}"),
                         mode=wandb_mode, config=identity, settings=wandb.Settings(init_timeout=120))
    # Panel order: sections sort alphabetically in W&B, so the numbered prefixes put the decisive charts first.
    tracker.define_metric("ladder/episodes"); tracker.define_metric("ladder/steps")
    for section in ("01_outcome/*", "02_generalisation/*"):
        tracker.define_metric(section, step_metric="ladder/point")

    def progress(**values):
        state.update(values); state["updated_at"] = time.time(); atomic_json(root / "progress.json", state)

    try:
        progress()
        train_suite, _ = get_or_generate_suite(args.model, bench, root, "train", seed=12, count=args.capture_count, repeats=1, workers=args.workers)
        heldout_suite, _ = get_or_generate_suite(args.model, bench, root, "heldout", seed=args.heldout_seed, count=args.heldout_count, repeats=3, workers=args.workers)
        train_pre = preflight(args.model, root, train_suite, args.workers)
        heldout_pre = preflight(args.model, root, heldout_suite, args.workers)
        progress(phase="capture")
        train_manifest = get_or_capture(args.model, root, "train", train_suite, train_pre, workers=args.workers)
        heldout_manifest = get_or_capture(args.model, root, "heldout", heldout_suite, heldout_pre, workers=args.workers)
        from so_arm101_v2.learning.vision import VisionChunkedConfig, build_vision_chunked_model, train_vision_chunked
        from heldout_fit import boundary_losses
        import torch
        grid = [(d, e, args.steps) for e in args.encoders for d in args.data_sizes]
        sweep_data, sweep_encoder = int(args.sweep_point[0]), str(args.sweep_point[1])
        grid += [(sweep_data, sweep_encoder, s) for s in args.steps_sweep]
        frontier = {(max(args.data_sizes), e, args.steps) for e in args.encoders} | {(sweep_data, sweep_encoder, s) for s in args.steps_sweep}
        results = []
        for index, (data, encoder, steps) in enumerate(grid):
            label = f"d{data}_{encoder}_s{steps}"
            if label in done_points:
                results.append(done_points[label]); state["points"] = results; progress(phase=f"kept {label} ({index + 1}/{len(grid)})")
                continue
            progress(phase=f"train {label} ({index + 1}/{len(grid)})")
            config = VisionChunkedConfig(seed=202, max_steps=steps, hidden_width=args.hidden_width, frame_stride=args.frame_stride,
                                         encoder=encoder, episode_limit=data)
            started = time.time()
            trained = train_vision_chunked(train_manifest, root / "runs" / label, config=config, scratch_checkpoint=root / "runs" / label / "scratch.pt",
                                           checkpoint_interval=5000)
            train_seconds = time.time() - started
            checkpoint = torch.load(trained.checkpoint, map_location="cpu", weights_only=False)
            model = build_vision_chunked_model(int(checkpoint["hidden_width"]), int(checkpoint["chunk_horizon"]), encoder=encoder)
            model.load_state_dict(checkpoint["state_dict"]); model.chunk_horizon = int(checkpoint["chunk_horizon"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval().to(device)
            # Train loss on the SAME prefix the point trained on: score the prefix rows only.
            fit_train = boundary_losses(model, train_manifest, device=device, episode_limit=data)
            fit_heldout = boundary_losses(model, heldout_manifest, device=device)
            point = dict(label=label, episodes=data, encoder=encoder, steps=steps, parameters=int(sum(q.numel() for q in model.parameters())),
                         run_digest=json.loads(trained.report_json.read_text())["run_digest"], checkpoint=str(trained.checkpoint),
                         train_seconds=round(train_seconds, 1), normalized_mse=trained.normalized_mse,
                         train=fit_train, heldout=fit_heldout, heldout_start_90=fit_heldout["start_90"],
                         ratio_start_90=round(fit_heldout["start_90"] / max(fit_train["start_90"], 1e-12), 1))
            del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if args.eval_all or (data, encoder, steps) in frontier:
                progress(phase=f"evaluate {label}")
                spec = PolicySpec(kind="vision_chunked", checkpoint=str(trained.checkpoint), options=(("clamp_channels", (5,)), ("black_image", False)))
                result = evaluate_closed_loop(args.model, heldout_suite, {"vision": spec}, root / "evaluations" / label, environment_proven=True,
                                              record_video=False, workers=args.workers)
                report = json.loads(Path(result.report_json).read_text())
                rows = [r for r in report["rollouts"] if r["policy_id"] == "vision"]
                point["heldout_success"] = dict(successes=sum(r["success"] and not r["invalidated"] for r in rows), rollouts=len(rows),
                                                safety_frames=sum(sum(r[k] for k in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")) for r in rows),
                                                report=str(result.report_json))
            results.append(point)
            try:
                payload = {"ladder/point": index, "ladder/episodes": data, "ladder/steps": steps, "ladder/parameters": point["parameters"],
                           f"02_generalisation/heldout_start_90_{encoder}": fit_heldout["start_90"], f"02_generalisation/train_start_90_{encoder}": fit_train["start_90"],
                           f"02_generalisation/ratio_start_90_{encoder}": point["ratio_start_90"], "03_training/normalized_mse": trained.normalized_mse,
                           "04_throughput/train_seconds": train_seconds}
                if "heldout_success" in point:
                    payload[f"01_outcome/heldout_success_rate_{encoder}"] = point["heldout_success"]["successes"] / max(point["heldout_success"]["rollouts"], 1)
                tracker.log(payload, step=index)
            except Exception as exc:   # tracking must never stop the ladder
                print(f"wandb log failed: {exc}", flush=True)
            state["points"] = results
            atomic_json(root / "ladder.json", dict(identity=identity, points=results))
            print(f"LADDER {label}: params {point['parameters']} train90 {fit_train['start_90']:.2e} heldout90 {fit_heldout['start_90']:.2e} "
                  f"ratio {point['ratio_start_90']}x" + (f" success {point['heldout_success']['successes']}/{point['heldout_success']['rollouts']}" if "heldout_success" in point else ""), flush=True)
        fit = fit_power_law([r for r in results if r["steps"] == args.steps])
        summary = dict(identity=identity, points=results, power_law_fit=fit)
        atomic_json(root / "ladder.json", summary)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for encoder in args.encoders:
                pts = sorted([r for r in results if r["encoder"] == encoder and r["steps"] == args.steps], key=lambda r: r["episodes"])
                axes[0].plot([r["episodes"] for r in pts], [r["heldout_start_90"] for r in pts], "o-", label=f"{encoder} held-out")
                axes[0].plot([r["episodes"] for r in pts], [r["train"]["start_90"] for r in pts], "x--", alpha=0.5, label=f"{encoder} train")
            axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("unique placements"); axes[0].set_ylabel("loss at chunk start 90"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
            sweep = sorted([r for r in results if r["episodes"] == sweep_data and r["encoder"] == sweep_encoder], key=lambda r: r["steps"])
            axes[1].plot([r["steps"] for r in sweep], [r["heldout_start_90"] for r in sweep], "o-", label="held-out")
            axes[1].plot([r["steps"] for r in sweep], [r["train"]["start_90"] for r in sweep], "x--", label="train")
            axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel(f"optimizer steps ({sweep_data} placements, {sweep_encoder})"); axes[1].legend(); axes[1].grid(alpha=.3)
            fig.tight_layout(); fig.savefig(root / "ladder.png", dpi=110)
        except ImportError:
            pass
        if fit is not None:
            tracker.summary.update({f"power_law/{k}": v for k, v in fit.items()})
        tracker.finish()
        progress(status="complete", phase="complete")
        print(json.dumps(dict(power_law_fit=fit), indent=2))
        return 0
    except BaseException as exc:
        import traceback
        progress(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
