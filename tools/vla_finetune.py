"""Fine-tune SmolVLA or ACT on the bench placement captures, judged like every placement run (retry tranche, 2026-10-04).

The pi0.5 recipe (tools/pi05_finetune.py) for the other two pretrained policies of the tranche:
- data: the 4800-placement capture (stride 30) and the 2400-placement ladder capture (stride 18) together;
- random shift 12 px at 256 px; the 90-step residual chunk target for every model, so all of them make the
  same decisions at the same chunk boundaries as the CNN runs and pi0.5;
- the held-out start-90 curve during training (02_generalisation/*) against the CNN-4800 (3.6e-4) and the
  working fixed-square policy (2.5e-6);
- the 30 closed-loop held-out rollouts at execute 90 and 30, gripper clamp on, closure error in mm at step 250.
SmolVLA starts from lerobot/smolvla_base (frozen VLM, trained expert + projections, bf16 autocast); ACT starts
from an ImageNet ResNet18 (lerobot's ACT recipe: VAE, L1 + KL, backbone at 1e-5).

W&B: notes/wandb-standard.md through so_arm101_v2.tracking (group = model, job_type finetune).

``--probe`` measures step time for each micro-batch in ``--probe-micro`` and writes probe.json (Stage E).

Run in the lerobot env:
  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot/bin/python tools/vla_finetune.py --model smolvla --output-dir ...
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pi05_finetune import Batches, atomic_json, lr_factor  # noqa: E402
from so_arm101_v2.data._serialization import write_immutable_json  # noqa: E402

EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DEFAULT_DATA = [EXPERIMENTS / "placements4800_20260912", EXPERIMENTS / "scaling_ladder_20260910"]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PHASES = ("load", "train", "score", "rollouts", "complete")
TRANCHE_TAG = "retry-tranche-20261004"


def closed_loop(model_xml: Path, suite, model, execute: int, directory: Path, *, num_steps=None, tag=None) -> dict:
    """The 30 held-out rollouts of one execution schedule, with closure in mm (the pi0.5 report's fields)."""
    from so_arm101_v2.simulation.closure import closure_summary
    from so_arm101_v2.simulation.residual_policy import ResidualChunkPolicy
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop

    tag = tag or model.kind
    name = f"{tag}_exec{execute}"
    started = time.time()
    result = evaluate_closed_loop(model_xml, suite, {name: lambda: ResidualChunkPolicy(model, execute=execute, num_steps=num_steps, tag=tag)},
                                  directory, environment_proven=True, record_video=False, workers=1)
    report = json.loads(Path(result.report_json).read_text())
    rows = [r for r in report["rollouts"] if r["policy_id"] == name]
    ok = [bool(r["success"] and not r["invalidated"]) for r in rows]
    categories: dict[str, int] = {}
    for r in rows:
        categories[r["failure_category"]] = categories.get(r["failure_category"], 0) + 1
    closure = closure_summary(model_xml, [r["telemetry_path"] for r in rows if r.get("telemetry_path")])
    return dict(execute=execute, successes=int(sum(ok)), rollouts=len(rows),
                safety_frames=int(sum(sum(r[k] for k in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")) for r in rows)),
                pickups=int(sum(bool(r["pickup_completed"]) for r in rows)), categories=categories,
                closure={k: v for k, v in closure.items() if k != "rows"}, closure_rows=closure.get("rows", []),
                rollout_rows=[{k: r[k] for k in ("scenario_id", "repeat", "success", "invalidated", "failure_category", "pickup_completed", "maximum_height_gain_m") if k in r} for r in rows],
                seconds=round(time.time() - started, 1), report=str(result.report_json))


def log_outcome(tracker, r: dict, step: int, axis: str = "03_training/step") -> None:
    """01_outcome keys of one execution schedule (shared with tools/rl_train.py)."""
    e, c = r["execute"], r["closure"]
    payload = {axis: step, f"01_outcome/success_rate_exec{e}": r["successes"] / max(r["rollouts"], 1),
               f"01_outcome/successes_exec{e}": r["successes"], f"01_outcome/pickups_exec{e}": r["pickups"],
               f"01_outcome/safety_frames_exec{e}": r["safety_frames"],
               f"01_outcome/closure_lateral_median_mm_exec{e}": c.get("lateral_median_mm"),
               f"01_outcome/closure_lateral_rms_mm_exec{e}": c.get("lateral_rms_mm"),
               f"01_outcome/closure_within_8mm_exec{e}": c.get("within_tolerance"),
               f"01_outcome/closure_not_descended_exec{e}": c.get("not_descended"),
               f"01_outcome/closure_yaw_err_median_deg_exec{e}": c.get("yaw_err_median_deg")}
    tracker.log({k: v for k, v in payload.items() if v is not None})
    tracker.summary({f"01_outcome/successes_exec{e}": r["successes"], f"01_outcome/success_rate_exec{e}": r["successes"] / max(r["rollouts"], 1)})
    closure_by = {(x["scenario_id"], x["repeat"]): x for x in r["closure_rows"]}
    rows = []
    for row in r["rollout_rows"]:
        cr = closure_by.get((row["scenario_id"], row.get("repeat", 0)), {})
        rows.append([e, row["scenario_id"], row.get("repeat", 0), bool(row["success"] and not row["invalidated"]), row["failure_category"],
                     bool(row["pickup_completed"]), 1000 * float(row.get("maximum_height_gain_m", 0.0)), cr.get("lat"), cr.get("dz"), cr.get("yaw_err")])
    tracker.table(f"01_outcome/rollouts_exec{e}", ["execute", "scenario", "repeat", "success", "failure", "pickup", "max_lift_mm",
                                                   "closure_lat_mm", "closure_dz_mm", "yaw_err_deg"], rows)


def summary_figure(path: Path, label: str, curve_path: Path, final: dict | None, outcomes: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from so_arm101_v2.tracking import REFERENCES

    curve = [json.loads(line) for line in curve_path.read_text().splitlines()] if curve_path.exists() else []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    if curve:
        x = [c.get("step", c.get("episodes", 0)) for c in curve]
        ax.plot(x, [c["heldout_start_90"] for c in curve], "o-", label="held-out start 90")
        ax.plot(x, [c["train_start_90"] for c in curve], "x--", alpha=.7, label="train start 90")
    ax.axhline(REFERENCES["cnn4800_heldout_start_90"], color="tab:red", ls=":", label="CNN 4800 held-out (3.6e-4)")
    ax.axhline(REFERENCES["working_policy_start_90"], color="tab:green", ls=":", label="working fixed-square policy (2.5e-6)")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title(f"{label}: held-out start-90 chunk MSE")
    ax = axes[1]
    labels = [f"{label} exec {k}" for k in sorted(outcomes, key=lambda k: -int(k))] + ["CNN 4800", "pi0.5 FT exec 90", "pi0.5 FT exec 30"]
    succ = [outcomes[k]["successes"] for k in sorted(outcomes, key=lambda k: -int(k))] + [REFERENCES["cnn4800_successes"],
                                                                                        REFERENCES["pi05_ft_successes_exec90"], REFERENCES["pi05_ft_successes_exec30"]]
    ax.bar(np.arange(len(labels)), succ, color=["tab:blue"] * len(outcomes) + ["0.6"] * 3)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=15); ax.set_ylabel("successes of 30")
    ax.set_title("closed loop on 10 unseen placements x 3")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None) -> int:
    from so_arm101_v2.learning.vla_models import DEFAULTS, VLAConfig

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=("smolvla", "act"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dirs", type=Path, nargs="+", default=DEFAULT_DATA)
    p.add_argument("--scene", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--micro", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--shift", type=int, default=12)
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--scratch-every", type=int, default=2500)
    p.add_argument("--execs", type=int, nargs="+", default=[90, 30])
    p.add_argument("--rl-steps", type=int, default=None, help="flow models: also run the exec-90 rollouts at this denoising-step count (the K RL will use)")
    p.add_argument("--loader-threads", type=int, default=8)
    p.add_argument("--amp", choices=("auto", "on", "off"), default="auto", help="bf16 autocast (auto: on for SmolVLA, off for ACT)")
    p.add_argument("--rollout-limit", type=int, default=None, help="rehearsal only: first N held-out scenarios, 1 repeat")
    p.add_argument("--run-name", default=None)
    p.add_argument("--rehearsal", action="store_true")
    p.add_argument("--probe", action="store_true", help="measure steps/s for each --probe-micro and exit")
    p.add_argument("--probe-micro", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--probe-steps", type=int, default=12)
    args = p.parse_args(argv)

    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    root = args.output_dir.resolve()
    if args.rehearsal and "rehearsal" not in root.parts:
        raise RuntimeError("--rehearsal output must live under a directory named 'rehearsal'")
    if args.rollout_limit is not None and not args.rehearsal:
        raise RuntimeError("--rollout-limit is a rehearsal-only switch")
    root.mkdir(parents=True, exist_ok=True)

    import torch
    from so_arm101_v2 import tracking
    from so_arm101_v2.contracts.bench import scene_dependency_hash
    from so_arm101_v2.learning.pi05 import CaptureSource, QuantileStats
    from so_arm101_v2.learning.vla_models import HeldoutScorer, MasterWeights, build_model, load_model
    from so_arm101_v2.simulation.suites import load_suite_from_path

    defaults = dict(DEFAULTS[args.model])
    overrides = {k: v for k, v in dict(max_steps=args.steps, batch_size=args.batch, micro_batch=args.micro, learning_rate=args.lr,
                                        warmup_steps=args.warmup).items() if v is not None}
    defaults.update(overrides)
    defaults.setdefault("max_steps", 30000)
    config = VLAConfig(kind=args.model, random_shift=args.shift, seed=args.seed, num_inference_steps=args.num_inference_steps, **defaults)
    amp = args.amp == "on" or (args.amp == "auto" and args.model == "smolvla")
    data_dirs = [d.resolve() for d in args.data_dirs]
    train_manifests = [Path((d / "train_manifest.path").read_text().strip()) for d in data_dirs]
    heldout_manifest = Path((data_dirs[0] / "heldout_manifest.path").read_text().strip())
    suite = load_suite_from_path(Path((data_dirs[0] / "heldout_suite.path").read_text().strip()))
    if args.rollout_limit is not None:
        suite = dataclasses.replace(suite, suite_id=f"{suite.suite_id}_rehearsal{args.rollout_limit}", scenarios=suite.scenarios[: args.rollout_limit], repeats=1)
    scene_hash = scene_dependency_hash(args.scene)
    identity = dict(tool="vla_finetune", model=args.model, config=dataclasses.asdict(config), amp=amp, train_manifests=[str(m) for m in train_manifests],
                    heldout_manifest=str(heldout_manifest), heldout_suite=suite.suite_id, rollout_repeats=suite.repeats,
                    scene_dependencies_sha256=scene_hash, execs=args.execs, rl_steps=args.rl_steps, eval_every=args.eval_every,
                    rehearsal=bool(args.rehearsal), references=tracking.REFERENCES, tranche=TRANCHE_TAG)
    if not args.probe:
        write_immutable_json(root / "identity.json", identity)

    state = dict(status="running", phase="load", started_at=time.time(), model=args.model, steps_target=config.max_steps)
    previous = root / "progress.json"
    if previous.exists() and not args.probe:
        old = json.loads(previous.read_text())
        state.update({k: old[k] for k in ("final", "rollouts", "wandb_id", "wandb_url") if k in old})

    def progress(**values):
        state.update(values); state["updated_at"] = time.time(); atomic_json(root / "progress.json", state)

    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    sources = [CaptureSource.load(m, name=d.name) for m, d in zip(train_manifests, data_dirs)]
    stats_path = root / "stats.json"
    if stats_path.exists():
        stats = QuantileStats.from_json(json.loads(stats_path.read_text()))
    else:
        stats = QuantileStats.from_sources(sources, config.chunk_horizon, seed=config.seed)
        atomic_json(stats_path, stats.to_json())
    model, info = build_model(config, stats, "cuda")
    named = [(n, p) for n, p in model.policy.named_parameters() if p.requires_grad]
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else contextlib.nullcontext

    if args.probe:
        results = []
        for micro in args.probe_micro:
            batches = Batches(sources, micro, config.chunk_horizon, config.seed, threads=args.loader_threads)
            torch.cuda.reset_peak_memory_stats()
            model.train()
            try:
                times, waits = [], []
                for i in range(args.probe_steps):
                    t0 = time.time()
                    frames, st, tg = batches.next()
                    waits.append(time.time() - t0)
                    with autocast():
                        loss, _ = model.loss(frames, st, tg)
                    loss.backward()
                    model.policy.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if i >= 2:
                        times.append(time.time() - t0)
                row = dict(micro=micro, seconds_per_micro=float(np.mean(times)), samples_per_s=micro / float(np.mean(times)),
                           data_wait_fraction=float(np.sum(waits[2:]) / np.sum(times)), gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            except torch.cuda.OutOfMemoryError:
                row = dict(micro=micro, oom=True)
                torch.cuda.empty_cache()
            batches.pool.shutdown(wait=False, cancel_futures=True)
            results.append(row)
            print("PROBE", json.dumps(row), flush=True)
        atomic_json(root / "probe.json", dict(model=args.model, amp=amp, loader_threads=args.loader_threads, rows=results, info=info))
        return 0

    tracker = tracking.init_run(family=args.model, job_type="finetune", rehearsal=args.rehearsal, run_id=state.get("wandb_id"), config=identity,
                                name=(tracking.run_name(args.model, rehearsal=True) if args.rehearsal else
                                      args.run_name or tracking.run_name(args.model, config.train_scope, f"{config.max_steps // 1000}k", f"shift{args.shift}", scene_hash=scene_hash)),
                                tags=[config.train_scope, "placements", f"shift{args.shift}", TRANCHE_TAG], directory=root)
    tracking.persist_run_id(root / "wandb.json", tracker)
    progress(wandb_id=tracker.id, wandb_url=tracker.url)
    tracker.define("02_generalisation/heldout_start_90", summary="min")
    tracker.define("02_generalisation/median", summary="min")

    def phase(name, step):
        progress(phase=name)
        tracker.phase(name, PHASES, step)

    try:
        phase("load", 0)
        heldout = CaptureSource.load(heldout_manifest, name="heldout")
        scorer = HeldoutScorer(heldout, sources)
        samples_total = int(sum(s.rows.shape[0] for s in sources))
        print(f"VLA {args.model}: {info['parameters_total'] / 1e6:.0f}M parameters, {info['parameters_trainable'] / 1e6:.0f}M trainable; "
              f"{samples_total} frames; amp {amp}", flush=True)
        tracker.config_update({"parameters_total": info["parameters_total"], "parameters_trainable": info["parameters_trainable"],
                               "samples_total": samples_total, "epochs": config.max_steps * config.batch_size / samples_total})
        weights = MasterWeights([p for _, p in named])
        if hasattr(model, "param_groups"):
            groups = model.param_groups(list(zip([n for n, _ in named], weights.master)))
        else:
            groups = [dict(params=weights.master, lr=config.learning_rate, base_lr=config.learning_rate)]
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)
        scratch = root / "scratch.pt"
        start_step = 0
        if scratch.exists() and "final" not in state:
            saved = torch.load(scratch, map_location="cpu", weights_only=False)
            with torch.no_grad():
                for m, t in zip(weights.master, saved["master"]):
                    m.copy_(t.to(device))
            weights.push()
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            del saved
            print(f"VLA resumed at step {start_step}", flush=True)
        shift_gen = torch.Generator().manual_seed(config.seed + 7919 + start_step)
        torch.manual_seed(config.seed + start_step)
        curve_path = root / "heldout_curve.jsonl"

        def evaluate(step):
            curve = scorer.score(model)
            payload = {"03_training/step": step, **{f"02_generalisation/{k}": v for k, v in curve.items() if isinstance(v, (int, float))},
                       **tracking.REFERENCE_LINES, **{f"02_generalisation/pose/{pose}": v for pose, v in curve["per_pose"].items()}}
            tracker.log(payload)
            with open(curve_path, "a") as handle:
                handle.write(json.dumps(dict(step=step, **curve)) + "\n")
            print(f"VLA eval step {step}: heldout90 {curve['heldout_start_90']:.2e} train90 {curve['train_start_90']:.2e} "
                  f"ratio {curve['ratio_start_90']:.1f}x median {curve['median']:.1e} under2.5e-4 {curve['poses_under_threshold']}/{curve['poses']}", flush=True)
            return curve

        final_path = root / "model" / f"{args.model}.pt"
        if "final" not in state:
            phase("train", start_step)
            if start_step == 0:
                evaluate(0)
            n_micro = config.batch_size // config.micro_batch
            batches = Batches(sources, config.micro_batch, config.chunk_horizon, config.seed, skip=start_step * n_micro, threads=args.loader_threads)
            model.train()
            window = dict(loss=[], joints=[], grad=[], wait=0.0, t=time.time(), steps=0)
            run_started = time.time()
            for step in range(start_step, config.max_steps):
                factor = lr_factor(step, config.warmup_steps, config.max_steps, config.final_lr_fraction)
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * factor
                total, joints = 0.0, torch.zeros(6, device=device)
                for _ in range(n_micro):
                    t0 = time.time()
                    frames, st, tg = batches.next()
                    window["wait"] += time.time() - t0
                    with autocast():
                        loss, per_joint = model.loss(frames, st, tg, generator=shift_gen)
                    (loss / n_micro).backward()
                    total += float(loss.detach()) / n_micro
                    joints += per_joint.float() / n_micro
                weights.pull_grads()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(weights.master, config.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                weights.push()
                window["loss"].append(total); window["joints"].append(joints); window["grad"].append(grad_norm); window["steps"] += 1
                done = step + 1
                if done % 20 == 0 or done == config.max_steps:
                    elapsed = time.time() - window["t"]
                    rate = window["steps"] / max(elapsed, 1e-9)
                    jl = torch.stack(window["joints"]).mean(0).tolist()
                    tracker.log({"03_training/step": done, "03_training/loss": float(np.mean(window["loss"])),
                                 **{f"03_training/loss_{j}": v for j, v in zip(JOINTS, jl)},
                                 "03_training/grad_norm": float(np.mean(window["grad"])), "03_training/lr_main": config.learning_rate * factor,
                                 "03_training/epoch": done * config.batch_size / samples_total,
                                 "04_throughput/steps_per_s": rate, "04_throughput/samples_per_s": rate * config.batch_size,
                                 "04_throughput/data_wait_fraction": window["wait"] / max(elapsed, 1e-9),
                                 "04_throughput/gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                                 "04_throughput/eta_hours": (config.max_steps - done) / max(rate, 1e-9) / 3600,
                                 "04_throughput/elapsed_hours": (time.time() - run_started) / 3600})
                    if done % 200 == 0:
                        print(f"VLA step {done}/{config.max_steps} loss {np.mean(window['loss']):.4f} grad {np.mean(window['grad']):.3f} "
                              f"{rate * config.batch_size:.1f} samples/s wait {window['wait'] / max(elapsed, 1e-9):.0%} "
                              f"eta {(config.max_steps - done) / max(rate, 1e-9) / 3600:.1f} h", flush=True)
                    window = dict(loss=[], joints=[], grad=[], wait=0.0, t=time.time(), steps=0)
                    progress(step=done)
                if done % args.eval_every == 0 or done == config.max_steps:
                    evaluate(done)
                    model.train()
                if done % args.scratch_every == 0 and done < config.max_steps:
                    tmp = scratch.with_suffix(".tmp")
                    torch.save(dict(step=done, master=[m.detach().cpu() for m in weights.master], optimizer=optimizer.state_dict()), tmp)
                    tmp.replace(scratch)
            batches.pool.shutdown(wait=False, cancel_futures=True)
            del optimizer, weights
            torch.cuda.empty_cache()
            model.save(final_path, dict(identity=identity, info=info))
            phase("score", config.max_steps)
            model.eval()
            final = scorer.score(model)
            progress(final=final)
            if scratch.exists():
                scratch.unlink()
        else:
            del model
            torch.cuda.empty_cache()
            model = load_model(final_path)
            final = state["final"]
        model.eval()

        phase("rollouts", config.max_steps)
        outcomes = state.get("rollouts", {})
        plan = [(str(e), e, None) for e in args.execs]
        if args.rl_steps and model.flow:
            plan.append((f"90_K{args.rl_steps}", 90, args.rl_steps))
        for key, execute, steps in plan:
            if key not in outcomes:
                with autocast():
                    outcomes[key] = closed_loop(args.scene, suite, model, execute, root / "evaluations" / f"exec{key}", num_steps=steps,
                                                tag=f"{args.model}" + (f"_K{steps}" if steps else ""))
                progress(rollouts=outcomes)
            r = outcomes[key]
            print(f"VLA exec {key}: {r['successes']}/{r['rollouts']} success, {r['pickups']} pickups, safety {r['safety_frames']}, "
                  f"closure median {r['closure'].get('lateral_median_mm', float('nan')):.1f} mm, {r['categories']} ({r['seconds']:.0f} s)", flush=True)
            if steps is None:
                log_outcome(tracker, r, config.max_steps)
            else:
                tracker.log({"03_training/step": config.max_steps, f"01_outcome/successes_exec90_K{steps}": r["successes"]})
        summary_figure(root / "summary.png", args.model, curve_path, final, {k: v for k, v in outcomes.items() if k.isdigit()})
        tracker.image("01_outcome/summary_figure", root / "summary.png")
        best = max((v for k, v in outcomes.items() if k.isdigit()), key=lambda r: r["successes"])
        tracker.summary({"01_outcome/best_successes": best["successes"], "01_outcome/best_execute": best["execute"],
                         "02_generalisation/final_heldout_start_90": final["heldout_start_90"],
                         "02_generalisation/final_ratio_start_90": final["ratio_start_90"], "02_generalisation/final_median": final["median"]})
        atomic_json(root / "result.json", dict(identity=identity, info=info, final=final, rollouts=outcomes, wandb=dict(id=tracker.id, url=tracker.url)))
        phase("complete", config.max_steps)
        tracker.finish()
        progress(status="complete", phase="complete")
        return 0
    except BaseException as exc:
        import traceback
        progress(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
