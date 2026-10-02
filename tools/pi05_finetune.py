"""Fine-tune pi0.5 on the bench placement captures and judge it the way every placement run is judged (2026-10-02).

What carries over from the placement tranche (notes/vision-rung-notebook.md, AGENTS.md):
- data: the 4800-placement capture (stride 30) AND the 2400-placement ladder capture (stride 18) together;
  more placements was the one lever that moved held-out loss (data exponent 0.4, ratio 5250x -> 1.6x);
- random-shift augmentation, 12 px at 256 px, always (memorisation came back every time it was off);
- the same residual 90-step chunk target and the same held-out capture (10 unseen placements), so the
  held-out start-90 MSE is in the CNN runs' units: CNN-4800 3.6e-4, working fixed-square policy 2.5e-6;
- the held-out curve is logged during training (02_generalisation/*), not only at the end;
- the same 30 closed-loop held-out rollouts, gripper clamp on, plus closure error in millimetres at step 250
  against the working policy's jaw point (simulation/closure.py; tolerance ~8 mm, CNN-4800 median 5.1 mm);
- the open-loop finding of 2026-09-18: rollouts run twice, executing 90 commands per frame (the CNN schedule)
  and 30 (re-reading the camera at every row a training frame exists for), so we see whether a pretrained
  VLA that looks more often corrects anything without perturbed-teacher data.

W&B (project so-arm101-v2-scaling, group pi05): numbered sections, most important first:
01_outcome (success, closure mm, failures), 02_generalisation (held-out curve vs the CNN and the working
policy), 03_training (loss, per-joint loss, lr, grad norm), 04_throughput, 05_phases. The workspace layout
(multi-line, log-scale panels in that order) is written by tools/pi05_wandb_workspace.py.

Run in the lerobot-pi05 env:
  PYTHONNOUSERSITE=1 MUJOCO_GL=egl ~/miniforge3/envs/lerobot-pi05/bin/python tools/pi05_finetune.py --output-dir ...
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from so_arm101_v2.data._serialization import write_immutable_json  # noqa: E402

EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DEFAULT_DATA = [EXPERIMENTS / "placements4800_20260912", EXPERIMENTS / "scaling_ladder_20260910"]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Reference points (tranche_analysis_20260918, augmentation_shift12_4800_20260912).
REF = dict(cnn4800_heldout_start_90=3.6e-4, working_policy_start_90=2.5e-6, cnn4800_successes=6, cnn_best_pooled_rate=33 / 90,
           cnn4800_closure_lateral_median_mm=5.1, working_policy_closure_lateral_median_mm=1.7, grasp_tolerance_mm=8.0)
PHASES = ("load", "train", "score", "rollouts", "complete")


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.replace(path)


class Batches:
    """Seeded epoch-shuffled micro-batches over all sources, prefetched by threads (memmap reads + target build)."""

    def __init__(self, sources, micro: int, horizon: int, seed: int, skip: int = 0, threads: int = 4, ahead: int = 8) -> None:
        self.sources, self.micro, self.horizon = sources, micro, horizon
        self.index = np.concatenate([np.stack([np.full(s.rows.shape[0], k), s.rows], 1) for k, s in enumerate(sources)])
        self.rng = np.random.default_rng(seed)
        self.order = np.empty((0, 2), dtype=np.int64)
        self.pool = ThreadPoolExecutor(threads)
        self.queue: deque = deque()
        for _ in range(skip):      # resume: replay the draw stream without reading frames
            self._draw()
        for _ in range(ahead):
            self.queue.append(self.pool.submit(self._load, self._draw()))

    def _draw(self) -> np.ndarray:
        if self.order.shape[0] < self.micro:
            self.order = np.concatenate([self.order, self.index[self.rng.permutation(self.index.shape[0])]])
        pick, self.order = self.order[: self.micro], self.order[self.micro:]
        return pick

    def _load(self, pick: np.ndarray):
        images = np.empty((pick.shape[0], 256, 256, 3), dtype=np.uint8)
        state = np.empty((pick.shape[0], 7), dtype=np.float32)
        targets = np.empty((pick.shape[0], self.horizon, 6), dtype=np.float32)
        for k, source in enumerate(self.sources):
            where = np.flatnonzero(pick[:, 0] == k)
            if where.size:
                rows = pick[where, 1]
                images[where] = source.images(rows); state[where] = source.state(rows); targets[where] = source.targets(rows, self.horizon)
        return images, state, targets

    def next(self):
        future = self.queue.popleft()
        self.queue.append(self.pool.submit(self._load, self._draw()))
        return future.result()


def lr_factor(step: int, warmup: int, total: int, final: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def rollouts(model_xml: Path, suite, policy, config, stats, prompt, execute: int, directory: Path, closure_model: Path):
    from so_arm101_v2.simulation.closure import closure_summary
    from so_arm101_v2.simulation.pi05_policy import Pi05ChunkPolicy
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop

    name = f"pi05_exec{execute}"
    built = []

    def factory():
        p = Pi05ChunkPolicy(policy, config, stats, prompt, execute=execute, clamp_channels=(5,), tag="pi05")
        built.append(p)
        return p

    started = time.time()
    result = evaluate_closed_loop(model_xml, suite, {name: factory}, directory, environment_proven=True, record_video=False, workers=1)
    report = json.loads(Path(result.report_json).read_text())
    rows = [r for r in report["rollouts"] if r["policy_id"] == name]
    ok = [bool(r["success"] and not r["invalidated"]) for r in rows]
    categories: dict[str, int] = {}
    for r in rows:
        categories[r["failure_category"]] = categories.get(r["failure_category"], 0) + 1
    per_scenario: dict[str, int] = {}
    for r, s in zip(rows, ok):
        per_scenario[r["scenario_id"]] = per_scenario.get(r["scenario_id"], 0) + int(s)
    closure = closure_summary(closure_model, [r["telemetry_path"] for r in rows if r.get("telemetry_path")])
    return dict(execute=execute, successes=int(sum(ok)), rollouts=len(rows),
                safety_frames=int(sum(sum(r[k] for k in ("clipping_frames", "limiting_frames", "nonfinite_frames", "unsafe_contact_frames")) for r in rows)),
                pickups=int(sum(bool(r["pickup_completed"]) for r in rows)), categories=categories, per_scenario=per_scenario,
                closure={k: v for k, v in closure.items() if k != "rows"}, closure_rows=closure.get("rows", []),
                rollout_rows=[{k: r[k] for k in ("scenario_id", "repeat", "success", "invalidated", "failure_category", "pickup_completed", "maximum_height_gain_m") if k in r} for r in rows],
                seconds=round(time.time() - started, 1), report=str(result.report_json))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dirs", type=Path, nargs="+", default=DEFAULT_DATA, help="dirs holding train_manifest.path (first one also gives the held-out pointers)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--scope", choices=("expert", "expert_lora"), default="expert_lora")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--shift", type=int, default=12)
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--scratch-every", type=int, default=1000)
    p.add_argument("--execs", type=int, nargs="+", default=[90, 30])
    p.add_argument("--rollout-limit", type=int, default=None, help="rehearsal only: first N held-out scenarios, 1 repeat")
    p.add_argument("--run-name", default=None)
    p.add_argument("--rehearsal", action="store_true", help="toy settings under a rehearsal/ directory; W&B offline; nothing may be quoted from it")
    args = p.parse_args(argv)

    import mujoco
    if mujoco.__version__ != "3.9.0":
        raise RuntimeError("bench requires MuJoCo 3.9.0")
    root = args.output_dir.resolve()
    if args.rehearsal and "rehearsal" not in root.parts:
        raise RuntimeError("--rehearsal output must live under a directory named 'rehearsal'")
    if args.rollout_limit is not None and not args.rehearsal:
        raise RuntimeError("--rollout-limit is a rehearsal-only switch; real runs use the full 30-rollout suite")
    root.mkdir(parents=True, exist_ok=True)

    import torch
    import wandb
    from so_arm101_v2.contracts.bench import scene_dependency_hash
    from so_arm101_v2.learning.pi05 import (CaptureSource, Pi05FinetuneConfig, Pi05Scorer, PromptTokenizer, QuantileStats, build_policy,
                                            flow_loss, prepare_images, save_checkpoint)
    from so_arm101_v2.simulation.suites import load_suite_from_path

    config = Pi05FinetuneConfig(train_scope=args.scope, lora_rank=args.lora_rank, lora_alpha=float(args.lora_rank), batch_size=args.batch,
                                micro_batch=args.micro, max_steps=args.steps, learning_rate=args.lr, lora_learning_rate=args.lora_lr,
                                warmup_steps=args.warmup, random_shift=args.shift, seed=args.seed)
    data_dirs = [d.resolve() for d in args.data_dirs]
    train_manifests = [Path((d / "train_manifest.path").read_text().strip()) for d in data_dirs]
    heldout_manifest = Path((data_dirs[0] / "heldout_manifest.path").read_text().strip())
    suite = load_suite_from_path(Path((data_dirs[0] / "heldout_suite.path").read_text().strip()))
    if args.rollout_limit is not None:
        suite = dataclasses.replace(suite, suite_id=f"{suite.suite_id}_rehearsal{args.rollout_limit}", scenarios=suite.scenarios[: args.rollout_limit], repeats=1)
    identity = dict(tool="pi05_finetune", config=dataclasses.asdict(config), train_manifests=[str(m) for m in train_manifests],
                    heldout_manifest=str(heldout_manifest), heldout_suite=suite.suite_id, rollout_repeats=suite.repeats,
                    scene_dependencies_sha256=scene_dependency_hash(args.model), execs=args.execs, eval_every=args.eval_every,
                    pretrained_revision=str(sorted(Path.home().glob(".cache/huggingface/hub/models--lerobot--pi05_base/snapshots/*"))[-1].name)
                    if Path.home().joinpath(".cache/huggingface/hub/models--lerobot--pi05_base").exists() else None,
                    rehearsal=bool(args.rehearsal), references=REF)
    write_immutable_json(root / "pi05_identity.json", identity)

    state = dict(status="running", phase="load", started_at=time.time())
    previous = root / "progress.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        state.update({k: old[k] for k in ("final", "rollouts", "wandb_id") if k in old})

    def progress(**values):
        state.update(values); state["updated_at"] = time.time(); atomic_json(root / "progress.json", state)

    scene = identity["scene_dependencies_sha256"][:8]
    tracker = wandb.init(project="so-arm101-v2-scaling", group="pi05", job_type="finetune",
                         name=("pi05-rehearsal" if args.rehearsal else (args.run_name or f"pi05-{args.scope}-r{args.lora_rank}-{args.steps // 1000}k-shift{args.shift}-{scene}")),
                         id=state.get("wandb_id"), resume="allow", mode=("offline" if args.rehearsal else "online"), config=identity,
                         tags=["pi05", args.scope, "placements", f"shift{args.shift}"], settings=wandb.Settings(init_timeout=120))
    progress(wandb_id=tracker.id, wandb_url=getattr(tracker, "url", None))
    tracker.define_metric("03_training/step")
    for section in ("01_outcome/*", "02_generalisation/*", "03_training/*", "04_throughput/*", "05_phases/*"):
        tracker.define_metric(section, step_metric="03_training/step")
    for key in ("02_generalisation/heldout_start_90", "02_generalisation/per_pose_median"):
        tracker.define_metric(key, step_metric="03_training/step", summary="min")

    def log(payload):
        try:
            tracker.log(payload)
        except Exception as exc:   # tracking must never stop the run
            print(f"wandb log failed: {exc}", flush=True)

    def phase(name, step):
        progress(phase=name)
        log({"03_training/step": step, "05_phases/phase": PHASES.index(name), **{f"05_phases/{p}": int(p == name) for p in PHASES}})

    try:
        device = torch.device("cuda")
        torch.manual_seed(config.seed)
        phase("load", 0)
        sources = [CaptureSource.load(m, name=d.name) for m, d in zip(train_manifests, data_dirs)]
        heldout = CaptureSource.load(heldout_manifest, name="heldout")
        stats_path = root / "stats.json"
        if stats_path.exists():
            stats = QuantileStats.from_json(json.loads(stats_path.read_text()))
        else:
            stats = QuantileStats.from_sources(sources, config.chunk_horizon, seed=config.seed)
            atomic_json(stats_path, stats.to_json())
        prompt = PromptTokenizer(config.tokenizer, config.tokenizer_max_length)
        scorer = Pi05Scorer(heldout, sources, horizon=config.chunk_horizon)
        samples_total = int(sum(s.rows.shape[0] for s in sources))
        print(f"PI05 data: {[ (s.name, int(s.rows.shape[0])) for s in sources]} = {samples_total} frames; held-out {len(set(heldout.scenario_ids))} poses", flush=True)
        policy, info = build_policy(config, device="cuda")
        print(f"PI05 model: {info['parameters_total'] / 1e9:.2f}B parameters, {info['parameters_trainable'] / 1e6:.0f}M trainable, "
              f"{info.get('lora_modules', 0)} LoRA modules", flush=True)
        tracker.config.update({"parameters_total": info["parameters_total"], "parameters_trainable": info["parameters_trainable"],
                               "samples_total": samples_total, "epochs": args.steps * args.batch / samples_total}, allow_val_change=True)
        named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        master = [p.detach().float().clone() if p.dtype != torch.float32 else p for p in params]
        lora_flags = [".lora_" in n for n, _ in named]
        groups = [dict(params=[m for m, f in zip(master, lora_flags) if not f], lr=config.learning_rate, base_lr=config.learning_rate),
                  dict(params=[m for m, f in zip(master, lora_flags) if f], lr=config.lora_learning_rate, base_lr=config.lora_learning_rate)]
        groups = [g for g in groups if g["params"]]
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)
        scratch = root / "scratch.pt"
        start_step = 0
        if scratch.exists() and "final" not in state:
            saved = torch.load(scratch, map_location="cpu", weights_only=False)
            with torch.no_grad():
                for m, t in zip(master, saved["master"]):
                    m.copy_(t.to(device))
                for p, m in zip(params, master):
                    p.copy_(m)
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
            del saved
            print(f"PI05 resumed at step {start_step}", flush=True)
        shift_gen = torch.Generator().manual_seed(config.seed + 7919 + start_step)
        torch.manual_seed(config.seed + start_step)
        curve_path = root / "heldout_curve.jsonl"

        def evaluate(step):
            curve = scorer.score(policy, stats, prompt, device)
            payload = {"03_training/step": step, **{f"02_generalisation/{k}": v for k, v in curve.items() if isinstance(v, (int, float))},
                       **{f"02_generalisation/ref_{k}": REF[k] for k in ("cnn4800_heldout_start_90", "working_policy_start_90")},
                       "02_generalisation/ref_pose_threshold_2.5e-4": 2.5e-4}
            payload.update({f"02_generalisation/pose/{pose}": v for pose, v in curve["per_pose"].items()})
            log(payload)
            with open(curve_path, "a") as handle:
                handle.write(json.dumps(dict(step=step, **curve)) + "\n")
            print(f"PI05 eval step {step}: heldout90 {curve['heldout_start_90']:.2e} ({curve['heldout_start_90_deg_rms']:.2f} deg rms) "
                  f"train90 {curve['train_start_90']:.2e} ratio {curve['ratio_start_90']:.1f}x median {curve['median']:.1e} "
                  f"under2.5e-4 {curve['poses_under_threshold']}/{curve['poses']} start0 {curve['heldout_start_0']:.1e} "
                  f"later {curve['heldout_starts_180_270_360']:.1e}", flush=True)
            return curve

        final_path = root / "model" / "pi05.pt"
        if "final" not in state:
            phase("train", start_step)
            if start_step == 0:
                evaluate(0)
            batches = Batches(sources, config.micro_batch, config.chunk_horizon, config.seed,
                              skip=start_step * (config.batch_size // config.micro_batch))
            n_micro = config.batch_size // config.micro_batch
            policy.train()
            window = dict(loss=[], joints=[], grad=[], wait=0.0, t=time.time(), steps=0)
            run_started = time.time()
            for step in range(start_step, config.max_steps):
                factor = lr_factor(step, config.warmup_steps, config.max_steps, config.final_lr_fraction)
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * factor
                total_loss, joint_loss = 0.0, torch.zeros(6, device=device)
                for _ in range(n_micro):
                    t0 = time.time()
                    images_np, state_np, targets_np = batches.next()
                    window["wait"] += time.time() - t0
                    images = prepare_images(images_np, device, shift=config.random_shift, generator=shift_gen)
                    tokens, masks = prompt(stats.norm_state(state_np), device)
                    norm_actions = stats.norm_action(torch.from_numpy(targets_np).to(device))
                    losses = flow_loss(policy, images, tokens, masks, norm_actions)
                    loss = losses.mean() / n_micro
                    loss.backward()
                    total_loss += float(loss.detach())
                    joint_loss += losses.detach().mean(dim=(0, 1)) / n_micro
                for p, m in zip(params, master):
                    if p.grad is None:
                        m.grad = None
                    else:
                        m.grad = p.grad.float() if m is not p else p.grad
                        if m is not p:
                            p.grad = None
                grad_norm = float(torch.nn.utils.clip_grad_norm_(master, config.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, m in zip(params, master):
                        if m is not p:
                            p.copy_(m)
                window["loss"].append(total_loss); window["joints"].append(joint_loss); window["grad"].append(grad_norm); window["steps"] += 1
                done = step + 1
                if done % 10 == 0 or done == config.max_steps:
                    elapsed = time.time() - window["t"]
                    rate = window["steps"] / max(elapsed, 1e-9)
                    joints = torch.stack(window["joints"]).mean(0).tolist()
                    log({"03_training/step": done, "03_training/flow_loss": float(np.mean(window["loss"])),
                         **{f"03_training/flow_loss_{j}": v for j, v in zip(JOINTS, joints)},
                         "03_training/grad_norm": float(np.mean(window["grad"])), "03_training/lr_expert": config.learning_rate * factor,
                         "03_training/lr_lora": config.lora_learning_rate * factor, "03_training/epoch": done * config.batch_size / samples_total,
                         "04_throughput/steps_per_s": rate, "04_throughput/samples_per_s": rate * config.batch_size,
                         "04_throughput/data_wait_fraction": window["wait"] / max(elapsed, 1e-9),
                         "04_throughput/gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                         "04_throughput/eta_hours": (config.max_steps - done) / max(rate, 1e-9) / 3600,
                         "04_throughput/elapsed_hours": (time.time() - run_started) / 3600})
                    if done % 100 == 0:
                        print(f"PI05 step {done}/{config.max_steps} loss {np.mean(window['loss']):.4f} grad {np.mean(window['grad']):.3f} "
                              f"{rate * config.batch_size:.1f} samples/s wait {window['wait'] / max(elapsed, 1e-9):.0%} "
                              f"eta {(config.max_steps - done) / max(rate, 1e-9) / 3600:.1f} h", flush=True)
                    window = dict(loss=[], joints=[], grad=[], wait=0.0, t=time.time(), steps=0)
                if done % args.eval_every == 0 or done == config.max_steps:
                    evaluate(done)
                    policy.train()
                if done % args.scratch_every == 0 and done < config.max_steps:
                    tmp = scratch.with_suffix(".tmp")
                    torch.save(dict(step=done, master=[m.detach().cpu() for m in master], optimizer=optimizer.state_dict()), tmp)
                    tmp.replace(scratch)
                    progress(phase="train", step=done)
            batches.pool.shutdown(wait=False, cancel_futures=True)
            del optimizer, master
            torch.cuda.empty_cache()
            save_checkpoint(final_path, policy, config, stats, dict(identity=identity, info=info))
            phase("score", config.max_steps)
            final = scorer.score(policy, stats, prompt, device)
            progress(final=final)
            if scratch.exists():
                scratch.unlink()
        else:
            from so_arm101_v2.learning.pi05 import load_checkpoint
            del policy
            torch.cuda.empty_cache()
            policy, _, stats, _ = load_checkpoint(final_path)
            final = state["final"]
        policy.eval()

        phase("rollouts", config.max_steps)
        done_rollouts = state.get("rollouts", {})
        for execute in args.execs:
            key = str(execute)
            if key not in done_rollouts:
                print(f"PI05 rollouts exec {execute}: {len(suite.scenarios) * suite.repeats} episodes", flush=True)
                done_rollouts[key] = rollouts(args.model, suite, policy, config, stats, prompt, execute, root / "evaluations" / f"exec{execute}", args.model)
                progress(rollouts=done_rollouts)
            r = done_rollouts[key]
            c = r["closure"]
            print(f"PI05 exec {execute}: {r['successes']}/{r['rollouts']} success, {r['pickups']} pickups, safety frames {r['safety_frames']}, "
                  f"closure lateral median {c.get('lateral_median_mm', float('nan')):.1f} mm, within 8 mm {c.get('within_tolerance')}/{c.get('count')}, "
                  f"not descended {c.get('not_descended')}, {r['categories']} ({r['seconds']:.0f} s)", flush=True)
            log({"03_training/step": config.max_steps,
                 f"01_outcome/success_rate_exec{execute}": r["successes"] / max(r["rollouts"], 1),
                 f"01_outcome/successes_exec{execute}": r["successes"], f"01_outcome/pickups_exec{execute}": r["pickups"],
                 f"01_outcome/safety_frames_exec{execute}": r["safety_frames"],
                 f"01_outcome/closure_lateral_median_mm_exec{execute}": c.get("lateral_median_mm"),
                 f"01_outcome/closure_lateral_rms_mm_exec{execute}": c.get("lateral_rms_mm"),
                 f"01_outcome/closure_within_8mm_exec{execute}": c.get("within_tolerance"),
                 f"01_outcome/closure_not_descended_exec{execute}": c.get("not_descended"),
                 f"01_outcome/closure_yaw_err_median_deg_exec{execute}": c.get("yaw_err_median_deg")})
            tracker.summary.update({f"01_outcome/success_rate_exec{execute}": r["successes"] / max(r["rollouts"], 1),
                                    f"01_outcome/successes_exec{execute}": r["successes"],
                                    f"01_outcome/closure_lateral_median_mm_exec{execute}": c.get("lateral_median_mm")})
            table = wandb.Table(columns=["execute", "scenario", "repeat", "success", "failure", "pickup", "max_lift_mm", "closure_lat_mm", "closure_dz_mm", "yaw_err_deg"])
            closure_by = {(x["scenario_id"], x["repeat"]): x for x in r["closure_rows"]}
            for row in r["rollout_rows"]:
                cr = closure_by.get((row["scenario_id"], row.get("repeat", 0)), {})
                table.add_data(execute, row["scenario_id"], row.get("repeat", 0), bool(row["success"] and not row["invalidated"]), row["failure_category"],
                               bool(row["pickup_completed"]), 1000 * float(row.get("maximum_height_gain_m", 0.0)), cr.get("lat"), cr.get("dz"), cr.get("yaw_err"))
            log({f"01_outcome/rollouts_exec{execute}": table})

        summary_figure(root, final, done_rollouts, curve_path)
        try:
            log({"01_outcome/summary_figure": wandb.Image(str(root / "pi05_summary.png"))})
        except Exception as exc:
            print(f"figure log failed: {exc}", flush=True)
        best_exec = max(done_rollouts.values(), key=lambda r: r["successes"])
        tracker.summary.update({"01_outcome/best_successes": best_exec["successes"], "01_outcome/best_execute": best_exec["execute"],
                                "02_generalisation/final_heldout_start_90": final["heldout_start_90"],
                                "02_generalisation/final_ratio_start_90": final["ratio_start_90"],
                                "02_generalisation/final_per_pose_median": final["median"]})
        atomic_json(root / "pi05_result.json", dict(identity=identity, info=info, final=final,
                                                    rollouts={k: {kk: vv for kk, vv in v.items()} for k, v in done_rollouts.items()},
                                                    wandb=dict(id=tracker.id, url=getattr(tracker, "url", None))))
        phase("complete", config.max_steps)
        tracker.finish()
        progress(status="complete", phase="complete")
        return 0
    except BaseException as exc:
        import traceback
        progress(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise


def summary_figure(root: Path, final: dict, done_rollouts: dict, curve_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = [json.loads(line) for line in curve_path.read_text().splitlines()] if curve_path.exists() else []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    if curve:
        steps = [c["step"] for c in curve]
        ax.plot(steps, [c["heldout_start_90"] for c in curve], "o-", label="held-out start 90")
        ax.plot(steps, [c["median"] for c in curve], "s-", label="held-out per-pose median")
        ax.plot(steps, [c["train_start_90"] for c in curve], "x--", alpha=.7, label="train start 90")
    ax.axhline(REF["cnn4800_heldout_start_90"], color="tab:red", ls=":", label="CNN 4800 held-out (3.6e-4)")
    ax.axhline(REF["working_policy_start_90"], color="tab:green", ls=":", label="working fixed-square policy (2.5e-6)")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("start-90 chunk MSE (1 unit = pi rad)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("generalisation during training")
    ax = axes[1]
    poses = sorted(final["per_pose"]); x = np.arange(len(poses))
    ax.bar(x, [final["per_pose"][p] for p in poses], color="tab:blue")
    ax.axhline(2.5e-4, color="k", ls=":"); ax.axhline(REF["working_policy_start_90"], color="tab:green", ls=":")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(poses, rotation=45, fontsize=7); ax.set_title("final held-out start-90 per pose"); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    labels, succ, lat = [], [], []
    for key, r in sorted(done_rollouts.items(), key=lambda kv: -int(kv[0])):
        labels.append(f"pi0.5 exec {key}"); succ.append(r["successes"]); lat.append(r["closure"].get("lateral_median_mm", np.nan))
    labels += ["CNN 4800 (exec 90)"]; succ += [REF["cnn4800_successes"]]; lat += [REF["cnn4800_closure_lateral_median_mm"]]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, succ, 0.4, label="successes / 30", color="tab:blue")
    ax2 = ax.twinx(); ax2.bar(xs + 0.2, lat, 0.4, label="closure lateral median (mm)", color="tab:orange")
    ax2.axhline(REF["grasp_tolerance_mm"], color="tab:orange", ls=":"); ax2.axhline(REF["working_policy_closure_lateral_median_mm"], color="tab:green", ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("successes of 30"); ax2.set_ylabel("mm")
    ax.set_title("closed loop on 10 unseen placements x 3"); ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(root / "pi05_summary.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
