"""RL fine-tuning on the bench task: ReinFlow (pi0.5, SmolVLA) or Gaussian PPO (ACT), retry tranche 2026-10-04.

Every run gets the same budget and the same conditions, so the three methods compare:
- a fixed number of training episodes (--episodes), walking the same fixed order of teacher-screened poses
  (rl/scenarios.py, seeds 30-37, disjoint from the held-out suite and both captures);
- the same reward (rl/reward.py: +1 success, -1 safety invalidation, bounded potential shaping on contact
  milestones only), the same chunk schedule (execute 90: the decision points of the CNN runs), the same PPO;
- the same monitoring suite (seed 21, 20 poses, noise off) every --monitor-every episodes, and at the end the
  same 30 held-out rollouts as every placement run (execute 90 and 30, closure mm), noise off.

Iteration: collect --episodes-per-iter episodes over --envs worker processes with batched inference, then one
PPO update (critic first; the actor is frozen for the first --critic-warmup iterations), log, checkpoint.
Resume is automatic from the output directory. Abort guards free the GPU early when a run has clearly
diverged: a non-finite loss, post-update KL above 10x the target for 3 updates in a row, or monitoring
success at 0 for 3 evaluations in a row while the starting policy was above 0.

Modes: --probe (throughput sweep over --probe-envs, writes probe.json), --noise-probe (monitor suite with noise
on at each --noise-levels value: success and safety, to choose the noise range), --rehearsal.

pi0.5 runs in the lerobot-pi05 env; SmolVLA and ACT in lerobot.
"""
from __future__ import annotations

import argparse
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

EXPERIMENTS = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DEFAULT_SUITES = EXPERIMENTS / "retry_tranche_20261004/rl_suites/rl_suites.json"
DEFAULT_HELDOUT = EXPERIMENTS / "scaling_ladder_20260910/heldout_suite.path"
PHASES = ("load", "warmup", "train", "heldout", "complete")
TRANCHE_TAG = "retry-tranche-20261004"
DEFAULTS = {
    "pi05": dict(epochs=4, actor_lr=1e-6, clip=0.05, target_kl=0.1, grad_clip=0.25, micro=8, minibatch=32, steps=10, sigma_min=0.005, sigma_max=0.015),
    "smolvla": dict(epochs=4, actor_lr=3e-6, clip=0.05, target_kl=0.1, grad_clip=0.25, micro=16, minibatch=64, steps=10, sigma_min=0.005, sigma_max=0.015),
    "act": dict(epochs=2, actor_lr=1e-5, clip=0.1, target_kl=0.02, grad_clip=0.5, micro=64, minibatch=64, std_init=0.01),
}


def rss_gb() -> float:
    try:
        import psutil
        proc = psutil.Process()
        return (proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))) / 1e9
    except Exception:
        return float("nan")


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.replace(path)


def load_policy(kind: str, checkpoint: Path, args, *, amp: bool):
    from so_arm101_v2.learning.vla_models import ACTModel, Pi05Model, SmolVLAModel
    if kind == "pi05":
        model = Pi05Model.from_checkpoint(checkpoint)
    elif kind == "smolvla":
        model = SmolVLAModel.from_checkpoint(checkpoint)
    else:
        model = ACTModel.from_checkpoint(checkpoint)
    if kind == "act":
        from so_arm101_v2.rl.act_ppo import GaussianChunkPolicy
        policy = GaussianChunkPolicy(model, std_init=args.std_init, std_max=args.std_max, micro=args.micro)
    else:
        from so_arm101_v2.rl.reinflow import ReinFlowPolicy
        policy = ReinFlowPolicy(model, steps=args.steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max, micro=args.micro, amp=amp,
                                scope=args.rl_scope)
    return model, policy


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=("pi05", "smolvla", "act"))
    p.add_argument("--checkpoint", type=Path, required=True, help="the fine-tune to start from")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8000, help="training-episode budget (equal across methods)")
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--episodes-per-iter", type=int, default=48)
    p.add_argument("--execute", type=int, default=90)
    p.add_argument("--batch-fraction", type=float, default=0.5)
    p.add_argument("--watchdog", type=float, default=30.0, help="seconds without an answer before an env worker is restarted (a chunk takes 1-3 s)")
    p.add_argument("--steps", type=int, default=None, help="denoising steps K (flow models)")
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--std-init", type=float, default=None, help="ACT Gaussian std (normalised residual units)")
    p.add_argument("--std-max", type=float, default=0.03, help="ACT std cap (noise above ~0.02 drives the shoulder through its floor)")
    p.add_argument("--actor-lr", type=float, default=None)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=None)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--minibatch", type=int, default=None)
    p.add_argument("--micro", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="PPO epochs per batch (KL early stop still applies)")
    p.add_argument("--rl-scope", choices=("out_head", "heads"), default="out_head",
                   help="flow models: out_head = action_out_proj + noise head on stored features (fast); heads = legacy stable heads")
    p.add_argument("--critic-warmup", type=int, default=10)
    p.add_argument("--monitor-every", type=int, default=480)
    p.add_argument("--rl-suites", type=Path, default=DEFAULT_SUITES)
    p.add_argument("--heldout-suite", type=Path, default=DEFAULT_HELDOUT)
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--execs", type=int, nargs="+", default=[90, 30])
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--amp", choices=("auto", "on", "off"), default="auto", help="bf16 autocast (auto: on for SmolVLA)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--rehearsal", action="store_true")
    p.add_argument("--rollout-limit", type=int, default=None, help="rehearsal only: first N held-out scenarios, 1 repeat")
    p.add_argument("--no-wandb", action="store_true", help="probes: record metrics without a W&B run")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--probe-envs", type=int, nargs="+", default=[4, 8, 12, 14])
    p.add_argument("--probe-episodes", type=int, default=None, help="episodes per probe point (default 2 x envs)")
    p.add_argument("--noise-probe", action="store_true")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    args = p.parse_args(argv)
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    amp = args.amp == "on" or (args.amp == "auto" and args.model == "smolvla")

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
    from so_arm101_v2.learning.vla_models import MasterWeights
    from so_arm101_v2.rl.collector import Collector, summarize
    from so_arm101_v2.rl.critic import make_critic
    from so_arm101_v2.rl.env import PRIVILEGED_DIM
    from so_arm101_v2.rl.ppo import PPOConfig, ppo_update
    from so_arm101_v2.rl.reward import RewardConfig
    from so_arm101_v2.rl.scenarios import ScenarioPool, load_monitor
    from so_arm101_v2.rl.vec_env import SubprocChunkVecEnv
    from so_arm101_v2.simulation.suites import load_suite_from_path

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pool = ScenarioPool.from_index(args.rl_suites)
    monitor = load_monitor(args.rl_suites)
    reward = RewardConfig()
    model, policy = load_policy(args.model, args.checkpoint, args, amp=amp)
    scene_hash = scene_dependency_hash(args.scene)

    # ------------------------------------------------------------------ probes
    if args.probe or args.noise_probe:
        rows = []
        if args.probe:
            for envs in args.probe_envs:
                n = args.probe_episodes or 2 * envs
                with SubprocChunkVecEnv(args.scene, envs, reward_kwargs=reward.as_dict()) as vec:
                    collector = Collector(vec, policy, execute=args.execute, batch_fraction=args.batch_fraction, watchdog_s=args.watchdog)
                    collector.collect([(i, pool.episode(i)) for i in range(envs)], deterministic=False, seed=args.seed)   # warm-up
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    batch, summaries, timing = collector.collect([(i, pool.episode(i)) for i in range(n)], deterministic=False, seed=args.seed)
                    collect_s = time.time() - t0
                    arrays = batch.arrays()
                    t1 = time.time()
                    with torch.no_grad():
                        logp, _ = policy.logprob(arrays["frames"][: args.minibatch], arrays["states"][: args.minibatch], arrays["records"][: args.minibatch])
                    consistency = float(np.max(np.abs(logp.cpu().numpy() - arrays["logp_rollout"][: args.minibatch])))
                    logp_grad, _ = policy.logprob(arrays["frames"][: args.minibatch], arrays["states"][: args.minibatch], arrays["records"][: args.minibatch])
                    logp_grad.sum().backward()
                    torch.cuda.synchronize()
                    update_s = (time.time() - t1) * len(batch) / max(1, args.minibatch) * 1.4   # + post-update KL forward
                    for _, param in policy.named_parameters():
                        param.grad = None
                    row = dict(envs=envs, episodes=n, chunks=len(batch), collect_seconds=collect_s, update_seconds_est=update_s,
                               episodes_per_hour=n / (collect_s + update_s) * 3600, collect_only_episodes_per_hour=n / collect_s * 3600,
                               policy_seconds=timing["policy_seconds"], env_wait_seconds=timing["env_wait_seconds"], mean_batch=timing["mean_batch"],
                               gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9, host_rss_gb=rss_gb(), logprob_consistency_max=consistency,
                               success_rate_noisy=summarize(summaries)["success_rate"], safety_frames_mean=summarize(summaries)["safety_frames_mean"])
                    rows.append(row)
                    print("PROBE", json.dumps(row), flush=True)
            atomic_json(root / "probe.json", dict(model=args.model, steps=getattr(args, "steps", None), micro=args.micro, minibatch=args.minibatch,
                                                  execute=args.execute, amp=amp, rows=rows))
        if args.noise_probe:
            results = []
            levels = [None] + list(args.noise_levels)
            with SubprocChunkVecEnv(args.scene, args.envs, reward_kwargs=reward.as_dict()) as vec:
                for level in levels:
                    if level is not None:
                        if args.model == "act":
                            with torch.no_grad():
                                policy.log_std.fill_(math.log(level))
                        else:
                            policy.sigma_min, policy.sigma_max = level * 0.999, level * 1.001
                    collector = Collector(vec, policy, execute=args.execute, batch_fraction=args.batch_fraction, watchdog_s=args.watchdog)
                    _, summaries, _ = collector.collect([(i, s) for i, s in enumerate(monitor)], deterministic=level is None, seed=args.seed)
                    row = dict(noise=level, **summarize(summaries))
                    results.append(row)
                    print("NOISE", json.dumps(row), flush=True)
            atomic_json(root / "noise_probe.json", dict(model=args.model, steps=getattr(args, "steps", None), rows=results))
        return 0

    # ------------------------------------------------------------------ run
    heldout = load_suite_from_path(Path(args.heldout_suite.read_text().strip()) if args.heldout_suite.suffix == ".path" else args.heldout_suite)
    if args.rollout_limit is not None:
        heldout = dataclasses.replace(heldout, suite_id=f"{heldout.suite_id}_rehearsal{args.rollout_limit}", scenarios=heldout.scenarios[: args.rollout_limit], repeats=1)
    ppo = PPOConfig(clip=args.clip, epochs=args.epochs, minibatch=args.minibatch, micro=args.micro, target_kl=args.target_kl,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr, grad_clip=args.grad_clip, critic_warmup_iterations=args.critic_warmup)
    policy_config = (dict(steps=args.steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max, scope=args.rl_scope) if args.model != "act"
                     else dict(std_init=args.std_init, std_max=args.std_max))
    identity = dict(tool="rl_train", model=args.model, start_checkpoint=str(args.checkpoint.resolve()), episodes_budget=args.episodes,
                    envs=args.envs, episodes_per_iter=args.episodes_per_iter, execute=args.execute, policy=policy_config, ppo=ppo.as_dict(),
                    reward=reward.as_dict(), amp=amp, rl_suites=str(args.rl_suites.resolve()), scenario_order_sha256=pool.order_sha256,
                    train_poses=len(pool), monitor_poses=len(monitor), heldout_suite=heldout.suite_id, scene_dependencies_sha256=scene_hash,
                    seed=args.seed, rehearsal=bool(args.rehearsal), references=tracking.REFERENCES, tranche=TRANCHE_TAG)
    identity_path = root / "identity.json"
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        if {k: v for k, v in old.items() if k != "envs"} != {k: v for k, v in identity.items() if k != "envs"}:
            raise RuntimeError("output dir holds a different run (identity mismatch); use a new --output-dir")
    else:
        atomic_json(identity_path, identity)

    state = dict(status="running", phase="load", started_at=time.time(), model=args.model, episodes_target=args.episodes, episodes_done=0)
    if (root / "progress.json").exists():
        old = json.loads((root / "progress.json").read_text())
        state.update({k: old[k] for k in ("wandb_id", "wandb_url", "start_monitor", "heldout", "monitor_history", "aborted") if k in old})

    def progress(**values):
        state.update(values); state["updated_at"] = time.time(); atomic_json(root / "progress.json", state)

    family = f"rl-{args.model}"
    name = tracking.run_name(family, rehearsal=True) if args.rehearsal else (args.run_name or tracking.run_name(
        family, *([f"K{args.steps}"] if args.model != "act" else []), f"n{args.episodes}", scene_hash=scene_hash, seed=args.seed))
    tracker = tracking.init_run(family=family, name=name, config=identity, job_type="rl", rehearsal=args.rehearsal, run_id=state.get("wandb_id"),
                                axes=(tracking.EPISODES, tracking.STEP), default_axis=tracking.EPISODES, enabled=not args.no_wandb,
                                tags=[args.model, "reinflow" if args.model != "act" else "gaussian-ppo", f"n{args.episodes}", TRANCHE_TAG], directory=root)
    tracking.persist_run_id(root / "wandb.json", tracker)
    progress(wandb_id=tracker.id, wandb_url=tracker.url)
    tracker.define("01_outcome/monitor_success_rate", step_metric=tracking.EPISODES, summary="max")

    def phase(name_, episodes):
        progress(phase=name_)
        tracker.phase(name_, PHASES, episodes, axis=tracking.EPISODES)

    vec = None
    try:
        phase("load", state["episodes_done"])
        critic = make_critic(PRIVILEGED_DIM).to(model.device)
        named = policy.named_parameters()
        weights = MasterWeights([p for _, p in named])
        actor_opt = torch.optim.AdamW([dict(params=weights.master, lr=args.actor_lr)], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=0.0)
        scratch = root / "scratch.pt"
        iteration = 0
        if scratch.exists():
            saved = torch.load(scratch, map_location="cpu", weights_only=False)
            with torch.no_grad():
                for m, t in zip(weights.master, saved["master"]):
                    m.copy_(t.to(m.device))
            weights.push()
            critic.load_state_dict(saved["critic"]); actor_opt.load_state_dict(saved["actor_opt"]); critic_opt.load_state_dict(saved["critic_opt"])
            iteration = int(saved["iteration"]); state["episodes_done"] = int(saved["episodes_done"])
            rng = np.random.default_rng(saved["rng_seed"])
            print(f"RL resumed at iteration {iteration}, {state['episodes_done']} episodes", flush=True)
            del saved

        def on_step(when: str) -> None:
            if when == "pre":
                weights.pull_grads()
            else:
                weights.push()

        vec = SubprocChunkVecEnv(args.scene, args.envs, reward_kwargs=reward.as_dict())
        if any(info["torch_imported"] for info in vec.info):
            print("warning: env workers imported torch", flush=True)
        collector = Collector(vec, policy, execute=args.execute, batch_fraction=args.batch_fraction, watchdog_s=args.watchdog)

        def run_monitor(episodes_done: int) -> dict:
            _, summaries, _ = collector.collect([(i, s) for i, s in enumerate(monitor)], deterministic=True, seed=10_000_000)
            m = summarize(summaries)
            tracker.log({tracking.EPISODES: episodes_done, "01_outcome/monitor_success_rate": m["success_rate"],
                         "01_outcome/monitor_pickup_rate": m["pickup_rate"], "01_outcome/monitor_grasp_rate": m["grasp_rate"],
                         "01_outcome/monitor_invalidation_rate": m["invalidation_rate"],
                         "01_outcome/monitor_closure_within_8mm_rate": m["closure_within_8mm_rate"],
                         "01_outcome/monitor_closure_lateral_median_mm": m["closure_lateral_median_mm"],
                         "01_outcome/ref_start_success_rate": (state.get("start_monitor") or m)["success_rate"]})
            history = state.get("monitor_history", []) + [dict(m, episodes=episodes_done)]
            progress(monitor_history=history)
            print(f"RL monitor @ {episodes_done}: success {m['success_rate']:.2f} pickup {m['pickup_rate']:.2f} "
                  f"closure median {m['closure_lateral_median_mm']:.1f} mm", flush=True)
            return m

        if "start_monitor" not in state:
            phase("warmup", 0)
            progress(start_monitor=run_monitor(0))
        start_rate = state["start_monitor"]["success_rate"]
        zero_streak, kl_streak = 0, 0
        next_monitor = (state["episodes_done"] // args.monitor_every + 1) * args.monitor_every
        run_started = time.time()
        episodes_at_start = state["episodes_done"]
        phase("train", state["episodes_done"])
        metrics_path = root / "metrics.jsonl"
        while state["episodes_done"] < args.episodes:
            first = state["episodes_done"]
            count = min(args.episodes_per_iter, args.episodes - first)
            t0 = time.time()
            batch, summaries, timing = collector.collect([(first + i, pool.episode(first + i)) for i in range(count)], deterministic=False,
                                                         seed=args.seed * 1_000_003 + first)
            collect_s = time.time() - t0
            arrays = batch.arrays()
            frames, states_, records = arrays["frames"], arrays["states"], arrays["records"]

            def logprob_fn(idx, _f=frames, _s=states_, _r=records):
                idx = np.asarray(idx)
                logp, _ = policy.logprob(_f[idx], _s[idx], [_r[i] for i in idx])
                return logp

            t1 = time.time()
            stats = ppo_update(batch=arrays, logprob_fn=logprob_fn, actor_opt=actor_opt, critic=critic, critic_opt=critic_opt, config=ppo,
                               train_actor=iteration >= args.critic_warmup, on_step=on_step, rng=rng)
            update_s = time.time() - t1
            with torch.no_grad():
                _, sigma_stats = policy.logprob(frames[:4], states_[:4], records[:4])
            iteration += 1
            state["episodes_done"] = first + count
            done = state["episodes_done"]
            train = summarize(summaries)
            elapsed = time.time() - run_started
            rate = (done - episodes_at_start) / max(elapsed, 1e-9) * 3600
            payload = {tracking.EPISODES: done, tracking.STEP: iteration,
                       "01_outcome/train_success_rate": train["success_rate"], "01_outcome/train_pickup_rate": train["pickup_rate"],
                       "01_outcome/train_grasp_rate": train["grasp_rate"], "01_outcome/train_invalidation_rate": train["invalidation_rate"],
                       "01_outcome/train_safety_frames_mean": train["safety_frames_mean"],
                       "03_training/return_mean": train["return_mean"],
                       "03_training/reward_success": float(np.mean(batch.reward_success)) if len(batch) else 0.0,
                       "03_training/reward_safety": float(np.mean(batch.reward_safety)) if len(batch) else 0.0,
                       "03_training/reward_shaping": float(np.mean(batch.reward_shaping)) if len(batch) else 0.0,
                       "03_training/kl_target": args.target_kl, "03_training/lr_actor": actor_opt.param_groups[0]["lr"], "03_training/lr_critic": args.critic_lr,
                       "03_training/chunks": len(batch), "03_training/logprob_rollout_per_dim": float(np.mean(arrays["logp_rollout"])) / (90 * 6),
                       **{f"03_training/{k}": v for k, v in stats.items() if isinstance(v, float) and math.isfinite(v)},
                       **{f"03_training/{k}": v for k, v in sigma_stats.items()},
                       "04_throughput/episodes_per_hour": rate, "04_throughput/env_seconds": collect_s - timing["policy_seconds"],
                       "04_throughput/policy_seconds": timing["policy_seconds"], "04_throughput/update_seconds": update_s,
                       "04_throughput/mean_inference_batch": timing["mean_batch"],
                       "04_throughput/gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9, "04_throughput/host_rss_gb": rss_gb(),
                       "04_throughput/elapsed_hours": elapsed / 3600, "04_throughput/eta_hours": (args.episodes - done) / max(rate, 1e-9),
                       "05_phases/episodes_done": done, "05_phases/episodes_target": args.episodes}
            if "logprob_old_mean" in stats:
                payload["03_training/logprob_old_per_dim"] = stats["logprob_old_mean"] / (90 * 6)
            tracker.log(payload)
            with open(metrics_path, "a") as handle:
                handle.write(json.dumps(dict(iteration=iteration, episodes=done, train=train, stats=stats, sigma=sigma_stats,
                                             collect_seconds=collect_s, update_seconds=update_s), default=float) + "\n")
            print(f"RL it {iteration} ep {done}/{args.episodes}: train success {train['success_rate']:.2f} pickup {train['pickup_rate']:.2f} "
                  f"return {train['return_mean']:.3f} kl {stats.get('kl', float('nan')):.4f} clip {stats.get('clip_fraction', float('nan')):.2f} "
                  f"vloss {stats['value_loss']:.4f} | {collect_s:.0f}+{update_s:.0f} s, {rate:.0f} ep/h", flush=True)
            # checkpoint every iteration (small: trainable heads, critic, optimisers)
            tmp = scratch.with_suffix(".tmp")
            torch.save(dict(iteration=iteration, episodes_done=done, master=[m.detach().cpu() for m in weights.master], critic=critic.state_dict(),
                            actor_opt=actor_opt.state_dict(), critic_opt=critic_opt.state_dict(), rng_seed=int(rng.integers(2**31))), tmp)
            tmp.replace(scratch)
            progress(episodes_done=done, iteration=iteration)
            # abort guards
            if not all(math.isfinite(float(v)) for v in (stats["value_loss"], stats.get("policy_loss", 0.0), stats.get("kl", 0.0))):
                raise FloatingPointError("non-finite PPO statistics")
            kl_streak = kl_streak + 1 if stats.get("kl", 0.0) > 10 * args.target_kl else 0
            if kl_streak >= 3:
                progress(aborted="kl_diverged")
                break
            if done >= next_monitor or done >= args.episodes:
                m = run_monitor(done)
                next_monitor += args.monitor_every
                zero_streak = zero_streak + 1 if (m["success_rate"] == 0.0 and start_rate > 0.0) else 0
                if zero_streak >= 3:
                    progress(aborted="monitor_collapsed")
                    break
        vec.close()
        vec = None

        # ------------------------------------------------------------------ final model + held-out
        final_path = root / "model" / f"{args.model}_rl.pt"
        extra = dict(identity=identity, rl=dict(policy=policy.state_dict(), episodes_done=state["episodes_done"], aborted=state.get("aborted")))
        model.save(final_path, extra)
        phase("heldout", state["episodes_done"])
        from vla_finetune import closed_loop, log_outcome, summary_figure
        import contextlib
        outcomes = state.get("heldout", {})
        steps = args.steps if args.model != "act" else None
        for execute in args.execs:
            key = str(execute)
            if key not in outcomes:
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
                with ctx:
                    outcomes[key] = closed_loop(args.scene, heldout, model, execute, root / "evaluations" / f"exec{execute}", num_steps=steps,
                                                tag=f"{args.model}_rl")
                progress(heldout=outcomes)
            r = outcomes[key]
            print(f"RL held-out exec {execute}: {r['successes']}/{r['rollouts']} success, {r['pickups']} pickups, safety {r['safety_frames']}, "
                  f"closure median {r['closure'].get('lateral_median_mm', float('nan')):.1f} mm, {r['categories']}", flush=True)
            log_outcome(tracker, r, state["episodes_done"], axis=tracking.EPISODES)
        summary_figure(root / "summary.png", f"{args.model} RL", root / "__none__", None, outcomes)
        tracker.image("01_outcome/summary_figure", root / "summary.png")
        best = max(outcomes.values(), key=lambda r: r["successes"])
        tracker.summary({"01_outcome/best_successes": best["successes"], "01_outcome/best_execute": best["execute"]})
        atomic_json(root / "result.json", dict(identity=identity, heldout=outcomes, start_monitor=state["start_monitor"],
                                               monitor_history=state.get("monitor_history", []), aborted=state.get("aborted"),
                                               wandb=dict(id=tracker.id, url=tracker.url)))
        phase("complete", state["episodes_done"])
        tracker.finish()
        progress(status="complete" if not state.get("aborted") else "aborted", phase="complete")
        return 0
    except BaseException as exc:
        import traceback
        progress(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    raise SystemExit(main())
