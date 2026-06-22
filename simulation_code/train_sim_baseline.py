#!/usr/bin/env python3
"""
Simple privileged-state PPO baseline for the SO-101 MuJoCo pickup task.

This is a diagnostic baseline, not the deployment policy. It deliberately avoids
images and ACT so we can verify the reward, reset distribution, and action units
before spending more time on visuomotor ACT training.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

np = None
torch = None
nn = None
F = None
wandb = None
SO101PickPlaceEnv = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps-per-episode", type=int, default=250)
    parser.add_argument("--steps-per-action", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--delta-scale", type=float, default=0.08, help="Max non-gripper joint target delta in radians.")
    parser.add_argument("--gripper-delta-scale", type=float, default=0.08, help="Max gripper target delta in radians.")
    parser.add_argument("--log-std-init", type=float, default=-0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--randomize-block-reset", action="store_true")
    parser.add_argument(
        "--curriculum-fixed-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the easier fixed block pose by default.",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B project so101-sim-baseline.")
    parser.add_argument("--checkpoint-path", type=Path, default=SCRIPT_DIR / "sim_baseline_ppo_checkpoint.pt")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    return parser.parse_args()


def load_dependencies() -> None:
    global np, torch, nn, F, wandb, SO101PickPlaceEnv
    try:
        import numpy as _np
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
    except ImportError as exc:
        raise RuntimeError("train_sim_baseline.py requires numpy and torch in the LeRobot/simulation environment.") from exc

    from mujoco_rendering import setup_mujoco_rendering

    setup_mujoco_rendering()
    try:
        import wandb as _wandb
    except ImportError:
        _wandb = None

    from so101_gym_env import SO101PickPlaceEnv as _SO101PickPlaceEnv

    np = _np
    torch = _torch
    nn = _nn
    F = _F
    wandb = _wandb
    SO101PickPlaceEnv = _SO101PickPlaceEnv


@dataclass
class Rollout:
    obs: object
    raw_actions: object
    old_log_probs: object
    rewards: object
    dones: object
    values: object


class ActorCritic(nn.Module if nn is not None else object):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, log_std_init: float):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def distribution(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def define_model_classes() -> None:
    global ActorCritic

    class _ActorCritic(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, log_std_init: float):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        def distribution(self, obs):
            mean = self.actor(obs)
            std = self.log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)

        def value(self, obs):
            return self.critic(obs).squeeze(-1)

    ActorCritic = _ActorCritic


def privileged_obs(obs: dict, info: dict, max_steps: int, device) -> object:
    state = obs["observation.state"].astype(np.float32)
    gripper = np.asarray(info["gripper_pos"], dtype=np.float32)
    block = np.asarray(info["block_pos"], dtype=np.float32)
    features = np.concatenate(
        [
            state,
            gripper,
            block,
            np.asarray(
                [
                    info["distance_to_block"],
                    info["block_height"],
                    float(info.get("success", False)),
                    float(info.get("step_count", 0)) / float(max_steps),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return torch.from_numpy(features).to(device)


def raw_to_env_action(raw_action, current_state, env, args):
    scales = torch.tensor(
        [args.delta_scale] * 5 + [args.gripper_delta_scale],
        dtype=current_state.dtype,
        device=current_state.device,
    )
    delta = torch.tanh(raw_action) * scales
    target = current_state + delta
    low = torch.from_numpy(env.joint_limits_low).to(target.device)
    high = torch.from_numpy(env.joint_limits_high).to(target.device)
    return torch.clamp(target, low, high)


def collect_rollout(env, model, device, args) -> tuple[Rollout, dict]:
    obs, info = env.reset()
    obs_tensors = []
    raw_actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    metrics = {
        "return": 0.0,
        "success": 0.0,
        "steps": 0,
        "contact_steps": 0,
        "grasp_steps": 0,
        "lift_steps": 0,
        "max_block_height": float(info["block_height"]),
        "final_distance": float(info["distance_to_block"]),
    }

    for _ in range(args.max_steps_per_episode):
        obs_tensor = privileged_obs(obs, info, args.max_steps_per_episode, device)
        dist = model.distribution(obs_tensor.unsqueeze(0))
        raw_action = dist.rsample().squeeze(0)
        log_prob = dist.log_prob(raw_action.unsqueeze(0)).sum(dim=-1).squeeze(0)
        value = model.value(obs_tensor.unsqueeze(0)).squeeze(0)

        env_action = raw_to_env_action(raw_action, obs_tensor[:6], env, args).detach().cpu().numpy()
        total_reward = 0.0
        done = False
        for _repeat in range(args.steps_per_action):
            obs, reward, terminated, truncated, info = env.step(env_action)
            total_reward += float(reward)
            metrics["steps"] += 1
            metrics["contact_steps"] += int(bool(info.get("contacted", False)))
            metrics["grasp_steps"] += int(bool(info.get("gripped", False)))
            metrics["lift_steps"] += int(bool(info.get("block_lifted", False)))
            metrics["max_block_height"] = max(metrics["max_block_height"], float(info.get("block_height", 0.0)))
            metrics["final_distance"] = float(info.get("distance_to_block", metrics["final_distance"]))
            metrics["success"] = float(info.get("success", False))
            done = bool(terminated or truncated)
            if done:
                break

        obs_tensors.append(obs_tensor)
        raw_actions.append(raw_action.detach())
        log_probs.append(log_prob.detach())
        rewards.append(total_reward)
        dones.append(float(done))
        values.append(value.detach())
        metrics["return"] += total_reward
        if done:
            break

    return (
        Rollout(
            obs=torch.stack(obs_tensors).to(device),
            raw_actions=torch.stack(raw_actions).to(device),
            old_log_probs=torch.stack(log_probs).to(device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device),
            values=torch.stack(values).float().to(device),
        ),
        metrics,
    )


def returns_advantages(rewards, dones, values, gamma: float, gae_lambda: float):
    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, device=rewards.device)
    next_value = torch.tensor(0.0, device=rewards.device)
    for t in reversed(range(rewards.numel())):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, advantages


def update(model, actor_opt, critic_opt, rollout: Rollout, args) -> dict:
    returns, advantages = returns_advantages(rollout.rewards, rollout.dones, rollout.values, args.gamma, args.gae_lambda)
    indices = torch.arange(rollout.rewards.numel(), device=rollout.rewards.device)
    stats = {"policy_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    updates = 0
    for _ in range(args.ppo_epochs):
        perm = indices[torch.randperm(indices.numel(), device=indices.device)]
        for start in range(0, indices.numel(), args.minibatch_size):
            mb = perm[start : start + args.minibatch_size]
            dist = model.distribution(rollout.obs[mb])
            new_log_probs = dist.log_prob(rollout.raw_actions[mb]).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            ratio = torch.exp(torch.clamp(new_log_probs - rollout.old_log_probs[mb], -20.0, 20.0))
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages[mb]
            policy_loss = -torch.min(unclipped, clipped).mean() - args.entropy_coef * entropy

            values = model.value(rollout.obs[mb])
            critic_loss = F.mse_loss(values, returns[mb])

            actor_opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], 1.0)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            critic_opt.step()

            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss.item())
                stats["critic_loss"] += float(critic_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float((rollout.old_log_probs[mb] - new_log_probs).mean().item())
                stats["clip_fraction"] += float(((ratio - 1.0).abs() > args.clip_epsilon).float().mean().item())
            updates += 1
    return {key: value / max(1, updates) for key, value in stats.items()}


def evaluate(env, model, device, args) -> dict:
    returns = []
    successes = []
    contacts = []
    lifts = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_episodes):
            obs, info = env.reset()
            total = 0.0
            contact_count = 0
            lift_count = 0
            success = False
            for _step in range(args.max_steps_per_episode):
                obs_tensor = privileged_obs(obs, info, args.max_steps_per_episode, device)
                mean = model.actor(obs_tensor.unsqueeze(0)).squeeze(0)
                env_action = raw_to_env_action(mean, obs_tensor[:6], env, args).detach().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(env_action)
                total += float(reward)
                contact_count += int(bool(info.get("contacted", False)))
                lift_count += int(bool(info.get("block_lifted", False)))
                success = success or bool(info.get("success", False))
                if terminated or truncated:
                    break
            returns.append(total)
            successes.append(float(success))
            contacts.append(contact_count)
            lifts.append(lift_count)
    if was_training:
        model.train()
    return {
        "eval/return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval/contact_steps_mean": float(np.mean(contacts)) if contacts else 0.0,
        "eval/lift_steps_mean": float(np.mean(lifts)) if lifts else 0.0,
    }


def make_env(args):
    block_pos = (0.0, 0.24, 0.0125) if args.curriculum_fixed_block else (0.0, 0.3, 0.0125)
    env = SO101PickPlaceEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_steps_per_episode,
        randomize_block=args.randomize_block_reset,
        task_instruction="pick up the block",
    )
    if not args.randomize_block_reset:
        original_reset = env.reset

        def reset_with_block(*reset_args, **reset_kwargs):
            options = reset_kwargs.pop("options", None) or {}
            options.setdefault("block_pos", block_pos)
            return original_reset(*reset_args, options=options, **reset_kwargs)

        env.reset = reset_with_block
    return env


def save_checkpoint(path: Path, episode: int, model, actor_opt, critic_opt, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model": model.state_dict(),
            "actor_optimizer": actor_opt.state_dict(),
            "critic_optimizer": critic_opt.state_dict(),
            "config": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model, actor_opt, critic_opt, device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    actor_opt.load_state_dict(checkpoint["actor_optimizer"])
    critic_opt.load_state_dict(checkpoint["critic_optimizer"])
    return int(checkpoint.get("episode", -1)) + 1


def main() -> int:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    load_dependencies()
    define_model_classes()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is unavailable; baseline PPO will run on CPU.", file=sys.stderr)

    env = make_env(args)
    model = ActorCritic(obs_dim=16, action_dim=6, hidden_dim=args.hidden_dim, log_std_init=args.log_std_init).to(device)
    actor_opt = torch.optim.Adam(list(model.actor.parameters()) + [model.log_std], lr=args.actor_lr)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=args.critic_lr)

    start_episode = 0
    if args.resume is not None:
        start_episode = load_checkpoint(args.resume, model, actor_opt, critic_opt, device)

    use_wandb = args.wandb and wandb is not None
    if use_wandb:
        wandb.init(project="so101-sim-baseline", config=vars(args))

    print("Starting privileged-state SO-101 MuJoCo PPO baseline")
    print(f"  checkpoint_path: {args.checkpoint_path}")
    print(f"  fixed_block: {not args.randomize_block_reset}, curriculum_fixed_block: {args.curriculum_fixed_block}")

    try:
        for episode in range(start_episode, args.episodes):
            start = time.time()
            rollout, rollout_metrics = collect_rollout(env, model, device, args)
            train_metrics = update(model, actor_opt, critic_opt, rollout, args)
            metrics = {
                "episode": episode,
                "rollout/return": rollout_metrics["return"],
                "rollout/success": rollout_metrics["success"],
                "rollout/steps": rollout_metrics["steps"],
                "rollout/contact_steps": rollout_metrics["contact_steps"],
                "rollout/grasp_steps": rollout_metrics["grasp_steps"],
                "rollout/lift_steps": rollout_metrics["lift_steps"],
                "rollout/final_distance": rollout_metrics["final_distance"],
                "rollout/max_block_height": rollout_metrics["max_block_height"],
                "train/log_std_mean": float(model.log_std.mean().item()),
                "time/episode_seconds": time.time() - start,
                **{f"train/{key}": value for key, value in train_metrics.items()},
            }
            if episode % args.eval_every == 0:
                metrics.update(evaluate(env, model, device, args))
                save_checkpoint(args.checkpoint_path, episode, model, actor_opt, critic_opt, args)
            if use_wandb:
                wandb.log(metrics, step=episode)
            print(
                f"[{episode:04d}] return={metrics['rollout/return']:.3f} "
                f"success={metrics['rollout/success']:.0f} "
                f"contact={metrics['rollout/contact_steps']} "
                f"grasp={metrics['rollout/grasp_steps']} "
                f"lift={metrics['rollout/lift_steps']} "
                f"critic_loss={metrics['train/critic_loss']:.3f}"
            )
    finally:
        save_checkpoint(args.checkpoint_path, max(start_episode, args.episodes - 1), model, actor_opt, critic_opt, args)
        env.close()
        if use_wandb:
            wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"train_sim_baseline.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
