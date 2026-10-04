"""PPO over chunk decisions, with the legacy stack's fixes carried over (retry tranche, 2026-10-04).

- GAE is computed per episode: a chunk bootstraps only from the next chunk of the same episode
  (notes/parallel-gae-trajectory-identity-fix.md: the flattened-batch shift mixed environments).
- The ratio's denominator is recomputed through the exact microbatched path the update uses, before the first
  step, so the ratio starts at 1 up to the kernels' own noise; the gap to the rollout's log-prob is logged as a
  consistency check (README "Legacy RL stack status").
- The KL that stops an epoch early is measured after each optimiser step on that minibatch (post-update KL),
  not estimated before it.
- The base model stays in eval mode throughout (notes/kl-divergence-bug-fix.md: dropout inside the wrapped
  model turned the same sample into a different log-prob, KL ~3e8).
- The actor's learning rate adapts to the measured KL (new, 2026-10-04): the RL runs start from fine-tunes that
  do not exist until the queue trains them, so their gradient scale cannot be tuned beforehand; a chunk of
  90 x 6 actions makes the KL very sensitive to the rate (a rehearsal ACT update reached KL ~1500).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99            # per chunk decision
    lam: float = 0.95
    clip: float = 0.05
    epochs: int = 1
    minibatch: int = 32
    micro: int = 8
    target_kl: float = 0.1
    actor_lr: float = 1e-6
    critic_lr: float = 1e-4
    critic_epochs: int = 4
    grad_clip: float = 0.25
    critic_warmup_iterations: int = 10
    normalize_advantages: bool = True
    adaptive_lr: bool = True       # KL-adaptive actor rate: / 1.5 above 2 x target, x 1.5 below target / 2
    lr_floor_factor: float = 0.01  # bounds relative to actor_lr
    lr_ceiling_factor: float = 10.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gae(rewards: Sequence[float], values: Sequence[float], *, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """One terminated episode (no bootstrap after the last chunk): (advantages, returns)."""
    rewards = np.asarray(rewards, np.float64)
    values = np.asarray(values, np.float64)
    advantages = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, advantages + values


def episode_gae(episode_ids: Sequence[int], rewards: Sequence[float], values: Sequence[float], *, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE over a flat batch whose samples are grouped by episode id (in time order within each episode)."""
    episode_ids = np.asarray(episode_ids)
    rewards = np.asarray(rewards, np.float64)
    values = np.asarray(values, np.float64)
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    for episode in np.unique(episode_ids):
        where = np.flatnonzero(episode_ids == episode)
        a, r = gae(rewards[where], values[where], gamma=gamma, lam=lam)
        advantages[where], returns[where] = a, r
    return advantages, returns


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var = float(np.var(returns))
    return float("nan") if var <= 1e-12 else 1.0 - float(np.var(returns - values)) / var


def clipped_surrogate(new_logp: Any, old_logp: Any, advantages: Any, clip: float) -> tuple[Any, dict[str, float]]:
    import torch
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages).mean()
    with torch.no_grad():
        stats = dict(clip_fraction=float(((ratio - 1.0).abs() > clip).float().mean()),
                     approx_kl=float(((ratio - 1.0) - log_ratio).mean()), ratio_max=float(ratio.max()), ratio_min=float(ratio.min()))
    return loss, stats


def ppo_update(*, batch: dict[str, Any], logprob_fn: Callable[[np.ndarray], Any], actor_opt: Any,
               critic: Any, critic_opt: Any, config: PPOConfig, train_actor: bool, on_step: Callable[[str], None] | None = None,
               rng: np.random.Generator) -> dict[str, float]:
    """One PPO iteration on a collected batch.

    ``batch``: ``episode`` (N,), ``reward`` (N,), ``privileged`` (N, D), ``logp_rollout`` (N,) and whatever
    ``logprob_fn(indices)`` needs (it returns the (len(indices),) log-probs with grad, microbatched inside).
    ``on_step("pre")`` runs between backward and the clip/step (move grads to fp32 master weights), ``on_step("post")``
    after the step (copy masters back into a bf16 model).
    """
    import torch
    device = next(critic.parameters()).device
    privileged = torch.as_tensor(np.asarray(batch["privileged"]), dtype=torch.float32, device=device)
    with torch.no_grad():
        values = critic(privileged).cpu().numpy()
    advantages, returns = episode_gae(batch["episode"], batch["reward"], values, gamma=config.gamma, lam=config.lam)
    n = len(advantages)
    stats: dict[str, float] = dict(advantage_mean=float(advantages.mean()), advantage_std=float(advantages.std()),
                                   explained_variance=explained_variance(values, returns), value_mean=float(values.mean()))
    if config.normalize_advantages and n > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Critic first (its own optimiser, several epochs; cheap).
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    losses = []
    for _ in range(config.critic_epochs):
        for chunk in np.array_split(rng.permutation(n), max(1, n // 64)):
            idx = torch.as_tensor(chunk, device=device)
            loss = torch.nn.functional.mse_loss(critic(privileged[idx]).squeeze(-1), returns_t[idx])
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            losses.append(float(loss))
    stats["value_loss"] = float(np.mean(losses))
    if not train_actor:
        stats.update(actor_frozen=1.0)
        return stats

    # Denominator through the update's own path (exact ratio start), and the rollout consistency gap.
    with torch.no_grad():
        old = torch.cat([logprob_fn(np.arange(s, min(s + config.minibatch, n))).detach() for s in range(0, n, config.minibatch)])
    rollout = torch.as_tensor(np.asarray(batch["logp_rollout"]), dtype=old.dtype, device=old.device)
    stats["logprob_consistency_max"] = float((old - rollout).abs().max())
    stats["logprob_old_mean"] = float(old.mean())
    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=old.device)
    kls, clips, grad_norms, policy_losses, ratio_max = [], [], [], [], []
    stopped = 0.0
    for _ in range(config.epochs):
        order = rng.permutation(n)
        for start in range(0, n, config.minibatch):
            idx = order[start:start + config.minibatch]
            new = logprob_fn(idx)
            loss, step_stats = clipped_surrogate(new, old[idx], adv_t[idx], config.clip)
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            if on_step is not None:
                on_step("pre")
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_([p for g in actor_opt.param_groups for p in g["params"]], config.grad_clip)))
            actor_opt.step()
            if on_step is not None:
                on_step("post")
            with torch.no_grad():
                after = logprob_fn(idx)
                log_ratio = after - old[idx]
                post_kl = float(((torch.exp(log_ratio) - 1.0) - log_ratio).mean())
            kls.append(post_kl); clips.append(step_stats["clip_fraction"]); policy_losses.append(float(loss)); ratio_max.append(step_stats["ratio_max"])
            if post_kl > 1.5 * config.target_kl:
                stopped = 1.0
                break
        if stopped:
            break
    stats.update(kl=float(np.mean(kls)), kl_max=float(np.max(kls)), clip_fraction=float(np.mean(clips)), policy_loss=float(np.mean(policy_losses)),
                 grad_norm=float(np.mean(grad_norms)), ratio_max=float(np.max(ratio_max)), early_stopped=stopped, minibatches=float(len(kls)))
    if config.adaptive_lr:
        lr = actor_opt.param_groups[0]["lr"]
        if stats["kl"] > 2.0 * config.target_kl:
            lr = max(lr / 1.5, config.actor_lr * config.lr_floor_factor)
        elif stats["kl"] < 0.5 * config.target_kl:
            lr = min(lr * 1.5, config.actor_lr * config.lr_ceiling_factor)
        for group in actor_opt.param_groups:
            group["lr"] = lr
        stats["lr_actor_next"] = float(lr)
    return stats


__all__ = ["PPOConfig", "clipped_surrogate", "episode_gae", "explained_variance", "gae", "ppo_update"]
