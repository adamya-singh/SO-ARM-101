"""Rollout collection over the vector env with one batched policy forward per group of chunk boundaries.

Environments simulate their 90-step chunks in worker processes while the GPU infers for the ones waiting at a
boundary: a batch is run once at least ``batch_fraction`` of the active environments wait (or nothing is left
in flight), so simulation and inference overlap. Each sample is one chunk decision: the boundary frame, the
7-dim state, the privileged critic input, the policy's record (ReinFlow chain or Gaussian action), its
rollout log-prob, and the chunk's reward once the environment answers.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from so_arm101_v2.simulation.residual_policy import chunk_state, residual_to_commands


@dataclass
class Batch:
    episode: list[int] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    privileged: list[np.ndarray] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    logp_rollout: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    reward_success: list[float] = field(default_factory=list)
    reward_safety: list[float] = field(default_factory=list)
    reward_shaping: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episode)

    def arrays(self) -> dict[str, Any]:
        return dict(episode=np.asarray(self.episode), frames=np.stack(self.frames), states=np.stack(self.states),
                    privileged=np.stack(self.privileged), records=self.records, logp_rollout=np.asarray(self.logp_rollout, np.float32),
                    reward=np.asarray(self.reward, np.float32))


class Collector:
    def __init__(self, vec: Any, policy: Any, *, execute: int = 90, batch_fraction: float = 0.5) -> None:
        self.vec, self.policy, self.execute, self.batch_fraction = vec, policy, int(execute), float(batch_fraction)
        self.inferences = 0

    def collect(self, episodes: list[tuple[int, Any]], *, deterministic: bool, seed: int) -> tuple[Batch, list[Any], dict[str, float]]:
        """Run every (episode id, scenario) once; returns the chunk samples, the episode summaries and timings."""
        vec = self.vec
        queue = deque(episodes)
        batch = Batch()
        summaries: list[Any] = []
        active: dict[int, dict[str, Any]] = {}
        waiting: list[tuple[int, dict[str, Any]]] = []
        timing = dict(policy_seconds=0.0, env_wait_seconds=0.0, batches=0, batch_size_sum=0)
        started = time.time()

        def start(env: int) -> None:
            episode_id, scenario = queue.popleft()
            active[env] = dict(id=episode_id, last=None)
            vec.send_reset(env, scenario)

        for env in range(vec.num_envs):
            if queue:
                start(env)
        while active:
            in_flight = len(vec.pending)
            threshold = max(1, int(np.ceil(len(active) * self.batch_fraction)))
            if waiting and (len(waiting) >= threshold or in_flight == 0):
                t0 = time.time()
                self._infer(waiting, active, batch, deterministic=deterministic, seed=seed)
                timing["policy_seconds"] += time.time() - t0
                timing["batches"] += 1
                timing["batch_size_sum"] += len(waiting)
                waiting = []
                continue
            t0 = time.time()
            ready = vec.ready()
            timing["env_wait_seconds"] += time.time() - t0
            for env in ready:
                payload = vec.recv(env)
                state = active[env]
                if state["last"] is None:                       # reset answer
                    waiting.append((env, payload))
                    continue
                observation, reward, done, info = payload
                index = state["last"]
                batch.reward[index] = reward["total"]
                batch.reward_success[index] = reward["success"]
                batch.reward_safety[index] = reward["safety"]
                batch.reward_shaping[index] = reward["shaping"]
                if done:
                    summaries.append(info["summary"])
                    del active[env]
                    if queue:
                        start(env)
                else:
                    waiting.append((env, observation))
        timing["wall_seconds"] = time.time() - started
        timing["mean_batch"] = timing["batch_size_sum"] / max(timing["batches"], 1)
        return batch, summaries, timing

    def _infer(self, waiting: list[tuple[int, dict[str, Any]]], active: dict[int, dict[str, Any]], batch: Batch, *, deterministic: bool, seed: int) -> None:
        frames = np.stack([obs["frame"] for _, obs in waiting])
        states = np.stack([chunk_state(obs["current_act"], obs["action_index"]) for _, obs in waiting])
        out = self.policy.act(frames, states, deterministic=deterministic, seed=seed + self.inferences)
        self.inferences += 1
        for k, (env, obs) in enumerate(waiting):
            commands = residual_to_commands(obs["current_act"], out["residual"][k][: self.execute])
            batch.episode.append(active[env]["id"])
            batch.frames.append(frames[k]); batch.states.append(states[k]); batch.privileged.append(obs["privileged"])
            batch.records.append(out["records"][k]); batch.logp_rollout.append(float(out["logp"][k]))
            batch.reward.append(0.0); batch.reward_success.append(0.0); batch.reward_safety.append(0.0); batch.reward_shaping.append(0.0)
            active[env]["last"] = len(batch.episode) - 1
            self.vec.send_step(env, commands)


def summarize(summaries: list[Any]) -> dict[str, float]:
    """Rates and means over finished episodes (the 01_outcome monitor/train keys)."""
    n = max(len(summaries), 1)
    closures = [s.closure for s in summaries if s.closure is not None]
    lat = np.asarray([c["lat"] for c in closures]) if closures else np.asarray([])
    descended = np.asarray([c["dz"] <= 6.0 for c in closures]) if closures else np.asarray([], bool)
    kinds: dict[str, float] = {}
    for s in summaries:
        if s.first_violation is not None:
            for kind in s.first_violation["kinds"] + [f"{k}_j{j}" for k in ("act_clip", "mujoco_clip", "relative_limit")
                                                      for j in s.first_violation.get(f"{k}_joints", [])]:
                kinds[f"violation_{kind}"] = kinds.get(f"violation_{kind}", 0.0) + 1.0
    first_chunk = [s for s in summaries if s.first_violation is not None and s.first_violation["action"] <= 90]
    return dict(kinds, violations_in_first_chunk=float(len(first_chunk)), episodes=float(len(summaries)), success_rate=sum(s.success for s in summaries) / n,
                pickup_rate=sum(s.pickup_completed for s in summaries) / n, grasp_rate=sum(s.strict_grasp_acquired for s in summaries) / n,
                invalidation_rate=sum(s.invalidated for s in summaries) / n, safety_frames_mean=float(np.mean([s.safety_frames for s in summaries])) if summaries else 0.0,
                return_mean=float(np.mean([s.return_total for s in summaries])) if summaries else 0.0,
                max_lift_mm_mean=1000 * float(np.mean([s.max_lift_m for s in summaries])) if summaries else 0.0,
                closure_lateral_median_mm=float(np.median(lat)) if lat.size else float("nan"),
                closure_within_8mm_rate=float(np.sum((lat <= 8.0) & descended)) / n if lat.size else 0.0)


__all__ = ["Batch", "Collector", "summarize"]
