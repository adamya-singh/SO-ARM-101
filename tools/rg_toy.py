#!/usr/bin/env python
"""Recursive generalization in a toy pick-and-place, PPO from scratch (2026-11-22, notes/rg-toy-20261122.md).

A point gripper in a unit box moves a cube onto a target. No arm, no camera: the point is to study, with many seeds
and on the CPU, what a staged curriculum buys a policy trained from scratch by RL, before doing it in the arm sim.

World (vectorised numpy): gripper p = (x, y, z), open or closed; cube c on the table; target t = (x, y).
  action   4 numbers in [-1, 1]: a step of up to 0.05 in x, y, z, and the gripper (closed when > 0)
  grasp    the gripper closes within 0.04 of the cube in xy, below z = 0.05  -> the cube moves with the gripper
  release  the gripper opens -> the cube falls straight down onto the table
  input    p, gripper, cube, target, held flag: ABSOLUTE coordinates, so going to a place the policy never
           visited is something it has to have learned, not something the input hands it
Tasks, all with the same input (no task label), each the earlier one plus more:
  reach    get the open gripper low over the cube
  lift     grasp it and raise it to z >= 0.2
  place    start holding the cube in the air; put it on the target and let go
  full     pick the cube up and put it on the target; "full_sparse" pays only for success
Every task pays +1 at success and ends; the staged tasks and "full_dense" add potential shaping.

A condition is a list of phases; a phase trains for some steps on a mix of tasks, each with the region its cube,
target and start are drawn from ("all", "L": x < 0.5, "R": x >= 0.5). The evaluation runs the mean action on every
task and region, and two mid-episode probes on the full task (the held cube is knocked out; the cube moves).

  tools/rg_toy.py run --experiment e1 --seeds 5 --workers 8      # trains every condition, writes results.json
  tools/rg_toy.py report
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
DEFAULT_ROOT = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/rg_toy_20261122"
TASKS = ("reach", "lift", "place", "full_sparse", "full_dense")
STEP, GRASP_XY, GRASP_Z, LIFT_Z, PLACE_XY, HORIZON = 0.05, 0.04, 0.05, 0.2, 0.05, 80
Z_MAX = 0.5


def region_x(rng, n, region):
    lo, hi = dict(all=(0.0, 1.0), L=(0.0, 0.5), R=(0.5, 1.0))[region]
    return rng.uniform(lo + 0.03, hi - 0.03, n) if region != "all" else rng.uniform(0.03, 0.97, n)


class Env:
    """N copies of the world; each copy has its own task and region, redrawn from ``mix`` at every reset."""

    def __init__(self, n, mix, seed, probe=None):
        self.n, self.rng, self.probe = n, np.random.default_rng(seed), probe
        self.set_mix(mix)
        self.p = np.zeros((n, 3)); self.c = np.zeros((n, 3)); self.t = np.zeros((n, 2))
        self.closed = np.zeros(n, bool); self.held = np.zeros(n, bool); self.time = np.zeros(n, int)
        self.task = np.zeros(n, int); self.phi = np.zeros(n); self.probed = np.zeros(n, bool)
        self.reset(np.ones(n, bool))

    def set_mix(self, mix):          # mix: list of (task, cube region, target region, weight)
        self.mix = [(TASKS.index(t), rc, rt) for t, rc, rt, _ in mix]
        w = np.array([m[3] for m in mix], float)
        self.weights = w / w.sum()

    def reset(self, mask):
        idx = np.flatnonzero(mask)
        if not idx.size:
            return
        choice = self.rng.choice(len(self.mix), idx.size, p=self.weights)
        for k, (task, rc, rt) in enumerate(self.mix):
            i = idx[choice == k]
            m = i.size
            if not m:
                continue
            self.task[i] = task
            self.c[i] = np.stack([region_x(self.rng, m, rc), self.rng.uniform(0.03, 0.97, m), np.zeros(m)], 1)
            for _ in range(20):      # the target is at least 0.15 from the cube
                self.t[i] = np.stack([region_x(self.rng, m, rt), self.rng.uniform(0.03, 0.97, m)], 1)
                near = np.linalg.norm(self.t[i] - self.c[i, :2], axis=1) < 0.15
                if not near.any():
                    break
                i_near = i[near]
                self.c[i_near, 1] = self.rng.uniform(0.03, 0.97, i_near.size)
            self.p[i] = np.stack([region_x(self.rng, m, rc), self.rng.uniform(0, 1, m), self.rng.uniform(0, Z_MAX, m)], 1)
            self.closed[i] = False; self.held[i] = False
            if TASKS[task] == "place":       # starts in the air holding the cube
                self.p[i, 2] = self.rng.uniform(0.1, 0.4, m)
                self.c[i] = self.p[i]; self.closed[i] = True; self.held[i] = True
        self.time[idx] = 0; self.probed[idx] = False
        self.phi[idx] = self.potential()[idx]

    def obs(self):
        return np.concatenate([self.p * [2, 2, 4] - 1, self.closed[:, None] * 2.0 - 1, self.c * [2, 2, 4] - 1,
                               self.t * 2 - 1, self.held[:, None] * 2.0 - 1], 1).astype(np.float32)

    def success(self):
        over = (np.linalg.norm(self.p[:, :2] - self.c[:, :2], axis=1) < GRASP_XY) & (self.p[:, 2] < GRASP_Z)
        placed = ~self.held & (self.c[:, 2] == 0) & (np.linalg.norm(self.c[:, :2] - self.t, axis=1) < PLACE_XY)
        name = np.array(TASKS)[self.task]
        return np.where(name == "reach", over & ~self.closed & ~self.held, np.where(name == "lift", self.held & (self.c[:, 2] >= LIFT_Z), placed))

    def potential(self):
        """Shaping potential: distance to the cube before the grasp, height then distance to the target after it."""
        to_cube = np.linalg.norm(self.p - self.c * [1, 1, 0], axis=1)
        to_target = np.linalg.norm(self.c[:, :2] - self.t, axis=1)
        name = np.array(TASKS)[self.task]
        before = -to_cube
        held = 0.5 + np.minimum(self.c[:, 2], LIFT_Z) / LIFT_Z * 0.5
        carry = np.where((name == "lift"), held, held + (1.5 - to_target))
        dropped = np.where(name == "lift", before, np.where(to_target < PLACE_XY, 3.0, before))      # on the table again
        phi = np.where(self.held, carry, dropped)
        return np.where(name == "full_sparse", 0.0, phi)

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.p = np.clip(self.p + STEP * a[:, :3], [0, 0, 0], [1, 1, Z_MAX])
        close = a[:, 3] > 0
        grasp = close & ~self.closed & ~self.held & (np.linalg.norm(self.p[:, :2] - self.c[:, :2], axis=1) < GRASP_XY) \
            & (self.p[:, 2] < GRASP_Z) & (self.c[:, 2] == 0)
        self.held = (self.held | grasp) & close
        self.closed = close
        self.c = np.where(self.held[:, None], self.p, self.c * [1, 1, 0])
        self.time += 1
        if self.probe == "drop":          # the held cube is knocked out once it is in the air: the gripper is left closed and empty
            hit = self.held & (self.c[:, 2] >= 0.15) & ~self.probed
            self.held &= ~hit; self.c[hit, 2] = 0; self.probed |= hit
        elif self.probe == "shift":       # the cube jumps 0.2 along y at step 6, before the gripper gets there
            hit = (self.time == 6) & ~self.held & ~self.probed
            self.c[hit, 1] = np.where(self.c[hit, 1] > 0.5, self.c[hit, 1] - 0.2, self.c[hit, 1] + 0.2); self.probed |= hit
        done_ok = self.success()
        phi = self.potential()
        reward = (phi - self.phi) + done_ok * 1.0
        self.phi = phi
        done = done_ok | (self.time >= HORIZON)
        info = dict(success=done_ok & done, done=done, task=self.task.copy(), probed=self.probed.copy())
        self.reset(done)
        return self.obs(), reward.astype(np.float32), done, info


def make_nets(torch, obs_dim=10, act_dim=4, width=128):
    nn = torch.nn
    mlp = lambda out: nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, out))
    actor, critic = mlp(act_dim), mlp(1)
    log_std = nn.Parameter(torch.full((act_dim,), -0.5))
    return actor, critic, log_std


def evaluate(torch, actor, mix, seed, episodes=1000, probe=None):
    env = Env(episodes, mix, seed, probe=probe)
    ok = np.zeros(episodes, bool); alive = np.ones(episodes, bool); probed = np.zeros(episodes, bool)
    obs = env.obs()
    for _ in range(HORIZON):
        with torch.no_grad():
            action = actor(torch.from_numpy(obs)).numpy()
        obs, _, done, info = env.step(action)
        ok |= alive & info["success"]; probed |= alive & info["probed"]
        alive &= ~done
    if probe:
        return float(ok[probed].mean()) if probed.any() else None
    return float(ok.mean())


EVALS = {      # name -> mix
    "reach_L": [("reach", "L", "L", 1)], "reach_R": [("reach", "R", "R", 1)],
    "lift_L": [("lift", "L", "L", 1)], "lift_R": [("lift", "R", "R", 1)],
    "place_L": [("place", "L", "L", 1)], "place_R": [("place", "R", "R", 1)],
    "full_L": [("full_sparse", "L", "L", 1)], "full_R": [("full_sparse", "R", "R", 1)],
    "full_LtoR": [("full_sparse", "L", "R", 1)], "full_RtoL": [("full_sparse", "R", "L", 1)],
    "full_all": [("full_sparse", "all", "all", 1)],
}


def evaluate_all(torch, actor, seed):
    out = {name: evaluate(torch, actor, mix, seed + 1000 + i) for i, (name, mix) in enumerate(EVALS.items())}
    out["probe_drop_all"] = evaluate(torch, actor, EVALS["full_all"], seed + 2000, probe="drop")
    out["probe_shift_all"] = evaluate(torch, actor, EVALS["full_all"], seed + 2001, probe="shift")
    return out


def train(condition, seed, n_envs=256, rollout=64, eval_every=250_000, log=None):
    """PPO over the condition's phases; returns the evaluation curve (one row per ``eval_every`` steps and phase end)."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    actor, critic, log_std = make_nets(torch)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=3e-4)
    gamma, lam, clip, epochs, batch, ent = 0.98, 0.95, 0.2, 4, 2048, 0.003
    curve, total, started = [], 0, time.time()
    row = lambda phase: curve.append(dict(steps=total, phase=phase, **evaluate_all(torch, actor, seed)))
    row(-1)
    for k, phase in enumerate(condition["phases"]):
        env = Env(n_envs, phase["mix"], seed * 100 + k)
        obs = env.obs()
        next_eval = total + eval_every
        end = total + phase["steps"]
        while total < end:
            O, A, R, D, V, LP = [], [], [], [], [], []
            for _ in range(rollout):
                with torch.no_grad():
                    o = torch.from_numpy(obs)
                    mean, value = actor(o), critic(o).squeeze(1)
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    action = dist.sample()
                    lp = dist.log_prob(action).sum(1)
                O.append(o); A.append(action); V.append(value); LP.append(lp)
                obs, reward, done, _ = env.step(action.numpy())
                R.append(torch.from_numpy(reward)); D.append(torch.from_numpy(done.astype(np.float32)))
            with torch.no_grad():
                last = critic(torch.from_numpy(obs)).squeeze(1)
            adv, gae = [None] * rollout, torch.zeros(n_envs)
            for t in reversed(range(rollout)):          # a time-out is treated as an end (the horizon is in no input; fine for a toy)
                nxt = last if t == rollout - 1 else V[t + 1]
                delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]
                gae = delta + gamma * lam * (1 - D[t]) * gae
                adv[t] = gae
            O, A, LP = torch.cat(O), torch.cat(A), torch.cat(LP)
            adv = torch.cat(adv); ret = adv + torch.cat(V)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            n = O.shape[0]
            for _ in range(epochs):
                for idx in torch.randperm(n).split(batch):
                    dist = torch.distributions.Normal(actor(O[idx]), log_std.exp())
                    lp = dist.log_prob(A[idx]).sum(1)
                    ratio = (lp - LP[idx]).exp()
                    loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean() \
                        + 0.5 * (critic(O[idx]).squeeze(1) - ret[idx]).pow(2).mean() - ent * dist.entropy().sum(1).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()
            total += n
            if total >= next_eval and total < end:
                row(k); next_eval += eval_every
        row(k)
        if log:
            log(f"{condition['name']} seed {seed} phase {k} done at {total} steps: full_all {curve[-1]['full_all']:.2f} "
                f"full_L {curve[-1]['full_L']:.2f} full_R {curve[-1]['full_R']:.2f} ({time.time() - started:.0f} s)")
    return curve


M = 1_000_000
ALL3 = lambda region, w=1: [("reach", region, region, w), ("lift", region, region, w), ("place", region, region, w)]


def experiments():
    e1 = [      # does a staged curriculum make from-scratch RL work? whole workspace everywhere
        dict(name="flat_sparse", phases=[dict(mix=[("full_sparse", "all", "all", 1)], steps=6 * M)]),
        dict(name="flat_dense", phases=[dict(mix=[("full_dense", "all", "all", 1)], steps=6 * M)]),
        dict(name="staged", phases=[dict(mix=[("reach", "all", "all", 1)], steps=M), dict(mix=[("lift", "all", "all", 1)], steps=M),
                                    dict(mix=[("place", "all", "all", 1)], steps=M), dict(mix=[("full_sparse", "all", "all", 1)], steps=3 * M)]),
        dict(name="staged_replay", phases=[dict(mix=[("reach", "all", "all", 1)], steps=M),
                                           dict(mix=[("lift", "all", "all", 2), ("reach", "all", "all", 1)], steps=M),
                                           dict(mix=[("place", "all", "all", 2), ("lift", "all", "all", 1), ("reach", "all", "all", 1)], steps=M),
                                           dict(mix=[("full_sparse", "all", "all", 2)] + ALL3("all", 1 / 3), steps=3 * M)]),
        dict(name="staged_reversed", phases=[dict(mix=[("place", "all", "all", 1)], steps=M), dict(mix=[("lift", "all", "all", 1)], steps=M),
                                             dict(mix=[("reach", "all", "all", 1)], steps=M), dict(mix=[("full_sparse", "all", "all", 1)], steps=3 * M)]),
        dict(name="parts_then_full", phases=[dict(mix=ALL3("all"), steps=3 * M), dict(mix=[("full_sparse", "all", "all", 1)], steps=3 * M)]),
        dict(name="all_at_once", phases=[dict(mix=ALL3("all") + [("full_sparse", "all", "all", 1)], steps=6 * M)]),
    ]
    full_L = ("full_sparse", "L", "L", 1)
    e2 = [      # does the full task, trained on the left half only, inherit the reach of parts trained everywhere?
        dict(name="parts_all__full_L", phases=[dict(mix=ALL3("all"), steps=3 * M), dict(mix=[full_L], steps=3 * M)]),
        dict(name="parts_all__full_L_replay", phases=[dict(mix=ALL3("all"), steps=3 * M), dict(mix=[("full_sparse", "L", "L", 3)] + ALL3("all", 1), steps=3 * M)]),
        dict(name="parts_L__full_L", phases=[dict(mix=ALL3("L"), steps=3 * M), dict(mix=[full_L], steps=3 * M)]),
        dict(name="dense_full_L", phases=[dict(mix=[("full_dense", "L", "L", 1)], steps=6 * M)]),
        # how far up does it carry? only the lowest part (reach) is trained everywhere
        dict(name="reach_all_rest_L__full_L_replay", phases=[
            dict(mix=[("reach", "all", "all", 1), ("lift", "L", "L", 1), ("place", "L", "L", 1)], steps=3 * M),
            dict(mix=[("full_sparse", "L", "L", 3), ("reach", "all", "all", 1), ("lift", "L", "L", 1), ("place", "L", "L", 1)], steps=3 * M)]),
        dict(name="reach_lift_all_place_L__full_L_replay", phases=[
            dict(mix=[("reach", "all", "all", 1), ("lift", "all", "all", 1), ("place", "L", "L", 1)], steps=3 * M),
            dict(mix=[("full_sparse", "L", "L", 3), ("reach", "all", "all", 1), ("lift", "all", "all", 1), ("place", "L", "L", 1)], steps=3 * M)]),
    ]
    return dict(e1=e1, e2=e2)


def _job(task):
    condition, seed, out = task
    path = Path(out) / f"{condition['name']}__s{seed}.json"
    if path.exists():
        return str(path)
    log = lambda text: print(text, flush=True)
    curve = train(condition, seed, log=log)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(condition=condition, seed=seed, curve=curve)) + "\n")
    tmp.replace(path)
    return str(path)


def cmd_run(args):
    from multiprocessing import get_context
    sets = experiments()
    names = args.experiment or list(sets)
    out = args.root / "runs"
    out.mkdir(parents=True, exist_ok=True)
    conditions = [c for name in names for c in sets[name] if not args.only or c["name"] in args.only]
    for c in conditions:
        if args.scale != 1.0:
            for phase in c["phases"]:
                phase["steps"] = int(phase["steps"] * args.scale)
    tasks = [(c, seed, str(out)) for seed in range(args.seeds) for c in conditions]
    with get_context("spawn").Pool(args.workers) as pool:
        for path in pool.imap_unordered(_job, tasks):
            print(f"DONE {path}", flush=True)
    return cmd_report(args)


def cmd_report(args):
    runs: dict[str, list] = {}
    for path in sorted((args.root / "runs").glob("*__s*.json")):
        data = json.loads(path.read_text())
        runs.setdefault(data["condition"]["name"], []).append(data)
    if not runs:
        raise SystemExit("no runs")
    keys = ["full_all", "full_L", "full_R", "full_LtoR", "full_RtoL", "reach_R", "lift_R", "place_R", "probe_drop_all", "probe_shift_all"]
    table = {}
    print(f"{'condition':<40}{'seeds':>6}" + "".join(f"{k.replace('probe_', 'p_'):>12}" for k in keys) + "   steps to 0.8 full_all")
    for name, items in runs.items():
        final = {k: [r["curve"][-1][k] for r in items if r["curve"][-1][k] is not None] for k in keys}
        phase_ends = {}
        for r in items:          # the evaluation at the end of every phase, on the full task
            last = {}
            for row in r["curve"]:
                last[row["phase"]] = row
            for phase, row in last.items():
                phase_ends.setdefault(phase, []).append({k: row[k] for k in ("full_all", "full_L", "full_R")})
        to80 = []
        for r in items:
            hit = [row["steps"] for row in r["curve"] if row["full_all"] >= 0.8]
            to80.append(hit[0] if hit else None)
        table[name] = dict(seeds=len(items), final={k: dict(mean=float(np.mean(v)) if v else None, min=float(np.min(v)) if v else None,
                                                             max=float(np.max(v)) if v else None, values=v) for k, v in final.items()},
                           steps_to_0p8_full_all=to80,
                           phase_ends={str(p): {k: float(np.mean([x[k] for x in rows])) for k in ("full_all", "full_L", "full_R")} for p, rows in sorted(phase_ends.items())})
        print(f"{name:<40}{len(items):>6}" + "".join(f"{np.mean(final[k]):>12.2f}" if final[k] else f"{'-':>12}" for k in keys)
              + "   " + " ".join("-" if s is None else f"{s / 1e6:.1f}M" for s in to80))
    (args.root / "results.json").write_text(json.dumps(table, indent=2) + "\n")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--experiment", nargs="*", default=None)
    r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--scale", type=float, default=1.0, help="multiply every phase's steps (smoke tests)")
    r.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("report")
    t.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    t.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
