#!/usr/bin/env python
"""PPO from scratch in the SO-101 arm sim, state inputs, staged tasks (2026-11-22, notes/rg-toy-20261122.md).

The arm-sim counterpart of tools/rg_toy.py, to find out what from-scratch RL costs here before spending a day on it:
no camera (the policy reads the joints and the cube, jaw and square positions from the simulator), the certified
command gate and physics, CPU only.

  input    6 joints, jaw point, cube, cube minus jaw, square centre, cube height gain, strict-grasp flag, time
  action   6 numbers in [-1, 1]: this action's joint target is the current joints plus action * 0.06 (act units)
  tasks    reach  the jaw point comes within 15 mm of a point 45 mm above the cube, without moving the cube
           lift   the cube is held in a strict two-sided grasp 30 mm above where it started
  reward   +5 at success (the episode ends); potential shaping on the distance to the hover point (reach) or to the
           cube, then on the height gain (lift); -0.02 for a held command, -0.5 and the end for an unsafe contact;
           the cube pushed more than 10 mm before a grasp ends a reach episode as a failure

A condition is a list of phases (task, steps), as in rg_toy. Training placements: the 7200 teacher-screened
placements of the perturbed-teacher capture's suite; evaluation: the 100 held-out probe placements (seed 31).

  tools/rg_arm.py run --condition reach --seed 0 --workers 8
  tools/rg_arm.py report
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
E = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DEFAULT_ROOT = E / "rg_arm_20261122"
SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml"
TRAIN_SUITE = E / "perturbed_so101_20261108/capture7200/train_suite.path"
EVAL_SUITE = E / "feature_probes_20261118/probe_suite.path"
TASKS = ("reach", "descend", "grasp", "lift")
# v2 (same day): the first lift runs stayed at 0. The gripper starts nearly closed (0.23) and the teacher opens it to 0.86
# before it descends, grasps at 0.19 with the jaw point 23 mm above the cube centre; the v1 reward pulled a closed gripper
# onto the cube. The four tasks are now prefixes of one another (toy lesson 1) under ONE potential, and differ only in
# where they end:  reach = open gripper at the hover point, descend = open gripper at the grasp point, grasp = strict
# two-sided grasp, lift = strict grasp 30 mm up.
HORIZON, ACTION_SCALE, HOVER_M, REACH_TOL_M, LIFT_M = 200, 0.06, 0.045, 0.015, 0.030
GRASP_Z_M, OPEN, CLOSED, DESCEND_TOL_M = 0.023, 0.70, 0.19, 0.008
SPLIT_Y = 0.0      # set from the training suite's median at import by split_y()
OBS_DIM = 6 + 3 + 3 + 3 + 2 + 1 + 1 + 1 + 1


class ArmEnv:
    def __init__(self, scenarios, seed):
        from so_arm101_v2.contracts.pick_place import load_pick_place_contract
        from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
        self.adapter = MujocoTaskAdapter(SCENE)
        self.contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=self.adapter.bench)
        self.rng = np.random.default_rng(seed)
        self.use(scenarios)
        self.task = "reach"
        self.order = None

    def use(self, scenarios):
        """Scenario pool and its halves: L = square centre left of the workspace midline (y below SPLIT_Y), R = the rest."""
        self.scenarios = scenarios
        y = np.array([s.square_center_xy[1] for s in scenarios])
        self.regions = dict(all=np.arange(len(scenarios)), L=np.flatnonzero(y < SPLIT_Y), R=np.flatnonzero(y >= SPLIT_Y))

    def jaw(self):
        d = self.adapter.data
        return 0.5 * (d.site("fixed_jaw_tip").xpos + d.site("moving_jaw_tip").xpos)

    def reset(self, task, index=None, region="all"):
        self.task = task
        if index is None:
            pool = self.regions[region]
            index = int(pool[self.rng.integers(len(pool))])
        scenario = self.scenarios[index]
        self.adapter.reset(scenario)
        self.t = 0
        self.cube0 = self.adapter.data.body("red_block").xpos.copy()
        self.gain, self.strict = 0.0, False
        self.phi = self.potential()
        return self.obs()

    def obs(self):
        a = self.adapter
        jaw, cube = self.jaw(), a.data.body("red_block").xpos
        square = np.asarray(a.square_center_xy, dtype=np.float64)
        return np.concatenate([a.current_act() / 3.0, jaw * 4, cube * 4, (cube - jaw) * 10, square * 4, [self.gain * 30],
                               [float(self.strict)], [self.t / HORIZON], [0.0]]).astype(np.float32)

    def geometry(self):
        jaw = self.jaw()
        xy = float(np.linalg.norm(jaw[:2] - self.cube0[:2]))
        funnel = self.cube0[2] + GRASP_Z_M + (HOVER_M - GRASP_Z_M) * min(xy / 0.03, 1.0)     # allowed height falls as the jaws centre
        return jaw, xy, float(abs(jaw[2] - funnel)), float(self.adapter.current_act()[5])

    def potential(self):
        """One potential for every task: centre over the cube with the gripper open, come down the funnel, close, lift."""
        if self.strict or self.gain > 0.005:
            return 6.0 + 60.0 * min(self.gain, LIFT_M)
        _, xy, dz, grip = self.geometry()
        approach = -10.0 * xy - 10.0 * dz
        at_cube = xy < 0.012 and dz < 0.012
        opened = 2.0 * min(grip, OPEN) / OPEN
        closing = 2.0 + 2.0 * (OPEN - float(np.clip(grip, CLOSED, OPEN))) / (OPEN - CLOSED)
        return approach + (closing if at_cube and self.task in ("grasp", "lift") else opened)

    def step(self, action):
        from so_arm101_v2.contracts.physical import _mask_reason, bench_hold_decision, physical_normalized_to_act
        a = self.adapter
        current = a.current_act()
        requested = current + np.clip(action, -1, 1).astype(np.float32) * ACTION_SCALE
        # The bench gate holds the WHOLE command when any joint asks for something out of bounds. For exploration that
        # freezes the arm at its limits, so the command is first clipped into the bounds (the gate still decides).
        decision = bench_hold_decision(current, requested, shoulder_floor=a.bench.shoulder_floor,
                                       joint_map=None if a.joint_map.legacy else a.joint_map, hold_on_any_mask=True)
        if decision.evaluation is not None and _mask_reason(decision.evaluation):
            # clipped into the bounds and rate limit (what the legacy lane would execute), then 0.5% inside the bound:
            # a joint resting exactly on its limit would otherwise mask again
            limited = physical_normalized_to_act(decision.evaluation.relative_limited_physical)
            requested = (limited * 0.995).astype(np.float32)
        command = a.apply_policy_command(requested)
        a.advance_control_period()
        m, _ = a.pick_place_measurement(command, footprint_edge_margin_m=self.contract.placement.footprint_edge_margin_m)
        pickup = m.pickup
        self.gain, self.strict = float(pickup.cube_height_gain_m), bool(pickup.strict_bilateral_grasp)
        self.t += 1
        held = bool(pickup.delta_limiter_activated or pickup.command_bound_violation)
        unsafe = bool(pickup.unsafe_contact)
        jaw, cube = self.jaw(), a.data.body("red_block").xpos
        pushed = float(np.linalg.norm(cube[:2] - self.cube0[:2]))
        grip = float(a.current_act()[5])
        grasp_point = self.cube0 + [0, 0, GRASP_Z_M]
        if self.task == "reach":
            success = float(np.linalg.norm(jaw - (self.cube0 + [0, 0, HOVER_M]))) < REACH_TOL_M and pushed < 0.005 and grip >= OPEN
        elif self.task == "descend":
            success = float(np.linalg.norm(jaw - grasp_point)) < DESCEND_TOL_M and pushed < 0.005 and grip >= OPEN
        elif self.task == "grasp":
            success = self.strict
        else:
            success = self.strict and self.gain >= LIFT_M
        failed = unsafe or (pushed > 0.010 and not self.strict and self.gain < 0.005)
        phi = self.potential()
        reward = (phi - self.phi) + 5.0 * success - 0.02 * held - 0.5 * failed
        self.phi = phi
        done = bool(success or failed or self.t >= HORIZON)
        return self.obs(), float(reward), done, dict(success=bool(success), unsafe=unsafe, held=held)


def _suite(pointer):
    from so_arm101_v2.simulation.suites import load_suite_from_path
    return load_suite_from_path(Path(Path(pointer).read_text().strip())).scenarios


def split_y(scenarios):
    return float(np.median([s.square_center_xy[1] for s in scenarios]))


def _worker(conn, seed, n_envs):
    import mujoco  # noqa: F401
    global SPLIT_Y
    train, held_out = _suite(TRAIN_SUITE), _suite(EVAL_SUITE)
    SPLIT_Y = split_y(train)
    envs = [ArmEnv(train, seed * 1000 + i) for i in range(n_envs)]
    rng = np.random.default_rng(seed + 7)
    mix = [("reach", 1.0, "all")]
    def draw():          # (task, region) for the next episode
        item = mix[int(rng.choice(len(mix), p=np.array([m[1] for m in mix]) / sum(m[1] for m in mix)))]
        return item[0], (item[2] if len(item) > 2 else "all")
    def fresh(e):
        task, region = draw()
        return e.reset(task, region=region)
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":          # payload: [(task, weight), ...]; each episode draws its task
            mix = payload
            conn.send(np.stack([fresh(e) for e in envs]))
        elif cmd == "step":
            out = []
            for e, a in zip(envs, payload):
                o, r, d, info = e.step(a)
                if d:
                    o = fresh(e)
                out.append((o, r, d, info["success"], info["unsafe"]))
            conn.send(out)
        elif cmd == "eval_reset":       # payload: (task, indices of held-out scenarios)
            task_eval, indices = payload
            for e in envs:
                e.use(held_out)
            conn.send(np.stack([e.reset(task_eval, index=i) for e, i in zip(envs, indices)]))
        elif cmd == "eval_step":
            conn.send([e.step(a) for e, a in zip(envs, payload)])
        elif cmd == "train_mode":
            for e in envs:
                e.use(train)
            conn.send(True)
        elif cmd == "close":
            return


class VecArm:
    def __init__(self, workers, envs_per_worker, seed):
        ctx = get_context("spawn")
        self.conns, self.procs, self.k = [], [], envs_per_worker
        for w in range(workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, seed * 100 + w, envs_per_worker), daemon=True)
            proc.start()
            self.conns.append(parent); self.procs.append(proc)
        self.n = workers * envs_per_worker

    def call(self, cmd, payloads):
        for conn, payload in zip(self.conns, payloads):
            conn.send((cmd, payload))
        return [conn.recv() for conn in self.conns]

    def reset(self, mix):
        return np.concatenate(self.call("reset", [mix] * len(self.conns)))

    def step(self, actions):
        out = [x for part in self.call("step", np.split(actions, len(self.conns))) for x in part]
        return (np.stack([o[0] for o in out]), np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out]),
                np.array([o[3] for o in out]), np.array([o[4] for o in out]))

    def evaluate(self, torch, actor, task, episodes=96, indices_all=None):
        """Mean action on held-out placements (the first ``episodes``, or ``indices_all``), rounded down to a multiple of the env count."""
        ok, unsafe, total = 0, 0, 0
        indices_all = np.arange(episodes) if indices_all is None else np.asarray(indices_all)
        for start in range(0, len(indices_all) - self.n + 1, self.n):
            indices = indices_all[start:start + self.n]
            obs = np.concatenate(self.call("eval_reset", [(task, part) for part in np.split(indices, len(self.conns))]))
            alive = np.ones(self.n, bool)
            for _ in range(HORIZON):
                with torch.no_grad():
                    action = actor(torch.from_numpy(obs)).numpy()
                out = [x for part in self.call("eval_step", np.split(action, len(self.conns))) for x in part]
                obs = np.stack([o[0] for o in out])
                for i, (_, _, done, info) in enumerate(out):
                    if alive[i] and done:
                        ok += info["success"]; unsafe += info["unsafe"]; alive[i] = False
                if not alive.any():
                    break
            total += self.n
        self.call("train_mode", [None] * len(self.conns))
        return dict(success=ok / total, unsafe=unsafe / total, episodes=total)

    def close(self):
        for conn in self.conns:
            conn.send(("close", None))


R, D, G, L = "reach", "descend", "grasp", "lift"
CONDITIONS = {      # phases: (task mix, steps)
    "staged": [([(R, 1)], 1_500_000), ([(D, 2), (R, 1)], 1_500_000), ([(G, 2), (D, 1), (R, 1)], 2_000_000), ([(L, 2), (G, 1), (D, 1), (R, 1)], 2_000_000)],
    # v3: no replay (in the toy, replaying shaped stages blocked the next stage), outputs kept unsaturated
    "staged_noreplay": [([(R, 1)], 1_500_000), ([(D, 1)], 1_500_000), ([(G, 1)], 2_000_000), ([(L, 1)], 2_000_000)],
    "flat": [([(L, 1)], 7_000_000)],
    # inheritance in the arm sim (toy result 2): reach on every placement or on the left half only, then descend on the
    # left half only (reach replayed; one potential, so the replay is consistent). Read: descend on the RIGHT half.
    "inherit_reach_all": [([(R, 1, "all")], 1_500_000), ([(D, 2, "L"), (R, 1, "all")], 1_500_000)],
    "inherit_reach_L": [([(R, 1, "L")], 1_500_000), ([(D, 2, "L"), (R, 1, "L")], 1_500_000)],
    "all_at_once": [([(L, 1), (G, 1), (D, 1), (R, 1)], 7_000_000)],
}


def train(name, phases, seed, workers, envs_per_worker, root, rollout=128, eval_every=500_000, scale=1.0):
    import torch
    from rg_toy import make_nets
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    actor, critic, log_std = make_nets(torch, obs_dim=OBS_DIM, act_dim=6, width=256, squash=True)     # bounded mean (toy lesson 3)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params + [log_std], lr=3e-4)
    gamma, lam, clip, epochs, batch, ent = 0.99, 0.95, 0.2, 4, 1024, 0.001
    vec = VecArm(workers, envs_per_worker, seed)
    out = root / "runs" / f"{name}__s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    curve, total, started = [], 0, time.time()

    def record(phase):
        row = dict(steps=total, phase=phase, seconds=round(time.time() - started), **{t: vec.evaluate(torch, actor, t) for t in tasks})
        for t in ("reach", "descend") if halves else ():
            for side, idx in halves.items():
                row[f"{t}_{side}"] = vec.evaluate(torch, actor, t, indices_all=idx)
        curve.append(row)
        (out / "curve.json").write_text(json.dumps(dict(condition=name, phases=phases, seed=seed, curve=curve)) + "\n")
        torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), log_std=log_std.detach()), out / "policy.pt")
        print(f"{name} s{seed} phase {phase} steps {total}: " + " ".join(f"{t} {v['success']:.2f}" for t, v in row.items() if isinstance(v, dict))
              + f" std {log_std.exp().mean():.2f} "
              f"{total / max(time.time() - started, 1):.0f} steps/s", flush=True)

    tasks, halves = TASKS, None
    if name.startswith("inherit"):
        held = _suite(EVAL_SUITE)
        y = np.array([s.square_center_xy[1] for s in held])
        cut = split_y(_suite(TRAIN_SUITE))
        tasks, halves = (), dict(L=np.flatnonzero(y < cut), R=np.flatnonzero(y >= cut))
    record(-1)
    for k, (mix, steps) in enumerate(phases):
        obs = vec.reset(mix)
        end, next_eval = total + int(steps * scale), total + eval_every
        while total < end:
            O, A, R, D, V, LP = [], [], [], [], [], []
            for _ in range(rollout):
                with torch.no_grad():
                    o = torch.from_numpy(obs)
                    dist = torch.distributions.Normal(actor(o), log_std.exp())
                    action = dist.sample()
                O.append(o); A.append(action); V.append(critic(o).squeeze(1).detach()); LP.append(dist.log_prob(action).sum(1))
                obs, reward, done, _, _ = vec.step(action.numpy())
                R.append(torch.from_numpy(reward)); D.append(torch.from_numpy(done.astype(np.float32)))
            with torch.no_grad():
                last = critic(torch.from_numpy(obs)).squeeze(1)
            adv, gae = [None] * rollout, torch.zeros(vec.n)
            for t in reversed(range(rollout)):
                nxt = last if t == rollout - 1 else V[t + 1]
                delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]
                gae = delta + gamma * lam * (1 - D[t]) * gae
                adv[t] = gae
            O, A, LP = torch.cat(O), torch.cat(A), torch.cat(LP)
            adv = torch.cat(adv); ret = adv + torch.cat(V)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(epochs):
                for idx in torch.randperm(O.shape[0]).split(batch):
                    pre = actor[0](O[idx])          # (mlp, tanh): the value before the tanh is kept small so no output
                    dist = torch.distributions.Normal(torch.tanh(pre), log_std.exp())     # saturates and freezes (toy lesson 4)
                    ratio = (dist.log_prob(A[idx]).sum(1) - LP[idx]).exp()
                    loss = 0.01 * pre.pow(2).mean() - torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean() \
                        + 0.5 * (critic(O[idx]).squeeze(1) - ret[idx]).pow(2).mean() - ent * dist.entropy().sum(1).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 0.5)
                    opt.step()
            total += O.shape[0]
            if total >= next_eval and total < end:
                record(k); next_eval += eval_every
        record(k)
    vec.close()
    return curve


def cmd_run(args):
    train(args.condition, CONDITIONS[args.condition], args.seed, args.workers, args.envs_per_worker, args.root, eval_every=args.eval_every, scale=args.scale)
    return 0


def cmd_report(args):
    for path in sorted((args.root / "runs").glob("*/curve.json")):
        data = json.loads(path.read_text())
        print(path.parent.name)
        for row in data["curve"]:
            print(f"  phase {row['phase']:>2} steps {row['steps']:>9} ({row['seconds']:>6} s): " + " ".join(f"{t} {v['success']:.2f}" for t, v in row.items() if isinstance(v, dict)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--envs-per-worker", type=int, default=4)
    r.add_argument("--eval-every", type=int, default=500_000)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("report")
    t.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    t.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
