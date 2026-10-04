"""Fit the simulated servos (stiffness, damping, dry friction per joint) to recorded physical runs. Simulation + logs only.

Each run's sent commands are replayed open-loop on the bench scene at the same control period, with
candidate servo parameters written onto the loaded model (``simulation/servo_model.py``). The score per joint is the
rms difference, in physical units, between the simulated and the measured joint over the whole run, with the
samples where that joint's command is at rest counted in full and the moving ones at a quarter (the resting error
at the pre-grasp pose is what decides a grasp; a first fit on plain rms left the wrist 3 units off there). One joint is
tuned at a time with the others held (a wrist's tracking depends on how far the elbow under it sags), first over a
wide grid, then over finer grids around the current value; candidates are scored in parallel.

Use the servo probe runs (``tools/run_physical_episode.py --servo-probe``) to fit and other episodes to check.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so_arm101_v2.contracts.coordinates import JOINT_NAMES  # noqa: E402
from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act  # noqa: E402
from so_arm101_v2.simulation.servo_model import ServoModel, apply_servo_model, model_servo  # noqa: E402

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def load_run(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(directory / "steps.csv")))
    sent = np.array([[float(r[f"sent_physical_{j}"]) for j in JOINT_NAMES] for r in rows], dtype=np.float32)
    measured = np.array([act_to_physical_normalized(np.array([float(r[f"measured_act_{j}"]) for j in JOINT_NAMES], np.float32)) for r in rows])
    return sent, measured


class Replayer:
    """One adapter, many replays: the joints the runner would have logged before each command, in physical units."""

    def __init__(self, model: Path) -> None:
        from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
        from so_arm101_v2.simulation.bench import bench_suite
        self.adapter = MujocoTaskAdapter(model)
        self.scenario = bench_suite(self.adapter.bench, [(0, 0)], label="servofit", repeats=1).scenarios[0]
        self.certified = model_servo(self.adapter.model)

    def replay(self, sent: np.ndarray, measured_start: np.ndarray, servo: ServoModel) -> np.ndarray:
        import mujoco
        adapter = self.adapter
        apply_servo_model(adapter.model, self.certified)
        adapter.reset(self.scenario)
        apply_servo_model(adapter.model, servo)
        # Start from the pose the run started at (the reset as the real arm held it), settled under the candidate servo.
        qpos = adapter.joint_map.act_to_mujoco(physical_normalized_to_act(measured_start.astype(np.float32)))
        for address, value, actuator in zip(adapter._joint_qpos, qpos, adapter._actuator_ids):
            adapter.data.qpos[address] = float(value)
            adapter.data.ctrl[actuator] = float(value)
        adapter.data.qvel[:] = 0.0
        mujoco.mj_forward(adapter.model, adapter.data)
        low, high = adapter.model.actuator_ctrlrange[adapter._actuator_ids].T
        out = []
        for command in sent:
            out.append(act_to_physical_normalized(adapter.current_act()))
            # The commands were already gated on the arm (clip-and-continue). Written straight to the actuators: the
            # simulator's own gate holds outright when a soft joint falls behind, which would freeze the replay.
            target = adapter.joint_map.act_to_mujoco(physical_normalized_to_act(command))
            adapter.data.ctrl[adapter._actuator_ids] = np.clip(target, low, high)
            adapter.advance_control_period()
        return np.array(out)

    def close(self) -> None:
        apply_servo_model(self.adapter.model, self.certified)
        self.adapter.close()


MOVING_WEIGHT = 0.25        # a sample taken while that joint's command is moving counts this much; at rest, 1
REST_SPEED_UNITS = 0.2      # "at rest": the joint's command moved less than this per step over the last five steps


def rest_weights(sent: np.ndarray) -> np.ndarray:
    """Per-sample, per-joint weights: where the arm rests is where a grasp is decided, so those samples count in full."""
    speed = np.abs(np.diff(sent, axis=0, prepend=sent[:1]))
    recent = np.stack([np.roll(speed, k, axis=0) for k in range(5)]).max(axis=0)
    recent[:5] = np.maximum(recent[:5], speed[:5].max(axis=0))
    return np.where(recent < REST_SPEED_UNITS, 1.0, MOVING_WEIGHT)


def score(replayer: Replayer, runs, servo: ServoModel, *, weighted: bool = True) -> np.ndarray:
    """Per-joint rms (physical units) over all runs; by default weighted toward the samples where the joint rests."""
    total, count = np.zeros(6), np.zeros(6)
    for sent, measured in runs:
        simulated = replayer.replay(sent, measured[0], servo)
        weights = rest_weights(sent) if weighted else np.ones_like(sent)
        total += (weights * (simulated - measured) ** 2).sum(axis=0)
        count += weights.sum(axis=0)
    return np.sqrt(total / count)


_WORKER: dict = {}


def _worker_init(model: str, run_dirs: list[str]) -> None:
    _WORKER["replayer"] = Replayer(Path(model))
    _WORKER["runs"] = [load_run(Path(d)) for d in run_dirs]


def _worker_score(servo: ServoModel) -> np.ndarray:
    return score(_WORKER["replayer"], _WORKER["runs"], servo)


def with_joint(servo: ServoModel, joint: int, kp: float, kv: float, friction: float) -> ServoModel:
    return ServoModel(tuple(kp if j == joint else servo.kp[j] for j in range(6)),
                      tuple(kv if j == joint else servo.kv[j] for j in range(6)),
                      tuple(friction if j == joint else servo.frictionloss[j] for j in range(6)))


def joint_grid(servo: ServoModel, joint: int, sweep: int) -> list[tuple[float, float, float]]:
    """Candidates for one joint: a wide grid on the first sweep, then grids around the current value.

    Damping is searched as a time constant kv / kp, which is what sets how far a joint trails a ramp.
    """
    if sweep == 0:
        stiffness = [4.0, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0]
        constants = [0.07, 0.09, 0.12]
        frictions = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6]
    else:
        factor = (1.2, 1.08)[min(sweep - 1, 1)]
        stiffness = [servo.kp[joint] / factor, servo.kp[joint], servo.kp[joint] * factor]
        constant = servo.kv[joint] / servo.kp[joint]
        constants = [constant / factor, constant, constant * factor]
        frictions = [servo.frictionloss[joint] / factor, servo.frictionloss[joint], servo.frictionloss[joint] * factor]
    grid = [(kp, kp * c, f) for kp in stiffness for c in constants for f in frictions]
    return grid + [(servo.kp[joint], servo.kv[joint], servo.frictionloss[joint])]


def main(argv=None) -> int:
    import multiprocessing

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-dir", type=Path, action="append", required=True, help="run directory to fit (repeatable)")
    p.add_argument("--check-dir", type=Path, action="append", default=[], help="run directory to score only (repeatable)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--joints", nargs="+", default=["elbow_flex", "shoulder_lift", "wrist_flex", "shoulder_pan", "wrist_roll"], choices=list(JOINT_NAMES),
                   help="joints to fit, in the order they are tuned (the most loaded first)")
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    fit_joints = [JOINT_NAMES.index(n) for n in args.joints]
    replayer = Replayer(args.model)
    certified = replayer.certified
    fit_runs = [load_run(d) for d in args.fit_dir]
    base = score(replayer, fit_runs, certified)
    print("certified scene rms:", dict(zip(JOINT_NAMES, base.round(2).tolist())), flush=True)
    fitted, rms = certified, base
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, initializer=_worker_init, initargs=(str(args.model), [str(d) for d in args.fit_dir])) as pool:
        for sweep in range(args.sweeps):
            for joint in fit_joints:
                candidates = [with_joint(fitted, joint, *values) for values in joint_grid(fitted, joint, sweep)]
                scores = pool.map(_worker_score, candidates)
                # One joint moves at a time; it is judged on its own tracking, and the others must not get worse overall.
                best = min(range(len(candidates)), key=lambda k: scores[k][joint])
                if scores[best][joint] < rms[joint] - 1e-9:
                    fitted, rms = candidates[best], scores[best]
            print(f"sweep {sweep}:", dict(kp=[round(v, 2) for v in fitted.kp], kv=[round(v, 3) for v in fitted.kv], frictionloss=[round(v, 3) for v in fitted.frictionloss]),
                  dict(zip(JOINT_NAMES, rms.round(2).tolist())), flush=True)
    rms = score(replayer, fit_runs, fitted)
    try:
        record = dict(tool="fit_servo_model", fit_runs=[str(d) for d in args.fit_dir], joints=args.joints, sweeps=args.sweeps,
                      objective=dict(moving_weight=MOVING_WEIGHT, rest_speed_units=REST_SPEED_UNITS),
                      fitted_rms_units_unweighted=dict(zip(JOINT_NAMES, score(replayer, fit_runs, fitted, weighted=False).round(3).tolist())),
                      certified=dict(kp=certified.kp, kv=certified.kv, frictionloss=certified.frictionloss, rms_units=dict(zip(JOINT_NAMES, base.round(3).tolist()))),
                      fitted=dict(kp=[round(v, 3) for v in fitted.kp], kv=[round(v, 4) for v in fitted.kv],
                                  frictionloss=[round(v, 4) for v in fitted.frictionloss], rms_units=dict(zip(JOINT_NAMES, rms.round(3).tolist()))),
                      checks={})
        for directory in args.check_dir:
            run = [load_run(directory)]
            before, after = score(replayer, run, certified, weighted=False), score(replayer, run, fitted, weighted=False)
            record["checks"][directory.name] = dict(certified=dict(zip(JOINT_NAMES, before.round(3).tolist())), fitted=dict(zip(JOINT_NAMES, after.round(3).tolist())))
            print(f"check {directory.name}: certified {before.round(2)} -> fitted {after.round(2)}", flush=True)
        if args.output is not None:
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        replayer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
