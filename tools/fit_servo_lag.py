"""Fit the simulator's servo-lag stress response to a recorded physical episode. Simulation + logs only.

Replays the episode's sent commands open-loop on the bench scene (same gate, same control period) with a
per-joint first-order response in front of the actuators, for a grid of response values, and reports per
joint the value whose simulated joint trajectory is closest (rms, physical units) to the measured one.
1.0 = the certified scene. The result goes into ``simulation/lag_stress.py``.
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

DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


def load_episode(directory: Path, steps: int) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(directory / "steps.csv")))[:steps]
    sent = np.array([[float(r[f"sent_physical_{j}"]) for j in JOINT_NAMES] for r in rows], dtype=np.float32)
    measured = np.array([act_to_physical_normalized(np.array([float(r[f"measured_act_{j}"]) for j in JOINT_NAMES], np.float32)) for r in rows])
    return sent, measured


def replay(model: Path, sent: np.ndarray, response: float | np.ndarray) -> np.ndarray:
    """Measured joints (physical units) before each command, as the runner logs them."""
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    adapter = MujocoTaskAdapter(model)
    try:
        adapter.reset(bench_suite(adapter.bench, [(0, 0)], label="lagfit", repeats=1).scenarios[0])
        adapter.command_response = np.broadcast_to(np.asarray(response, dtype=np.float64), (6,)).copy()
        out = []
        for command in sent:
            out.append(act_to_physical_normalized(adapter.current_act()))
            adapter.apply_policy_command(physical_normalized_to_act(command))
            adapter.advance_control_period()
        return np.array(out)
    finally:
        adapter.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-dir", type=Path, action="append", required=True)
    p.add_argument("--steps", type=int, default=240, help="fit window from the episode start")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    grid = [round(float(v), 3) for v in np.concatenate([np.arange(0.06, 0.30, 0.02), np.arange(0.30, 1.0001, 0.05)])]
    record = dict(tool="fit_servo_lag", grid=grid, steps=args.steps, episodes={})
    total = np.zeros((len(grid), 6))
    for directory in args.episode_dir:
        sent, measured = load_episode(directory, args.steps)
        rms = np.array([np.sqrt(np.mean((replay(args.model, sent, value) - measured) ** 2, axis=0)) for value in grid])
        total += rms ** 2
        best = rms.argmin(axis=0)
        record["episodes"][directory.name] = {
            name: dict(response=grid[best[i]], rms_units=round(float(rms[best[i], i]), 2), rms_units_certified_scene=round(float(rms[-1, i]), 2))
            for i, name in enumerate(JOINT_NAMES)}
        print(directory.name)
        for i, name in enumerate(JOINT_NAMES):
            print(f"  {name:14s} response {grid[best[i]]:.2f}  rms {rms[best[i], i]:.2f} units (certified scene {rms[-1, i]:.2f})")
    pooled = total.argmin(axis=0)
    record["pooled_response"] = {name: grid[pooled[i]] for i, name in enumerate(JOINT_NAMES)}
    print("pooled:", record["pooled_response"])
    if args.output is not None:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
