#!/usr/bin/env python3
"""Validate every curriculum state through sequential and spawned environments."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--subprocess-workers", type=int, default=12)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def validate_state(state: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from so101_gym_env import SO101PickPlaceEnv

    env = SO101PickPlaceEnv(
        image_size=32,
        randomize_block=False,
        randomize_appearance=False,
        reward_kwargs={"strict_transition": True},
    )
    proof = env.prove_curriculum_reset(state)
    env.close()
    return proof


def main() -> int:
    args = parse_args()
    payload = json.loads(args.reset_bank.read_text())
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("Reset bank must use preload-aware schema_version 2")
    states = []
    for stage in ("grasped", "pregrasp"):
        stage_states = list(payload.get(stage, []))
        if len(stage_states) < 32:
            raise ValueError(f"Reset bank requires 32 {stage} states; found {len(stage_states)}")
        states.extend(stage_states)

    sequential = [validate_state(state) for state in states]
    failures = [index for index, proof in enumerate(sequential) if not proof["valid"]]
    if failures:
        raise RuntimeError(f"Sequential production reset proof failed for indices {failures}")

    if args.subprocess_workers > 0:
        context = mp.get_context("spawn")
        with context.Pool(processes=args.subprocess_workers) as pool:
            subprocess_proofs = pool.map(validate_state, states)
        failures = [
            index for index, proof in enumerate(subprocess_proofs) if not proof["valid"]
        ]
        if failures:
            raise RuntimeError(f"Subprocess production reset proof failed for indices {failures}")
        for index, (expected, actual) in enumerate(zip(sequential, subprocess_proofs)):
            for key in ("height_gain", "lateral_displacement", "grip_force"):
                if abs(float(expected[key]) - float(actual[key])) > 1e-9:
                    raise RuntimeError(f"Sequential/subprocess mismatch at state {index}, metric {key}")

    report = {
        "valid": True,
        "schema_version": 2,
        "states": len(states),
        "grasped": len(payload["grasped"]),
        "pregrasp": len(payload["pregrasp"]),
        "subprocess_workers": args.subprocess_workers,
        "minimum_height_gain": min(float(proof["height_gain"]) for proof in sequential),
        "maximum_lateral_displacement": max(
            float(proof["lateral_displacement"]) for proof in sequential
        ),
        "minimum_grip_force": min(float(proof["grip_force"]) for proof in sequential),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
