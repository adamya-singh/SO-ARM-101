#!/usr/bin/env python3
"""Bounded physical pad calibration using the authoritative paired proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from generate_act_grasp_reset_bank import (
    DEFAULT_DATASET,
    load_demo_states,
    make_candidate_pair,
    solve_grasp_control,
)
from so101_gym_env import SO101PickPlaceEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260713)
    args = parser.parse_args()
    results = []
    for thickness in (0.002, 0.00225, 0.0025):
        for friction in (1.0, 1.5, 2.0):
            env = SO101PickPlaceEnv(
                image_size=32,
                randomize_block=False,
                randomize_appearance=False,
                reward_kwargs={"strict_transition": True},
            )
            for name in ("fixed_jaw_contact_pad", "moving_jaw_contact_pad"):
                geom_id = env.model.geom(name).id
                env.model.geom_size[geom_id, 0] = thickness
                env.model.geom_friction[geom_id, 0] = friction
            ik_data = mujoco.MjData(env.model)
            base = solve_grasp_control(
                env.model, ik_data, load_demo_states(DEFAULT_DATASET)
            )
            pair = make_candidate_pair(
                env, ik_data, base, np.random.default_rng(args.seed), 0
            )
            result = {
                "pad_half_thickness_m": thickness,
                "sliding_friction": friction,
                "passed": pair is not None,
            }
            if pair is not None:
                result["grasped_height_gain"] = pair[0]["proof"]["height_gain"]
                result["pregrasp_height_gain"] = pair[1]["proof"]["height_gain"]
            results.append(result)
            env.close()
    passing = [result for result in results if result["passed"]]
    if not passing:
        raise RuntimeError("No physical pad calibration passed the dynamic proof")
    selected = min(
        passing, key=lambda result: (result["pad_half_thickness_m"], result["sliding_friction"])
    )
    report = {"selection_rule": "smallest pad then lowest friction", "selected": selected,
              "results": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
