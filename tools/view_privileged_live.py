"""Watch the privileged preflight controller live in the MuJoCo viewer.

Usage (from the SO-ARM-101 repo root):
    PYTHONPATH=src python tools/view_privileged_live.py [--scenario N] [--speed 1.0]

Drives the same controller and scenario the preflight uses, in real time,
inside mujoco.viewer. Close the window (or Ctrl+C) to exit; the episode
restarts automatically when it ends.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from so_arm101_v2.contracts import TaskEvaluationState, evaluate_task_step, load_task_contract
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.privileged import PrivilegedStagedController
from so_arm101_v2.simulation.suites import load_simulation_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0, help="suite scenario index (0-4)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument(
        "--mujoco-model", default="simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    )
    args = parser.parse_args()

    suite = load_simulation_suite("fixed_pickup_contract_v1")
    scenario = suite.scenarios[args.scenario]
    contract = load_task_contract("fixed_cube_pickup_v1")
    adapter = MujocoTaskAdapter(args.mujoco_model)
    policy = PrivilegedStagedController()

    print(f"scenario: {scenario.scenario_id}  (close the viewer window to exit)")
    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        while viewer.is_running():
            adapter.reset(scenario)
            policy.reset(adapter)
            state = TaskEvaluationState()
            strict_frames = 0
            for action in range(contract.episode.max_actions):
                if not viewer.is_running():
                    break
                step_start = time.time()
                requested = policy.predict(None, adapter.current_act(), adapter)
                command = adapter.apply_policy_command(requested)
                adapter.advance_control_period()
                measurement, _ = adapter.measurement(command)
                state, evaluation = evaluate_task_step(contract, measurement, state)
                strict_frames += int(measurement.strict_bilateral_grasp)
                viewer.sync()
                if action % 30 == 0 or measurement.strict_bilateral_grasp:
                    print(
                        f"a={action + 1:3d} gain={1000 * measurement.cube_height_gain_m:6.1f}mm "
                        f"strict={measurement.strict_bilateral_grasp} "
                        f"unsafe={measurement.unsafe_contact} clip={measurement.command_bound_violation}"
                    )
                if evaluation.terminated or evaluation.truncated:
                    print(
                        f"episode over: success={evaluation.success} "
                        f"invalidated={evaluation.invalidated} strict_frames={strict_frames}"
                    )
                    break
                budget = (1.0 / 30.0) / max(args.speed, 1e-3) - (time.time() - step_start)
                if budget > 0:
                    time.sleep(budget)
            time.sleep(1.0)
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
