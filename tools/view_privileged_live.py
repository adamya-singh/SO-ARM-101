"""Watch the privileged controller or a learned policy live in the MuJoCo viewer.

Usage (from the SO-ARM-101 repo root):
    PYTHONPATH=src python tools/view_privileged_live.py [--scenario N] [--speed 1.0]
    PYTHONPATH=src python tools/view_privileged_live.py --checkpoint <model.pt>
    PYTHONPATH=src python tools/view_privileged_live.py --bench [--hold reset|viewing] [--speed 0.5]

Without --checkpoint this drives the same privileged controller and scenario
the preflight uses. With --checkpoint it loads a chunked clone checkpoint and
plays it under the fixed_pick_place_v3 setting it was evaluated in. With
--bench it plays the certified bench teacher on the active 20 mm bench scene;
--hold reset|viewing freezes the arm at that recorded pose instead (pick the
"wrist_camera" camera in the viewer's Rendering panel to see what the policy
sees). Close the window (or Ctrl+C) to exit; episodes restart automatically.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from so_arm101_v2.contracts import (
    PickPlaceEvaluationState,
    TaskEvaluationState,
    evaluate_pick_place_step,
    evaluate_task_step,
    load_pick_place_contract,
    load_task_contract,
)
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.chunked import ChunkedClonePolicy
from so_arm101_v2.simulation.privileged import PrivilegedStagedController
from so_arm101_v2.simulation.suites import load_simulation_suite


def _view_checkpoint(args: argparse.Namespace) -> int:
    """Play a chunked clone checkpoint under the v3 pick-place contract."""
    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = suite.scenarios[args.scenario]
    contract = load_pick_place_contract(suite.task_contract)
    adapter = MujocoTaskAdapter(args.mujoco_model)
    policy = ChunkedClonePolicy(Path(args.checkpoint))

    print(f"policy: {policy.policy_id}  scenario: {scenario.scenario_id}  "
          "(close the viewer window to exit)")
    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        while viewer.is_running():
            adapter.reset(scenario)
            policy.reset(adapter)
            state = PickPlaceEvaluationState()
            evaluation = None
            for action in range(contract.max_actions):
                if not viewer.is_running():
                    break
                step_start = time.time()
                requested = policy.predict(
                    np.empty((0,), dtype=np.uint8), adapter.current_act(), adapter,
                )
                command = adapter.apply_policy_command(requested)
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(
                    command,
                    footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
                )
                if not state.completed:
                    state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                viewer.sync()
                pickup = measurement.pickup
                if action % 30 == 0 or (pickup.strict_bilateral_grasp and action % 5 == 0):
                    print(
                        f"a={action + 1:3d} gain={1000 * pickup.cube_height_gain_m:6.1f}mm "
                        f"strict={pickup.strict_bilateral_grasp} "
                        f"unsafe={pickup.unsafe_contact} clip={pickup.command_bound_violation} "
                        f"limit={pickup.delta_limiter_activated}"
                    )
                if state.completed:
                    print(f"contract outcome: success={evaluation.success}")
                    break
                budget = (1.0 / 30.0) / max(args.speed, 1e-3) - (time.time() - step_start)
                if budget > 0:
                    time.sleep(budget)
            time.sleep(1.5)
    adapter.close()
    return 0


def _view_bench(args: argparse.Namespace) -> int:
    """Play the bench teacher on the active bench scene, or hold a recorded pose."""
    from so_arm101_v2.simulation.bench import bench_suite
    model_path = Path("simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml")
    adapter = MujocoTaskAdapter(model_path)
    bench = adapter.bench
    scenario = bench_suite(bench, [(0.0, 0.0)], label="view", repeats=1).scenarios[0]
    contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=bench)
    policy = PrivilegedStagedController()
    print(f"bench scene: {model_path}  teacher: pad {bench.grasp_pad}, {bench.approach_pitch_deg} deg, "
          f"lead {1000 * bench.depth_lead_m:.0f} mm, offset {1000 * bench.grasp_offset_m:.0f} mm")
    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        if args.hold:
            adapter.reset(scenario)
            if args.hold == "viewing":
                for address, value in zip(adapter._joint_qpos, bench.viewing_qpos, strict=True):
                    adapter.data.qpos[address] = float(value)
                    adapter.data.ctrl[:] = 0.0
                mujoco.mj_forward(adapter.model, adapter.data)
            print(f"holding the recorded {args.hold} pose; qpos={np.round(adapter.mujoco_qpos(), 4).tolist()} "
                  "(close the window to exit)")
            while viewer.is_running():
                # Keep the position actuators on the held pose so gravity does not sag it.
                for actuator, value in zip(adapter._actuator_ids, adapter.mujoco_qpos(), strict=True):
                    adapter.data.ctrl[actuator] = float(value)
                mujoco.mj_step(adapter.model, adapter.data)
                viewer.sync()
                time.sleep(adapter.model.opt.timestep)
            adapter.close()
            return 0
        while viewer.is_running():
            adapter.reset(scenario)
            policy.reset(adapter)
            state = PickPlaceEvaluationState()
            evaluation = None
            for action in range(contract.max_actions):
                if not viewer.is_running():
                    break
                step_start = time.time()
                command = adapter.apply_policy_command(policy.predict(None, adapter.current_act(), adapter))
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(command)
                if not state.completed:
                    state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                viewer.sync()
                pickup = measurement.pickup
                if action % 30 == 0 or (pickup.strict_bilateral_grasp and action % 5 == 0):
                    print(f"a={action + 1:3d} gain={1000 * pickup.cube_height_gain_m:6.1f}mm "
                          f"strict={pickup.strict_bilateral_grasp} unsafe={pickup.unsafe_contact} "
                          f"clip={pickup.command_bound_violation} limit={pickup.delta_limiter_activated}")
                if state.completed:
                    print(f"contract outcome: success={evaluation.success}")
                    break
                budget = (1.0 / 30.0) / max(args.speed, 1e-3) - (time.time() - step_start)
                if budget > 0:
                    time.sleep(budget)
            time.sleep(1.5)
    adapter.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="active 20 mm bench scene and certified teacher")
    parser.add_argument("--hold", choices=["reset", "viewing"], default=None,
                        help="with --bench: freeze the arm at the recorded pose instead of playing")
    parser.add_argument("--scenario", type=int, default=0, help="suite scenario index (0-4)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument(
        "--mujoco-model", default="simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="chunked clone model.pt to play instead of the privileged controller",
    )
    parser.add_argument(
        "--stop-on-success", action="store_true",
        help="end the episode at pickup success like the task contract does; "
        "by default the full sequence (including the napkin place) plays out",
    )
    args = parser.parse_args()
    if args.bench:
        return _view_bench(args)
    if args.checkpoint is not None:
        return _view_checkpoint(args)

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
            contract_done = False
            for action in range(contract.episode.max_actions):
                if not viewer.is_running():
                    break
                step_start = time.time()
                requested = policy.predict(None, adapter.current_act(), adapter)
                command = adapter.apply_policy_command(requested)
                adapter.advance_control_period()
                measurement, _ = adapter.measurement(command)
                # the contract refuses evaluation after a terminal outcome, so
                # once it ends we keep stepping physics for the place phase
                # without consulting it further
                if not contract_done:
                    state, evaluation = evaluate_task_step(contract, measurement, state)
                strict_frames += int(measurement.strict_bilateral_grasp)
                viewer.sync()
                if action % 30 == 0 or (measurement.strict_bilateral_grasp and not contract_done):
                    print(
                        f"a={action + 1:3d} gain={1000 * measurement.cube_height_gain_m:6.1f}mm "
                        f"strict={measurement.strict_bilateral_grasp} "
                        f"unsafe={measurement.unsafe_contact} clip={measurement.command_bound_violation}"
                    )
                if not contract_done and (evaluation.terminated or evaluation.truncated):
                    contract_done = True
                    print(
                        f"contract outcome: success={evaluation.success} "
                        f"invalidated={evaluation.invalidated} strict_frames={strict_frames}"
                    )
                    if args.stop_on_success or not evaluation.success:
                        break
                    print("...continuing past contract success to play the napkin place phase")
                budget = (1.0 / 30.0) / max(args.speed, 1e-3) - (time.time() - step_start)
                if budget > 0:
                    time.sleep(budget)
            time.sleep(1.0)
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
