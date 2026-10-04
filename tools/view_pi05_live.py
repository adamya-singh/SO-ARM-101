"""Watch the pi0.5 fine-tune drive the bench arm live in the MuJoCo viewer, on the held-out placements.

Each episode is set up the way ``evaluate_closed_loop`` sets it up (held-out suite scenario: placement,
appearance, the contract's stopping rule), so what you see is what the 30-rollout evaluation scored. The
bottom-left corner of the window shows the wrist frame the policy read at its latest replan. The evaluation's recorded outcome for
the pose is printed next to the live one.

Usage (lerobot-pi05 env, from the SO-ARM-101 repo root, in a terminal with a display):
  PYTHONNOUSERSITE=1 ~/miniforge3/envs/lerobot-pi05/bin/python tools/view_pi05_live.py [--execute 30] [--poses 2 3 6] [--speed 1.0]
Close the MuJoCo window (or Ctrl+C) to exit; it cycles through the poses until then.
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
sys.path.insert(0, str(ROOT / "src"))

RUN = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/pi05_lora16_10k_20261002"
MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
DATA = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/placements4800_20260912"


def recorded_outcomes(execute: int) -> dict[str, str]:
    result = RUN / "pi05_result.json"
    if not result.exists():
        return {}
    rows = json.loads(result.read_text())["rollouts"].get(str(execute), {}).get("rollout_rows", [])
    return {r["scenario_id"]: r["failure_category"] for r in rows if int(r.get("repeat", 0)) == 0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=RUN / "model/pi05.pt")
    p.add_argument("--execute", type=int, default=30, help="commands run per camera read (30 = the better schedule, 90 = CNN schedule)")
    p.add_argument("--poses", type=int, nargs="*", default=None, help="held-out pose indices to cycle (default all 10)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (1.0 = real time, 30 Hz)")
    p.add_argument("--no-wrist", action="store_true", help="do not overlay the wrist-camera frame")
    args = p.parse_args(argv)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("no display: run this from a terminal with DISPLAY set (WSLg: echo $DISPLAY prints :0)")
        return 1

    import mujoco
    import mujoco.viewer
    from so_arm101_v2.contracts import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
    from so_arm101_v2.learning.pi05 import PromptTokenizer, load_checkpoint
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.pi05_policy import Pi05ChunkPolicy
    from so_arm101_v2.simulation.suites import load_suite_from_path

    suite = load_suite_from_path(Path((DATA / "heldout_suite.path").read_text().strip()))
    scenarios = [suite.scenarios[i] for i in (args.poses if args.poses else range(len(suite.scenarios)))]
    recorded = recorded_outcomes(args.execute)
    print(f"loading {args.checkpoint} ...", flush=True)
    policy_model, config, stats, _ = load_checkpoint(args.checkpoint)
    prompt = PromptTokenizer(config.tokenizer, config.tokenizer_max_length)
    policy = Pi05ChunkPolicy(policy_model, config, stats, prompt, execute=args.execute, clamp_channels=(5,))
    adapter = MujocoTaskAdapter(MODEL)
    contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
    print(f"policy {policy.policy_id}: {len(scenarios)} held-out placements; close the MuJoCo window to exit", flush=True)

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        k = 0
        while viewer.is_running():
            scenario = scenarios[k % len(scenarios)]
            k += 1
            adapter.reset(scenario)
            policy.reset(adapter)
            viewer.sync()
            state = PickPlaceEvaluationState()
            evaluation = None
            max_lift = 0.0
            last_read = 0
            print(f"\n=== {scenario.scenario_id} (evaluation recorded: {recorded.get(scenario.scenario_id, 'n/a')}) ===", flush=True)
            for action in range(contract.max_actions):
                if not viewer.is_running():
                    break
                started = time.time()
                looking = policy.needs_frame
                raw, _, current = adapter.observation(render_pixels=True)
                if looking and not args.no_wrist:
                    # The viewer draws overlays itself (the bundled OpenCV is headless): the frame just read, 2x, bottom left.
                    frame = np.repeat(np.repeat(np.ascontiguousarray(raw, dtype=np.uint8), 2, axis=0), 2, axis=1)
                    viewer.set_images((mujoco.MjrRect(8, 8, frame.shape[1], frame.shape[0]), frame))
                if looking:
                    last_read = action
                status = (f"{scenario.scenario_id}   step {action + 1}/{contract.max_actions}   lift {1000 * max_lift:4.1f} mm",
                          f"pi0.5, replans every {args.execute} steps (last read: step {last_read})   evaluation recorded: {recorded.get(scenario.scenario_id, 'n/a')}")
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, status[0], status[1]))
                command = adapter.apply_policy_command(policy.predict(raw, current, adapter))
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(command, footprint_edge_margin_m=contract.placement.footprint_edge_margin_m)
                state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                viewer.sync()
                pickup = measurement.pickup
                max_lift = max(max_lift, pickup.cube_height_gain_m)
                for event in list(evaluation.pickup_events) + list(evaluation.events):
                    print(f"  step {action + 1:3d}: {event.value}", flush=True)
                if looking and action % 90 == 0:
                    print(f"  step {action:3d}: replanning from the wrist frame", flush=True)
                if evaluation.terminated or evaluation.truncated:
                    break
                budget = (1.0 / 30.0) / max(args.speed, 1e-3) - (time.time() - started)
                if budget > 0:
                    time.sleep(budget)
            if evaluation is not None:
                outcome = "SUCCESS" if evaluation.success else ("safety invalidation" if evaluation.invalidated else "failed")
                print(f"  -> {outcome}; max lift {1000 * max_lift:.1f} mm", flush=True)
            end = time.time()
            while viewer.is_running() and time.time() - end < 2.0:
                viewer.sync(); time.sleep(0.05)
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
