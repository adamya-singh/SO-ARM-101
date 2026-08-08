"""Replay a recorded ACT trajectory on the physical SO-101 arm, safely.

The earliest end-to-end physical smoke test: every command passes through
the v2 safety contract (`evaluate_physical_command`) BEFORE it is sent; any
mask refusal holds the arm instead of sending. Dry-run is the default —
motion requires --enable-motion AND an interactive confirmation.

Inputs: a v2 oracle capture manifest + scenario id (replays its
`executed_act` rows), or a CSV with act_0..act_5 columns (e.g. the legacy
`imitation-learning/outputs/*_actions*.csv` logs).

See notes/physical-smoke-runbook.md for the bench procedure.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from so_arm101_v2.contracts import evaluate_physical_command
from so_arm101_v2.contracts.physical import act_to_physical_normalized

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
CONTROL_HZ = 30.0
START_POSE_TOLERANCE_ACT = 0.35


def load_manifest_trajectory(manifest_path: Path, scenario_id: str) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    offset = 0
    for episode in episodes:
        if episode["scenario_id"] == scenario_id:
            rows = int(episode["rows"])
            break
        offset += int(episode["rows"])
    else:
        raise ValueError(f"scenario {scenario_id!r} not in manifest")
    with np.load(manifest_path.parent / manifest["arrays"]["path"]) as handle:
        executed = np.asarray(handle["executed_act"], dtype=np.float32)
    return executed[offset:offset + rows]


def load_csv_trajectory(csv_path: Path, prefix: str = "action_rad_") -> np.ndarray:
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [f"{prefix}{i}" for i in range(6)]
        if not all(name in (reader.fieldnames or []) for name in columns):
            raise ValueError(f"CSV lacks {prefix}0..5 columns: {reader.fieldnames}")
        for row in reader:
            rows.append([float(row[name]) for name in columns])
    if not rows:
        raise ValueError("CSV contains no rows")
    return np.asarray(rows, dtype=np.float32)


def gate_command(current_act: np.ndarray, target_act: np.ndarray) -> tuple[bool, str]:
    """True/reason if the command is safe to send (no mask fires)."""
    target = np.asarray(target_act, dtype=np.float32)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        return False, "nonfinite"
    evaluation = evaluate_physical_command(current_act, target)
    for name, mask in (
        ("act_clip", evaluation.act_clip_mask),
        ("mujoco_clip", evaluation.mujoco_clip_mask),
        ("physical_clip", evaluation.physical_clip_mask),
        ("relative_limit", evaluation.relative_limit_mask),
    ):
        if np.asarray(mask).any():
            joints = [JOINT_NAMES[i] for i in np.flatnonzero(mask)]
            return False, f"{name}:{','.join(joints)}"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-manifest", type=Path)
    source.add_argument("--csv", type=Path)
    parser.add_argument("--scenario", default="nominal")
    parser.add_argument("--csv-prefix", default="action_rad_")
    parser.add_argument("--robot-port", default="/dev/ttyUSB0")
    parser.add_argument("--robot-id", default="None")
    parser.add_argument("--max-relative-target", type=float, default=20.0)
    parser.add_argument("--enable-motion", action="store_true",
                        help="actually move the arm (default: dry run)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate inputs, connect, read one observation, exit")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.capture_manifest is not None:
        trajectory = load_manifest_trajectory(args.capture_manifest, args.scenario)
        source_name = f"{args.capture_manifest}:{args.scenario}"
    else:
        trajectory = load_csv_trajectory(args.csv, args.csv_prefix)
        source_name = str(args.csv)
    print(f"trajectory: {source_name} ({trajectory.shape[0]} steps)")

    # Offline validation first: the whole trajectory must gate cleanly
    # against itself (step-to-step), before any hardware is touched.
    refused = 0
    for index in range(1, trajectory.shape[0]):
        ok, reason = gate_command(trajectory[index - 1], trajectory[index])
        if not ok:
            refused += 1
            print(f"OFFLINE-GATE step {index}: {reason}")
    if refused:
        print(f"REFUSING: {refused} steps fail the safety contract offline")
        return 2
    print("offline gating: all steps clean")

    if not args.enable_motion and not args.preflight_only:
        print("DRY RUN complete (pass --enable-motion to move the arm)")
        return 0

    # Hardware path.
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        print(f"lerobot robot classes unavailable: {exc}")
        return 1
    config = SO101FollowerConfig(
        id=args.robot_id, port=args.robot_port,
        cameras={}, max_relative_target=args.max_relative_target,
        use_degrees=False,
    )
    robot = SO101Follower(config)
    robot.connect()
    try:
        observation = robot.get_observation()
        state = np.asarray(
            [observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32
        )
        current = np.zeros(6, dtype=np.float32)
        current[:5] = state[:5] / 100.0 * np.pi
        current[5] = state[5] / 100.0 * 1.7
        print(f"current pose (ACT): {np.round(current, 4).tolist()}")
        if args.preflight_only:
            print("preflight OK")
            return 0
        start_error = float(np.max(np.abs(current - trajectory[0])))
        if start_error > START_POSE_TOLERANCE_ACT:
            print(f"REFUSING: start pose differs from trajectory start by {start_error:.3f} ACT "
                  f"(> {START_POSE_TOLERANCE_ACT}); move the arm near the start pose first")
            return 2
        answer = input("Send motion to the physical arm? Press Enter to confirm, anything else aborts: ")
        if answer.strip():
            print("aborted")
            return 0
        log_rows = []
        control_dt = 1.0 / CONTROL_HZ
        for index in range(trajectory.shape[0]):
            start_time = time.time()
            target = trajectory[index]
            ok, reason = gate_command(current, target)
            if not ok:
                print(f"HOLD at step {index}: {reason}")
                log_rows.append((index, *current.tolist(), *target.tolist(), reason))
                continue
            physical = act_to_physical_normalized(target)
            action = {f"{name}.pos": float(physical[i]) for i, name in enumerate(JOINT_NAMES)}
            robot.send_action(action)
            current = target.copy()
            log_rows.append((index, *current.tolist(), *target.tolist(), ""))
            elapsed = time.time() - start_time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        print(f"replay complete: {len(log_rows)} steps")
        if args.log:
            with open(args.log, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", *[f"state_{n}" for n in JOINT_NAMES],
                                 *[f"target_{n}" for n in JOINT_NAMES], "hold_reason"])
                writer.writerows(log_rows)
            print(f"log: {args.log}")
        return 0
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
