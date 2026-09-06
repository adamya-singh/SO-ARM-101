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
from so_arm101_v2.contracts.bench import BenchConfig
from so_arm101_v2.contracts.physical_io import (
    assert_pinned_calibration, connect_read_only, disconnect_read_only, read_measured_act,
)

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


def gate_command(current_act: np.ndarray, target_act: np.ndarray, *,
                 max_relative_target: float = 20.0, shoulder_floor: float | None = None) -> tuple[bool, str]:
    """True/reason if the command is safe to send (no mask fires)."""
    target = np.asarray(target_act, dtype=np.float32)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        return False, "nonfinite"
    if np.asarray(current_act).shape != (6,) or not np.isfinite(current_act).all():
        return False, "nonfinite_current"
    evaluation = evaluate_physical_command(current_act, target,
        max_relative_target=max_relative_target, shoulder_floor=shoulder_floor)
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
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="None")
    parser.add_argument("--max-relative-target", type=float, default=20.0)
    parser.add_argument("--enable-motion", action="store_true",
                        help="actually move the arm (default: dry run)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate inputs, connect, read one observation, exit")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--bench-config", type=Path)
    args = parser.parse_args(argv)
    if (args.enable_motion or args.preflight_only) and args.bench_config is None:
        parser.error("physical execution requires --bench-config")
    if args.enable_motion and args.log is None:
        parser.error("motion requires --log")
    bench = BenchConfig.load(args.bench_config) if args.bench_config else None
    gate_options = dict(max_relative_target=args.max_relative_target,
                        shoulder_floor=bench.shoulder_floor if bench else None)

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
    for index in range(trajectory.shape[0]):
        ok, reason = gate_command(trajectory[max(0, index - 1)], trajectory[index], **gate_options)
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
    # The live calibration file must equal the pinned contract copy before
    # the bus is opened; the read-only connect then confirms the motors match.
    assert_pinned_calibration(robot)
    log_handle = None
    writer = None
    if args.enable_motion:
        # Claim the exclusive log before touching hardware so a log failure
        # can never interrupt a motion sequence, and so every motion attempt
        # leaves its own record.
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("x", newline="")
        writer = csv.writer(log_handle)
        writer.writerow(["step", "monotonic_time", *[f"measured_{n}" for n in JOINT_NAMES],
                         *[f"target_{n}" for n in JOINT_NAMES],
                         *[f"sent_physical_{n}" for n in JOINT_NAMES], "hold_reason"])
        log_handle.flush()
    # Avoid robot.connect(): it configures motors and toggles torque even
    # during the legacy "preflight-only" path.
    try:
        connect_read_only(robot)
    except Exception:
        if log_handle is not None:
            # Nothing was sent; do not leave a header-only log that blocks
            # reuse of the path.
            log_handle.close()
            args.log.unlink(missing_ok=True)
        raise
    try:
        current = read_measured_act(robot)
        print(f"current pose (ACT): {np.round(current, 4).tolist()}")
        if args.preflight_only:
            ok, reason = gate_command(current, trajectory[0], **gate_options)
            if not ok:
                print(f"preflight REFUSED first command: {reason}")
                return 2
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
        # Establish the present pose as the goal before enabling torque so
        # a stale goal left on the controller cannot cause a startup jump.
        current = read_measured_act(robot)
        ok, reason = gate_command(current, trajectory[0], **gate_options)
        if not ok:
            print(f"REFUSING first command: {reason}")
            return 2
        robot.bus.sync_write("Goal_Position", {
            name: float(value) for name, value in zip(JOINT_NAMES, act_to_physical_normalized(current))
        })
        robot.bus.enable_torque()
        log_rows = []
        control_dt = 1.0 / CONTROL_HZ
        for index in range(trajectory.shape[0]):
            start_time = time.monotonic()
            current = read_measured_act(robot)
            target = trajectory[index]
            ok, reason = gate_command(current, target, **gate_options)
            if not ok:
                print(f"HOLD at step {index}: {reason}")
                writer.writerow((index, start_time, *current.tolist(), *target.tolist(), *([""] * 6), reason))
                log_handle.flush()
                return 2
            physical = act_to_physical_normalized(target)
            action = {f"{name}.pos": float(physical[i]) for i, name in enumerate(JOINT_NAMES)}
            sent = robot.send_action(action)
            sent_values = [sent[f"{name}.pos"] for name in JOINT_NAMES]
            reason = "driver_modified_command" if not np.allclose(sent_values, physical, atol=1e-5, rtol=0) else ""
            writer.writerow((index, start_time, *current.tolist(), *target.tolist(), *sent_values, reason))
            log_handle.flush()
            log_rows.append(index)
            if reason:
                raise RuntimeError(reason)
            elapsed = time.monotonic() - start_time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        print(f"replay complete: {len(log_rows)} steps")
        print(f"log: {args.log}")
        return 0
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    finally:
        if log_handle is not None:
            log_handle.close()
        # Preserve torque state. Motion runs retain their last goal;
        # inspection never disables torque and lets an unsupported arm fall.
        disconnect_read_only(robot)


if __name__ == "__main__":
    raise SystemExit(main())
