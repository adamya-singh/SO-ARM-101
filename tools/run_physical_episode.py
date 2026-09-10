"""Run the trained bench vision policy on the physical SO-ARM-101 (or rehearse it in simulation).

Three modes, in the order they should be used:

  --sim-rehearsal   drive the same tool path on the MuJoCo scene (no hardware); writes the same
                    evidence layout and must reproduce the nominal simulation success.
  --preflight-only  hardware, read-only: pinned calibration, fresh pose, camera rate, the fast
                    resampler proven bit-identical to the lens contract on live frames, the
                    pan-sign check (you rotate the base by hand), and a policy dry pass that prints
                    the first chunk's joint deltas without sending anything. No torque.
  --enable-motion   everything above, then: your confirmation -> torque on at the present pose ->
                    the gated approach to the recorded reset pose (shoulder lift included) ->
                    your second confirmation -> the 16 s episode at 30 Hz, camera recorded.

Safety contract (same as tools/replay_physical_trajectory.py): the live calibration must match the
pinned resource before the bus opens; the run directory is claimed before torque; the bus is opened
read-only and robot.connect()/disconnect() are never called; every command passes the shared bench
gate (bench_hold_decision: any clip/limiter mask holds the current pose); the driver may not modify a
command; torque is never disabled by this tool. Ctrl-C stops issuing commands and leaves torque on;
it is not a mechanical stop. Support the arm before removing power.

Observation contract: raw 1920x1080 MJPEG frame -> BGR->RGB -> exact area filter -> 256x256. No
undistortion, no crop (policies from the lens-matched scene 7c765d4b and later).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from so_arm101_v2.contracts import JOINT_NAMES, load_pick_place_contract  # noqa: E402
from so_arm101_v2.contracts.bench import BenchConfig, scene_bench_config, scene_dependency_hash  # noqa: E402
from so_arm101_v2.contracts.physical import (  # noqa: E402
    act_to_physical_normalized,
    load_physical_calibration,
    physical_normalized_to_act,
)
from so_arm101_v2.contracts.physical_io import (  # noqa: E402
    assert_pinned_calibration,
    connect_read_only,
    disconnect_read_only,
    read_measured_act,
)
from so_arm101_v2.data.resources import read_resource_bytes  # noqa: E402
from so_arm101_v2.physical.camera import FrameGrabber  # noqa: E402
from so_arm101_v2.physical.dry_pass import check_reset_frame, policy_dry_pass  # noqa: E402
from so_arm101_v2.physical.evidence import BoundaryStore, StepLog, VideoRecorder, sha256_file, write_run_record  # noqa: E402
from so_arm101_v2.physical.lerobot_backend import MIN_SERVO_VOLTAGE_V, LeRobotBackend, check_servo_voltage, fast_area_resampler, verify_fast_resampler  # noqa: E402
from so_arm101_v2.physical.runner import (  # noqa: E402
    CONTROL_HZ,
    approach_plan,
    approach_step,
    run_episode,
)
from so_arm101_v2.physical.sim_backend import SimBackend  # noqa: E402
from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909/models/vision_h90/5e96018881d4140f/model.pt"
DEFAULT_MODEL = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
LIVE_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/None.json"
START_POSE_TOLERANCE_ACT = 0.35
PREFIX_TOLERANCE_RAD = 0.05
APPROACH_TIMEOUT_S = 45.0
PAN_CHECK_UNITS = 2.0


class Refused(RuntimeError):
    """A preflight or gate refusal: recorded in run.json, nothing was sent."""


# ----------------------------------------------------------------------------- helpers

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state() -> dict[str, Any]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
        return dict(commit=rev, dirty=dirty)
    except Exception as exc:  # pragma: no cover
        return dict(commit=None, dirty=None, error=str(exc)[:120])


def pan_sign_expectation(model_path: Path, bench: BenchConfig) -> dict[str, Any]:
    """Which way the square moves in the observation for +pan in the model, and the physical sign that produces +pan."""
    import mujoco

    lens = bench.lens_model
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    cid = m.camera("wrist_camera").id
    square = np.array([*bench.square_center_xy, bench.square_thickness_m])

    def project_x(qpos):
        for name, value in zip(JOINT_NAMES, qpos):
            d.qpos[m.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(m, d)
        cpos = d.cam_xpos[cid]; R = d.cam_xmat[cid].reshape(3, 3)
        pc = R.T @ (square - cpos)
        if pc[2] >= 0:
            return None
        x, y = pc[0] / -pc[2], -pc[1] / -pc[2]
        xd, _ = lens.distort(np.array([x]), np.array([y]))
        return float(lens.cx + lens.fx * xd[0])

    base = np.asarray(bench.reset_qpos, dtype=np.float64)
    plus = base.copy(); plus[0] += 0.1
    x0, x1 = project_x(base), project_x(plus)
    direction = "right" if x1 > x0 else "left"     # image x grows to the right
    # Physical delta sign that yields +pan in the model (through the joint map).
    physical = np.asarray(bench.reset_physical, dtype=np.float32)
    up = physical.copy(); up[0] += 1.0
    dq = bench.joint_map_object.act_to_mujoco(physical_normalized_to_act(up))[0] - bench.joint_map_object.act_to_mujoco(physical_normalized_to_act(physical))[0]
    return dict(model_plus_pan_moves_square=direction, physical_delta_sign_for_plus_pan=int(np.sign(dq)),
                square_x_at_reset_px=x0, square_x_at_plus_pan_px=x1)


def run_pan_sign_check(robot, grabber, bench: BenchConfig, expectation: dict[str, Any], *, preview=None,
                       seconds: float = 45.0, read=read_measured_act, clock=time.time) -> dict[str, Any]:
    """Read-only: the user rotates the base so the square moves the shown way; the pan reading's sign must match."""
    direction = expectation["model_plus_pan_moves_square"]
    baseline = act_to_physical_normalized(read(robot))[0]
    text = (f"PAN SIGN CHECK (torque off): rotate the base BY HAND so the white square moves to the {direction.upper()} "
            f"in the live view by a few centimetres, then hold. Waiting up to {seconds:.0f}s for a {PAN_CHECK_UNITS:.0f}-unit pan change.")
    print(text, flush=True)
    if preview is not None:
        w = bench.lens_model.image_size[0]
        box = [w - 300, 300, w - 20, 780] if direction == "right" else [20, 300, 300, 780]
        preview.set_overlays([dict(box=box, label=f"move the square this way ({direction})", color=(255, 220, 0))])
        preview.set_status(text)
    deadline = clock() + seconds
    delta = 0.0
    while clock() < deadline:
        delta = act_to_physical_normalized(read(robot))[0] - baseline
        if abs(delta) >= PAN_CHECK_UNITS:
            break
        time.sleep(0.1)
    measured_sign = int(np.sign(delta)) if abs(delta) >= PAN_CHECK_UNITS else 0
    expected_sign = int(expectation["physical_delta_sign_for_plus_pan"])
    result = dict(direction_requested=direction, baseline_pan=float(baseline), delta_pan=float(delta),
                  measured_sign=measured_sign, expected_sign=expected_sign, passed=bool(measured_sign == expected_sign and measured_sign != 0))
    if preview is not None:
        preview.set_overlays([])
    return result


def send_physical(robot, physical: np.ndarray) -> np.ndarray:
    action = {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, physical)}
    returned = robot.send_action(action)
    sent = np.asarray([float(returned[f"{name}.pos"]) for name in JOINT_NAMES], dtype=np.float64)
    if not np.allclose(sent, physical, atol=1e-5, rtol=0):
        raise RuntimeError(f"driver modified the approach command {physical.tolist()} -> {sent.tolist()}")
    return sent


def run_approach(robot, bench: BenchConfig, log_path: Path, *, read=read_measured_act, sleep=time.sleep,
                 clock=time.monotonic, timeout_s: float = APPROACH_TIMEOUT_S, on_status=print) -> dict[str, Any]:
    """Gated ramp from the present pose to the recorded reset; torque must already be on. Leaves torque on."""
    calibration = load_physical_calibration()
    period = 1.0 / CONTROL_HZ
    current = read(robot)
    previous = act_to_physical_normalized(current).astype(np.float32)
    started = clock()
    steps = 0
    with log_path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["monotonic", *[f"measured_{n}" for n in JOINT_NAMES], *[f"target_{n}" for n in JOINT_NAMES], "hold_reason", "settled"])
        while True:
            t = clock()
            current = read(robot)
            step = approach_step(current, previous, bench, calibration=calibration)
            writer.writerow([f"{t - started:.3f}", *[f"{v:.3f}" for v in step.measured_physical], *[f"{v:.3f}" for v in step.target_physical], step.reason, int(step.settled)])
            handle.flush()
            if step.settled:
                break
            if t - started > timeout_s:
                raise RuntimeError(f"approach timed out after {timeout_s:.0f}s; no further commands sent (torque stays on)")
            send_physical(robot, step.target_physical.astype(np.float64))
            previous = step.target_physical
            steps += 1
            if steps % 30 == 0:
                on_status(f"approach {t - started:4.1f}s: measured {np.round(step.measured_physical, 2).tolist()}")
            elapsed = clock() - t
            if elapsed < period:
                sleep(period - elapsed)
    residual = act_to_physical_normalized(read(robot)) - np.asarray(bench.reset_physical, dtype=np.float32)
    return dict(steps=steps, seconds=round(clock() - started, 2), residual_physical=[round(float(v), 3) for v in residual])


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--run-dir", type=Path, default=None, help="new directory for the evidence (required for motion and rehearsal)")
    p.add_argument("--robot-port", default="/dev/ttyACM0")
    p.add_argument("--camera-device", type=int, default=0)
    p.add_argument("--camera-size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"),
                   help="capture size; the camera scales the same field of view, and 1080p cannot stream at rate over USB-over-IP (3-6 fps vs 29 at 720p; zero-shift equivalence verified 2026-09-09)")
    p.add_argument("--max-actions", type=int, default=480)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--enable-motion", action="store_true")
    p.add_argument("--sim-rehearsal", action="store_true")
    p.add_argument("--skip-approach", action="store_true", help="the arm is already at the reset pose")
    p.add_argument("--pan-check", action="store_true", help="run the hand-rotation pan-sign check at preflight (verified 2026-09-09: physical/pan_sign_check_20260909.json; off by default)")
    p.add_argument("--skip-pan-check", action="store_true", help=argparse.SUPPRESS)  # legacy no-op: the check is opt-in now
    p.add_argument("--pan-check-seconds", type=float, default=45.0, help="how long to wait for the hand rotation")
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args(argv)
    modes = int(args.preflight_only) + int(args.enable_motion) + int(args.sim_rehearsal)
    if modes != 1:
        p.error("choose exactly one of --preflight-only, --enable-motion, --sim-rehearsal")
    if (args.enable_motion or args.sim_rehearsal) and args.run_dir is None:
        p.error("--run-dir is required for motion and rehearsal")

    bench = scene_bench_config(args.model)
    if bench is None or bench.lens is None or bench.reset_physical is None:
        raise SystemExit("the bench scene must carry a lens block and a recorded reset")
    scene_hash = scene_dependency_hash(args.model)
    import torch
    torch.set_num_threads(1)
    policy = VisionChunkedPolicy(args.checkpoint, black_image=False, clamp_channels=(5,))
    record: dict[str, Any] = dict(
        status="started", started_at=now_iso(), mode=("sim_rehearsal" if args.sim_rehearsal else "preflight" if args.preflight_only else "motion"),
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint), report_content_sha256=policy.report_content_sha256,
        policy_id=policy.policy_id, chunk_horizon=policy.chunk_horizon, teacher_horizon=policy.teacher_horizon,
        scene_dependencies_sha256=scene_hash, bench=asdict(bench), lens=bench.lens,
        observation_contract="raw MJPEG frame at the camera's full field of view (calibrated at 1920x1080; captured at --camera-size) -> BGR->RGB -> exact area filter -> 256x256 (no undistortion, no crop)",
        calibration=dict(resource="physical_inference_calibration_20260620.json",
                         resource_sha256=hashlib.sha256(read_resource_bytes("physical_inference_calibration_20260620.json")).hexdigest(),
                         live_file=str(LIVE_CALIBRATION), live_sha256=(sha256_file(LIVE_CALIBRATION) if LIVE_CALIBRATION.exists() else None)),
        git=git_state(), torch=dict(version=torch.__version__, threads=torch.get_num_threads()), control_hz=CONTROL_HZ,
        max_actions=int(args.max_actions), confirmations=[], holds=None,
    )
    contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=bench)

    if args.sim_rehearsal:
        return _sim_rehearsal(args, bench, contract, policy, record)
    return _hardware(args, bench, contract, policy, record)


def _finish(run_dir: Path | None, record: dict[str, Any], status: str, **extra) -> None:
    record.update(status=status, finished_at=now_iso(), **extra)
    if run_dir is not None:
        write_run_record(run_dir / "run.json", record)
    print(f"[{status}] " + (f"run record: {run_dir / 'run.json'}" if run_dir else json.dumps({k: extra[k] for k in extra})), flush=True)


def _sim_rehearsal(args, bench, contract, policy, record) -> int:
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = MujocoTaskAdapter(args.model)
    suite = bench_suite(bench, [(0, 0)], label="rehearsal", repeats=1)
    backend = SimBackend(adapter, suite.scenarios[0], contract=contract)
    log = StepLog(run_dir / "steps.csv")
    store = BoundaryStore()
    try:
        result = run_episode(backend, policy, bench=bench, max_actions=args.max_actions, on_step=lambda r: (log.write(r), store.add(r)))
        boundaries = store.write(run_dir / "boundaries")
        _finish(run_dir, record, "completed", backend="simulation", actions=result.actions, success=result.success,
                invalidated=result.invalidated, hold_frames=result.hold_frames, aborted_reason=result.aborted_reason,
                boundaries=boundaries, safety_counts=backend.counts, steps_csv_sha256=sha256_file(run_dir / "steps.csv"))
        print(f"rehearsal: success={result.success} actions={result.actions} holds={result.hold_frames} counts={backend.counts}")
        return 0 if result.success else 2
    finally:
        log.close(); backend.close()


def _real_frame_gate(policy, grabber, bench, source_size, current, run_dir, label: str) -> dict[str, Any]:
    """Grab a fresh frame, resample it as the runner would, and run the real-frame gate on it (nothing is sent)."""
    _seq, _stamp, frame = grabber.wait_for_new(-1)
    observation = fast_area_resampler(bench.lens_model, source_size=source_size)(np.array(frame, copy=True))
    result = check_reset_frame(policy, observation, current, bench, label=f"{label}_observation")
    result["dry_pass"] = {k: v for k, v in result["dry_pass"].items() if k != "chunk_physical"}
    if run_dir is not None:
        from PIL import Image
        (run_dir / "preflight").mkdir(exist_ok=True)
        path = run_dir / "preflight" / f"real_frame_gate_{label}.png"
        Image.fromarray(observation).save(path)
        result["observation_png"] = str(path.relative_to(run_dir))
        result["observation_array_sha256"] = hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
    print(f"real-frame gate ({label}): {'PASS' if result['passed'] else 'FAIL'} {result['dry_pass']['max_abs_delta_from_start_units']} holds={result['dry_pass']['holds_in_dry_chunk']} {result['reasons']}", flush=True)
    return result


def _hardware(args, bench, contract, policy, record) -> int:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from camera_preview import PreviewWindow, _HeadlessPreview
    config = SO101FollowerConfig(id="None", port=args.robot_port, cameras={}, max_relative_target=20.0, use_degrees=False)
    robot = SO101Follower(config)
    assert_pinned_calibration(robot)                     # before the bus opens
    run_dir = args.run_dir
    if run_dir is not None:
        if run_dir.exists():
            raise SystemExit(f"run directory already exists: {run_dir} (each attempt gets a new one; it may hold a previous attempt's evidence)")
        run_dir.mkdir(parents=True, exist_ok=False)      # claimed before torque
    grabber = None
    connected = False
    recorder = None
    backend = None
    store = None
    status = "started"
    try:
        grabber = FrameGrabber(args.camera_device, width=int(args.camera_size[0]), height=int(args.camera_size[1]))
        record["camera"] = dict(device=f"/dev/video{args.camera_device}", requested_size=list(args.camera_size), **grabber.properties)
        if (grabber.properties["width"], grabber.properties["height"]) != tuple(int(v) for v in args.camera_size):
            raise Refused(f"camera delivered {grabber.properties['width']}x{grabber.properties['height']}, not the requested size")
        connect_read_only(robot)
        connected = True
        current = read_measured_act(robot)
        physical = act_to_physical_normalized(current)
        record["measured_pose_physical"] = [round(float(v), 3) for v in physical]
        print(f"measured pose (physical units): {np.round(physical, 2).tolist()}", flush=True)
        # Servo supply (read-only register). The 2026-09-09 supply read 5.4 V: gripper voltage error, elbow sag.
        record["servo_voltage"] = check_servo_voltage(robot)
        print(f"servo voltage: {record['servo_voltage']}", flush=True)
        if not record["servo_voltage"]["ok"]:
            raise Refused(f"servo supply {record['servo_voltage']['lowest_v']:.1f} V is below {MIN_SERVO_VOLTAGE_V:.1f} V "
                          f"(per motor: {record['servo_voltage']['volts']}); fix the power supply before any motion")
        # Camera and observation contract.
        rate = grabber.measure_rate(1.0)
        record["camera"]["measured_fps"] = round(rate, 1)
        if rate < 25:
            raise Refused(f"camera delivers {rate:.1f} frames/s (< 25)")
        frames = []
        seq = -1
        for _ in range(3):
            seq, _stamp, frame = grabber.wait_for_new(seq)
            frames.append(np.array(frame, copy=True))
        record["resampler"] = verify_fast_resampler(bench.lens_model, frames)
        source_size = (frames[-1].shape[1], frames[-1].shape[0])
        observation = fast_area_resampler(bench.lens_model, source_size=source_size)(frames[-1])
        if run_dir is not None:
            from PIL import Image
            (run_dir / "preflight").mkdir(exist_ok=True)
            Image.fromarray(observation).save(run_dir / "preflight" / "observation.png")
            Image.fromarray(np.ascontiguousarray(frames[-1][:, :, ::-1])).save(run_dir / "preflight" / "raw.png")
        preview = _HeadlessPreview() if args.no_preview else PreviewWindow("physical episode: preflight", grabber)
        # Pan sign (read-only, torque off).
        expectation = pan_sign_expectation(args.model, bench)
        record["pan_sign"] = dict(expectation=expectation)
        if not args.pan_check:
            record["pan_sign"]["skipped"] = "verified 2026-09-09 (physical/pan_sign_check_20260909.json); pass --pan-check to repeat"
        else:
            import threading
            outcome = {}
            worker = threading.Thread(target=lambda: outcome.update(run_pan_sign_check(robot, grabber, bench, expectation, preview=preview, seconds=args.pan_check_seconds)), daemon=True)
            worker.start()
            preview.run(until=lambda: not worker.is_alive())
            worker.join(timeout=5.0)
            record["pan_sign"].update(outcome)
            print(f"pan-sign check: {outcome}", flush=True)
            if not outcome.get("passed"):
                raise Refused("pan sign not confirmed (rotate the base the shown way; a mismatch means the joint map's pan sign is wrong)")
        # Policy dry pass on the live observation (nothing is sent; informational: the arm is usually at gravity rest here).
        record["dry_pass"] = {k: v for k, v in policy_dry_pass(policy, observation, current, bench).items() if k != "chunk_physical"}
        print(f"policy dry pass: {record['dry_pass']}", flush=True)
        record["start_pose_delta_act"] = float(np.max(np.abs(current - physical_normalized_to_act(np.asarray(bench.reset_physical, np.float32)))))
        record["approach_plan"] = approach_plan(current, bench)
        if args.preflight_only:
            if record["start_pose_delta_act"] <= START_POSE_TOLERANCE_ACT:
                # At the reset pose already: the real-frame gate applies to this frame.
                record["real_frame_check"] = _real_frame_gate(policy, grabber, bench, source_size, current, run_dir, "preflight")
                if not record["real_frame_check"]["passed"]:
                    raise Refused("policy fails the real-frame gate at the reset pose: " + "; ".join(record["real_frame_check"]["reasons"]))
            else:
                record["real_frame_check"] = dict(skipped="arm not at the reset pose; the gate runs after the approach in motion mode")
            _finish(run_dir, record, "preflight_ok")
            return 0
        # ---- motion
        print("APPROACH PLAN: " + json.dumps(record["approach_plan"]), flush=True)
        answer = input("Enable torque at the present pose and run the gated approach to the reset pose? Press Enter to confirm, anything else aborts: ")
        record["confirmations"].append(dict(phase="approach", at=now_iso(), answer=answer))
        if answer.strip():
            _finish(run_dir, record, "aborted_by_user"); return 0
        backend = LeRobotBackend(robot, grabber, lens=bench.lens_model, source_size=source_size)
        current = read_measured_act(robot)
        backend.engage(act_to_physical_normalized(current))   # goal = present pose, then torque
        if not args.skip_approach:
            record["approach"] = run_approach(robot, bench, run_dir / "approach.csv")
            print(f"approach done: {record['approach']}", flush=True)
        current = read_measured_act(robot)
        delta = float(np.max(np.abs(current - physical_normalized_to_act(np.asarray(bench.reset_physical, np.float32)))))
        record["start_pose_delta_act"] = delta
        if delta > START_POSE_TOLERANCE_ACT:
            raise Refused(f"start pose differs from the recorded reset by {delta:.3f} ACT (> {START_POSE_TOLERANCE_ACT})")
        if act_to_physical_normalized(current)[1] < bench.shoulder_floor:
            raise Refused("shoulder is below the floor at the start of the episode")
        # Real-frame gate on a fresh frame at the reset pose: the chunk must be hold-like (the arm keeps holding torque on refusal).
        record["real_frame_check"] = _real_frame_gate(policy, grabber, bench, source_size, current, run_dir, "reset")
        if not record["real_frame_check"]["passed"]:
            raise Refused("policy fails the real-frame gate at the reset pose (no episode): " + "; ".join(record["real_frame_check"]["reasons"]))
        answer = input(f"Arm is at the reset pose (max delta {delta:.3f} ACT). Run the {args.max_actions}-action episode? Press Enter to confirm, anything else aborts: ")
        record["confirmations"].append(dict(phase="episode", at=now_iso(), answer=answer))
        if answer.strip():
            _finish(run_dir, record, "aborted_by_user"); return 0
        recorder = VideoRecorder(grabber, run_dir / "camera.mp4")
        recorder.start()
        log = StepLog(run_dir / "steps.csv")
        store = BoundaryStore()
        prefix = {}
        record["episode_started_at"] = now_iso()

        def on_step(r):
            log.write(r, r.step / CONTROL_HZ)
            store.add(r)
            if r.step == bench.observation_steps - 1:
                qpos = bench.joint_map_object.act_to_mujoco(r.current_act)[:5]
                prefix.update(step=r.step, max_abs_rad=float(np.max(np.abs(qpos - np.asarray(bench.viewing_qpos)[:5]))))
                prefix["ok"] = prefix["max_abs_rad"] <= PREFIX_TOLERANCE_RAD

        try:
            result = run_episode(backend, policy, bench=bench, max_actions=args.max_actions, on_step=on_step)
        finally:
            log.close()
        status = "completed" if result.aborted_reason is None else "aborted"
        record.update(actions=result.actions, hold_frames=result.hold_frames, aborted_reason=result.aborted_reason,
                      boundaries_steps=result.boundaries, prefix=prefix,
                      timing=dict(overruns=backend.overruns, reanchors=backend.reanchors))
        print(f"episode: actions={result.actions} holds={result.hold_frames} aborted={result.aborted_reason} overruns={backend.overruns}", flush=True)
        return 0 if result.aborted_reason is None else 2
    except Refused as exc:
        status = "refused"; record["reason"] = str(exc); print(f"REFUSED: {exc}", flush=True); return 2
    except KeyboardInterrupt:
        status = "interrupted"; print("interrupted: torque stays on; support the arm before removing power", flush=True); return 130
    except Exception as exc:
        status = "aborted"; record["reason"] = f"{type(exc).__name__}: {exc}"; print(f"ABORTED: {record['reason']}", flush=True); return 2
    finally:
        if recorder is not None:
            record["video"] = recorder.stop()
        if store is not None and run_dir is not None:
            record["boundaries"] = store.write(run_dir / "boundaries")
            if (run_dir / "steps.csv").exists():
                record["steps_csv_sha256"] = sha256_file(run_dir / "steps.csv")
        if connected:
            disconnect_read_only(robot)    # torque preserved
        if grabber is not None:
            grabber.close()
        if status not in ("started",):
            _finish(run_dir, record, status)


if __name__ == "__main__":
    raise SystemExit(main())
