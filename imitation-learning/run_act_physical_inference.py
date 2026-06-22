#!/usr/bin/env python3
"""
Run ACT inference on a physical SO-101 follower arm with a wrist camera.

The script is conservative by default: it reads hardware and predicts actions,
but it does not command motors unless --enable-motion is passed and confirmed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "simulation_code"
    / "outputs"
    / "train"
    / "act_so101_physical"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)
DEFAULT_CALIBRATION_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "so101_pickplace_v1"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="ACT pretrained_model path.")
    parser.add_argument("--robot-port", default="/dev/ttyUSB0", help="WSL serial device for the SO-101 arm.")
    parser.add_argument("--robot-id", default="None", help="Robot id matching the calibration filename stem.")
    parser.add_argument("--calibration-path", type=Path, default=None, help="Explicit LeRobot calibration JSON.")
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
        help="Root containing LeRobot calibration/robots folders.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV wrist camera index.")
    parser.add_argument("--camera-capture-width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--camera-capture-height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--camera-target-width", type=int, default=256, help="ACT wrist image width.")
    parser.add_argument("--camera-target-height", type=int, default=256, help="ACT wrist image height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Physical dataset used for start-pose diagnostics.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for ACT inference.")
    parser.add_argument("--control-hz", type=float, default=10.0, help="Control loop frequency.")
    parser.add_argument("--max-steps", type=int, default=0, help="Max loop steps; 0 means run until Ctrl-C.")
    parser.add_argument("--max-relative-target", type=float, default=5.0, help="LeRobot relative target safety cap.")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="Blend factor toward target command.")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Scale delta from current motor pose to target.")
    parser.add_argument("--enable-motion", action="store_true", help="Actually send predicted commands to the arm.")
    parser.add_argument("--dry-run", action="store_true", help="Predict actions but do not move. This is the default.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate checkpoint/hardware, then exit.")
    parser.add_argument("--list-devices", action="store_true", help="List WSL/Windows serial and camera hints, then exit.")
    parser.add_argument("--wsl-attach-help", action="store_true", help="Print Windows usbipd commands for attaching USB devices to WSL, then exit.")
    parser.add_argument("--save-video", type=Path, default=None, help="Directory for wrist-camera MP4 output.")
    parser.add_argument("--log-actions", type=Path, default=None, help="CSV path for observations/actions/commands.")
    parser.add_argument("--verbose", action="store_true", help="Print per-step state/action details.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0")
    if args.max_relative_target <= 0:
        raise ValueError("--max-relative-target must be > 0")
    if not 0 <= args.smooth_alpha <= 1:
        raise ValueError("--smooth-alpha must be in [0, 1]")
    if args.action_scale <= 0:
        raise ValueError("--action-scale must be > 0")


def validate_checkpoint(checkpoint: Path) -> None:
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ACT checkpoint is missing required files at {checkpoint}: {', '.join(missing)}")
    empty = [name for name in required if (checkpoint / name).stat().st_size == 0]
    if empty:
        raise RuntimeError(f"ACT checkpoint has empty required files at {checkpoint}: {', '.join(empty)}")


def select_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return device


def serial_candidates() -> list[str]:
    patterns = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyS*"]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern)))
    return candidates


def video_candidates() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def opencv_camera_candidates(max_index: int = 8) -> list[str]:
    candidates = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        try:
            opened = cap.isOpened()
            if opened:
                ok, frame = cap.read()
                shape = None if not ok or frame is None else tuple(frame.shape)
                candidates.append(f"{index}: opened=True read={ok} shape={shape}")
        finally:
            cap.release()
    return candidates


def windows_device_hints() -> str:
    powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not Path(powershell).exists():
        return "powershell.exe not found; Windows device hints unavailable."
    command = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.Name -match 'USB|Serial|COM|Camera|Video|CH340|CH343|CP210|STM|Feetech|SO-101|SO101' } | "
        "Select-Object Name,DeviceID | Format-Table -AutoSize"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - host-specific diagnostic
        return f"Windows device hint command failed: {exc}"
    return (result.stdout or result.stderr).strip() or "No Windows device hints returned."


def windows_usbipd_list() -> str:
    powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not Path(powershell).exists():
        return "powershell.exe not found; usbipd list unavailable."
    command = "if (Get-Command usbipd -ErrorAction SilentlyContinue) { usbipd list } else { 'usbipd is not installed or is not on PATH.' }"
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - host-specific diagnostic
        return f"usbipd list command failed: {exc}"
    return (result.stdout or result.stderr).strip() or "usbipd list returned no output."


def print_device_report() -> None:
    print("WSL serial candidates:")
    for item in serial_candidates() or ["  none"]:
        print(f"  {item}")
    print("\nWSL video device candidates:")
    for item in video_candidates() or ["  none"]:
        print(f"  {item}")
    print("\nOpenCV camera candidates:")
    for item in opencv_camera_candidates() or ["  none"]:
        print(f"  {item}")
    print("\nWindows device hints:")
    print(windows_device_hints())


def print_wsl_attach_help() -> None:
    print("WSL USB attachment helper")
    print("=" * 60)
    print("This script cannot safely attach USB devices itself because usbipd bind/attach may require")
    print("Windows-side permissions and will detach the device from Windows while WSL owns it.")
    print()
    print("1. Install usbipd-win on Windows if needed:")
    print("   winget install --interactive --exact dorssel.usbipd-win")
    print()
    print("2. Open an Administrator PowerShell and list USB devices:")
    print("   usbipd list")
    print()
    print("3. Find these devices in the list:")
    print("   - SO-101 serial adapter: USB-Enhanced-SERIAL CH343, VID:PID 1A86:55D3")
    print("   - Wrist camera: USB camera/composite device, currently seen as VID:PID 0C45:6366")
    print()
    print("4. Share/bind each BUSID once from Administrator PowerShell:")
    print("   usbipd bind --busid <SERIAL_BUSID>")
    print("   usbipd bind --busid <CAMERA_BUSID>")
    print()
    print("5. Attach each device to WSL from PowerShell:")
    print("   usbipd attach --wsl --busid <SERIAL_BUSID>")
    print("   usbipd attach --wsl --busid <CAMERA_BUSID>")
    print()
    print("6. Back in WSL, verify:")
    print("   lsusb")
    print("   ls /dev/ttyUSB* /dev/ttyACM* /dev/video*")
    print("   python run_act_physical_inference.py --list-devices")
    print("   python run_act_physical_inference.py --preflight-only")
    print()
    print("7. To return a USB device to Windows:")
    print("   usbipd detach --busid <BUSID>")
    print()
    print("Current usbipd list output:")
    print("-" * 60)
    print(windows_usbipd_list())
    print()
    print("Current Windows USB hints:")
    print("-" * 60)
    print(windows_device_hints())


def resolve_calibration(args: argparse.Namespace) -> tuple[str, Path]:
    if args.calibration_path is not None:
        cal_path = args.calibration_path.expanduser().resolve()
        if not cal_path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {cal_path}")
        return cal_path.stem, cal_path.parent

    cal_dir = args.calibration_root.expanduser() / "so101_follower"
    robot_id = args.robot_id if args.robot_id is not None else "None"
    candidates = [
        cal_dir / f"{robot_id}.json",
        cal_dir / "so101_follower_1.json",
        cal_dir / "None.json",
    ]
    candidates = list(dict.fromkeys(candidates))
    for cal_path in candidates:
        if cal_path.is_file():
            if cal_path.stem != robot_id:
                logging.info("Using calibration file %s instead of requested robot id %s", cal_path, robot_id)
            return cal_path.stem, cal_path.parent

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No SO-101 calibration file found. Missing calibration is treated as a blocker. "
        f"Tried: {tried}"
    )


def host_device_errors(args: argparse.Namespace, require_camera: bool = True) -> list[str]:
    errors = []
    if not Path(args.robot_port).exists():
        serials = ", ".join(serial_candidates()) or "none"
        errors.append(
            f"Robot port does not exist in WSL: {args.robot_port}. "
            f"Visible serial candidates: {serials}. Attach the CH343 adapter into WSL first."
        )

    if require_camera:
        cap = cv2.VideoCapture(args.camera_index)
        try:
            if not cap.isOpened():
                videos = ", ".join(video_candidates()) or "none"
                errors.append(
                    f"OpenCV camera index {args.camera_index} did not open. "
                    f"Visible /dev/video* devices: {videos}. Attach the wrist camera into WSL first."
                )
            else:
                ok, _frame = cap.read()
                if not ok:
                    errors.append(f"OpenCV camera index {args.camera_index} opened but did not return a frame.")
        finally:
            cap.release()
    return errors


def preflight_host_devices(args: argparse.Namespace, require_camera: bool = True) -> None:
    errors = host_device_errors(args, require_camera=require_camera)
    if errors:
        raise RuntimeError("\n  - ".join(errors))


def load_policy_and_processors(checkpoint: Path, device: torch.device) -> tuple[Any, Any, Any]:
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.config.device = device.type
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    return policy, preprocessor, postprocessor


def motor_obs_to_state_radians(obs: dict[str, Any]) -> np.ndarray:
    raw = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)
    state[:5] = raw[:5] / 100.0 * np.pi
    state[5] = raw[5] / 100.0 * 1.7
    return state


def action_radians_to_motor(action_rad: np.ndarray) -> dict[str, float]:
    action = np.asarray(action_rad, dtype=np.float32).reshape(-1)
    if action.size != 6:
        raise ValueError(f"Expected action dimension 6, got {action.size}")

    command: dict[str, float] = {}
    for i, name in enumerate(JOINT_NAMES):
        if i < 5:
            value = float(np.clip((action[i] / np.pi) * 100.0, -100.0, 100.0))
        else:
            value = float(np.clip((action[i] / 1.7) * 100.0, 0.0, 100.0))
        command[f"{name}.pos"] = value
    return command


def current_motor_command(obs: dict[str, Any]) -> dict[str, float]:
    return {f"{name}.pos": float(obs[f"{name}.pos"]) for name in JOINT_NAMES}


def smooth_action(current: dict[str, float], target: dict[str, float], alpha: float) -> dict[str, float]:
    if alpha <= 0:
        return dict(target)
    return {key: float(current[key] + alpha * (target[key] - current[key])) for key in target}


def apply_action_scale(current: dict[str, float], target: dict[str, float], scale: float) -> dict[str, float]:
    if scale == 1.0:
        return dict(target)
    return {key: float(current[key] + scale * (target[key] - current[key])) for key in target}


def resize_wrist(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)


def build_model_observation(obs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    wrist = resize_wrist(obs["wrist"], args.camera_target_width, args.camera_target_height)
    wrist_tensor = torch.from_numpy(wrist).to(torch.float32)
    if wrist_tensor.max() > 1.0:
        wrist_tensor = wrist_tensor / 255.0
    wrist_tensor = wrist_tensor.permute(2, 0, 1).contiguous()

    return {
        "observation.images.wrist": wrist_tensor,
        "observation.state": torch.from_numpy(motor_obs_to_state_radians(obs)).to(torch.float32),
    }


def predict_action(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    observation: dict[str, Any],
) -> np.ndarray:
    model_obs = preprocessor(observation)
    with torch.inference_mode():
        action = policy.select_action(model_obs)
        action = postprocessor(action)
    if torch.is_tensor(action):
        return action.detach().cpu().numpy().squeeze().astype(np.float32)
    return np.asarray(action, dtype=np.float32).squeeze()


def observation_video_frame(observation: dict[str, Any]) -> np.ndarray:
    image = observation["observation.images.wrist"]
    if torch.is_tensor(image):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        image_np = np.asarray(image)
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return image_np


def make_robot(args: argparse.Namespace, robot_id: str, calibration_dir: Path) -> SO101Follower:
    camera_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        fps=args.camera_fps,
        width=args.camera_capture_width,
        height=args.camera_capture_height,
    )
    robot_cfg = SO101FollowerConfig(
        id=robot_id,
        port=args.robot_port,
        cameras={"wrist": camera_cfg},
        max_relative_target=args.max_relative_target,
        use_degrees=False,
        calibration_dir=calibration_dir,
    )
    return SO101Follower(robot_cfg)


def maybe_confirm_motion(args: argparse.Namespace, obs: dict[str, Any]) -> bool:
    print_start_pose_diagnostic(args.dataset_path, motor_obs_to_state_radians(obs))
    if not args.enable_motion:
        print("Dry-run mode: predictions will be printed/logged, but robot.send_action will not be called.")
        return False
    print("\nMotion is enabled.")
    print("Current motor positions:")
    for key, value in current_motor_command(obs).items():
        print(f"  {key}: {value:.3f}")
    answer = input("Press Enter to start sending ACT commands, or type anything else to abort: ")
    if answer.strip():
        print("Motion aborted by user; continuing in dry-run mode.")
        return False
    return True


def dataset_start_pose_stats(dataset_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    data_files = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        return None
    starts = []
    for path in data_files:
        df = pd.read_parquet(path, columns=["observation.state", "episode_index", "frame_index"])
        df = df.sort_values(["episode_index", "frame_index"])
        for _episode, group in df.groupby("episode_index", sort=True):
            starts.append(np.asarray(group.iloc[0]["observation.state"], dtype=np.float32))
    if not starts:
        return None
    start_states = np.stack(starts)
    return start_states.mean(axis=0), start_states.std(axis=0)


def print_start_pose_diagnostic(dataset_path: Path, current_state: np.ndarray) -> None:
    stats = dataset_start_pose_stats(dataset_path.expanduser().resolve())
    if stats is None:
        print("Dataset start-pose diagnostic unavailable; could not read demo parquet files.")
        return
    mean, std = stats
    delta = current_state - mean
    z = np.abs(delta) / np.maximum(std, 1e-6)
    print("\nDataset start-pose diagnostic:")
    print(f"  current_state: {np.round(current_state, 4).tolist()}")
    print(f"  demo_start_mean: {np.round(mean, 4).tolist()}")
    print(f"  current_minus_demo_mean: {np.round(delta, 4).tolist()}")
    print(f"  max_abs_z: {float(np.max(z)):.2f}")
    if np.max(z) > 3.0:
        print("  warning: current pose is far from the recorded demo start distribution.")


def open_action_log(path: Path | None) -> tuple[Any | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    fieldnames = [
        "step",
        "timestamp",
        *[f"state_{name}" for name in JOINT_NAMES],
        *[f"action_rad_{name}" for name in JOINT_NAMES],
        *[f"command_{name}" for name in JOINT_NAMES],
        "motion_enabled",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def log_step(
    writer: csv.DictWriter | None,
    step: int,
    state: np.ndarray,
    action_rad: np.ndarray,
    command: dict[str, float],
    motion_enabled: bool,
) -> None:
    if writer is None:
        return
    row: dict[str, Any] = {"step": step, "timestamp": time.time(), "motion_enabled": int(motion_enabled)}
    row.update({f"state_{name}": float(state[i]) for i, name in enumerate(JOINT_NAMES)})
    row.update({f"action_rad_{name}": float(action_rad[i]) for i, name in enumerate(JOINT_NAMES)})
    row.update({f"command_{name}": float(command[f"{name}.pos"]) for name in JOINT_NAMES})
    writer.writerow(row)


def write_video(frames: list[np.ndarray], output_dir: Path | None, fps: int) -> None:
    if output_dir is None or not frames:
        return
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"act_physical_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved wrist video: {path}")


def run_fake_checkpoint_smoke(checkpoint: Path, device: torch.device) -> None:
    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint, device)
    fake_obs = build_model_observation(
        {
            "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            **{f"{name}.pos": 0.0 for name in JOINT_NAMES},
        },
        argparse.Namespace(camera_target_width=256, camera_target_height=256),
    )
    action = predict_action(policy, preprocessor, postprocessor, fake_obs)
    if action.shape != (6,):
        raise RuntimeError(f"Checkpoint smoke test expected action shape (6,), got {action.shape}")
    print(f"Checkpoint smoke action: {np.round(action, 4).tolist()}")


def run_preflight(args: argparse.Namespace, checkpoint: Path, device: torch.device) -> None:
    validate_checkpoint(checkpoint)
    run_fake_checkpoint_smoke(checkpoint, device)

    errors = []
    errors.extend(host_device_errors(args))

    robot_id = None
    calibration_dir = None
    try:
        robot_id, calibration_dir = resolve_calibration(args)
    except Exception as exc:
        errors.append(str(exc))

    if errors:
        raise RuntimeError("Physical ACT preflight blockers:\n  - " + "\n  - ".join(errors))

    robot = make_robot(args, robot_id, calibration_dir)
    print("Connecting to robot for preflight...")
    robot.connect()
    try:
        obs = robot.get_observation()
        model_obs = build_model_observation(obs, args)
        print("Preflight observation:")
        print(f"  state: {np.round(model_obs['observation.state'], 4).tolist()}")
        print(f"  wrist image: {model_obs['observation.images.wrist'].shape} {model_obs['observation.images.wrist'].dtype}")
    finally:
        robot.disconnect()
    print("Physical ACT preflight completed successfully.")


def run_loop(args: argparse.Namespace, checkpoint: Path, device: torch.device) -> None:
    validate_checkpoint(checkpoint)
    robot_id, calibration_dir = resolve_calibration(args)
    preflight_host_devices(args)

    logging.info("Loading ACT checkpoint: %s", checkpoint)
    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint, device)
    robot = make_robot(args, robot_id, calibration_dir)
    control_dt = 1.0 / args.control_hz
    frames: list[np.ndarray] = []
    if args.log_actions is None:
        args.log_actions = SCRIPT_DIR / "outputs" / f"act_physical_actions_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_handle, log_writer = open_action_log(args.log_actions)
    print(f"Logging physical ACT actions to: {args.log_actions}")
    gripper_predictions: list[float] = []

    logging.info("Connecting to robot on %s with camera index %s", args.robot_port, args.camera_index)
    robot.connect()
    try:
        first_obs = robot.get_observation()
        motion_enabled = maybe_confirm_motion(args, first_obs)
        policy.reset()

        step = 0
        while args.max_steps == 0 or step < args.max_steps:
            start = time.time()
            obs = first_obs if step == 0 else robot.get_observation()
            observation = build_model_observation(obs, args)
            state_tensor = observation["observation.state"]
            state = state_tensor.detach().cpu().numpy() if torch.is_tensor(state_tensor) else np.asarray(state_tensor)
            action_rad = predict_action(policy, preprocessor, postprocessor, observation)
            gripper_predictions.append(float(action_rad[5]))
            current_cmd = current_motor_command(obs)
            target_cmd = action_radians_to_motor(action_rad)
            target_cmd = apply_action_scale(current_cmd, target_cmd, args.action_scale)
            command = smooth_action(current_cmd, target_cmd, args.smooth_alpha)

            if motion_enabled:
                sent = robot.send_action(command)
            else:
                sent = command

            log_step(log_writer, step, state, action_rad, sent, motion_enabled)
            if args.save_video is not None:
                frames.append(observation_video_frame(observation))

            if args.verbose or not motion_enabled:
                print(
                    f"step={step:05d} motion={int(motion_enabled)} "
                    f"state={np.round(state, 3).tolist()} "
                    f"action_rad={np.round(action_rad, 3).tolist()} "
                    f"cmd={[round(sent[f'{name}.pos'], 2) for name in JOINT_NAMES]}"
                )

            if step > 0 and step % 30 == 0:
                recent = np.asarray(gripper_predictions[-30:], dtype=np.float32)
                if float(recent.max() - recent.min()) < 0.05:
                    print(
                        "warning: predicted gripper range over the last 30 steps is very small "
                        f"({float(recent.max() - recent.min()):.4f} rad)."
                    )

            step += 1
            elapsed = time.time() - start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopping physical ACT inference.")
    finally:
        try:
            robot.disconnect()
        finally:
            if log_handle is not None:
                log_handle.close()
            write_video(frames, args.save_video, args.camera_fps)


def main() -> int:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.list_devices:
        print_device_report()
        return 0

    if args.wsl_attach_help:
        print_wsl_attach_help()
        return 0

    checkpoint = args.checkpoint.expanduser().resolve()
    device = select_device(args.device)

    if args.preflight_only:
        run_preflight(args, checkpoint, device)
    else:
        run_loop(args, checkpoint, device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
