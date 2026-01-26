#!/usr/bin/env python3
"""
Run SmolVLA inference on a physical SO-101 arm using a wrist camera.

This script mirrors the inference flow in:
- simulation_code/run_mujoco_simulation.py
- simulation_code/run_reinflow_inference.py

and uses the physical control patterns from:
- imitation-learning/physical-so101-teleop/0_so100_keyboard_joint_control.py

Key assumptions:
- Dataset was recorded with record_single_arm.py
- observation.state is in radians using the same conversion as record_single_arm.py
- observation.images.wrist is RGB uint8 (H, W, C)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from transformers import AutoTokenizer

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_defaults_from_config(args: argparse.Namespace) -> None:
    if not args.config_path:
        return
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as f:
        cfg = json.load(f)

    robot_cfg = cfg.get("robot", {})
    cam_cfg = cfg.get("camera", {})

    if args.robot_port is None:
        args.robot_port = robot_cfg.get("port")
    if args.robot_id is None:
        args.robot_id = robot_cfg.get("id")

    if args.camera_index is None:
        args.camera_index = cam_cfg.get("device")
    if args.camera_capture_width is None:
        args.camera_capture_width = cam_cfg.get("capture_width")
    if args.camera_capture_height is None:
        args.camera_capture_height = cam_cfg.get("capture_height")
    if args.camera_target_width is None:
        args.camera_target_width = cam_cfg.get("target_width")
    if args.camera_target_height is None:
        args.camera_target_height = cam_cfg.get("target_height")
    if args.camera_fps is None:
        args.camera_fps = cfg.get("recording", {}).get("fps")

    if args.task is None:
        args.task = cfg.get("task_description")

    if args.dataset_repo_id is None:
        args.dataset_repo_id = cfg.get("hub", {}).get("repo_id")


def _find_calibration_file(robot_id: str | None, robot_type: str = "so101_follower") -> str | None:
    """
    Find calibration file with fallback patterns, matching record_single_arm.py logic.
    
    Args:
        robot_id: Robot identifier from config/args
        robot_type: Robot type folder under calibration/robots (e.g., so101_follower)
    
    Returns:
        The robot_id that matches an existing calibration file, or None if none found.
    """
    if robot_id is None:
        robot_id = "None"
    
    cal_dir = (
        Path.home() / ".cache" / "huggingface" / "lerobot"
        / "calibration" / "robots" / robot_type
    )

    # Try robot_id.json first, then fallback patterns (matching record_single_arm.py)
    candidates = [
        cal_dir / f"{robot_id}.json",
        cal_dir / "so101_follower_1.json",
        cal_dir / "None.json",
    ]

    for cal_path in candidates:
        if cal_path.exists():
            # Extract the robot_id from the filename
            found_id = cal_path.stem
            if found_id != robot_id:
                logging.info(f"Found calibration file: {cal_path} (using id '{found_id}' instead of '{robot_id}')")
            else:
                logging.info(f"Found calibration file: {cal_path}")
            return found_id

    logging.warning(f"No calibration file found in {cal_dir}")
    logging.warning(f"Tried: {[c.name for c in candidates]}")
    return None


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _motor_obs_to_state_radians(obs: dict[str, Any]) -> np.ndarray:
    """Convert motor observations to radians using record_single_arm.py conventions."""
    raw = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)
    state[:5] = raw[:5] / 100.0 * np.pi
    state[5] = raw[5] / 100.0 * 1.7
    return state


def _action_radians_to_motor(action_rad: np.ndarray) -> dict[str, float]:
    """Convert radians action back to motor normalized units (-100..100, gripper 0..100)."""
    action = np.array(action_rad, dtype=np.float32).reshape(-1)
    if action.size != 6:
        raise ValueError(f"Expected action dim 6, got {action.size}")

    motor_cmd = {}
    for i, name in enumerate(JOINT_NAMES):
        if i < 5:
            val = (action[i] / np.pi) * 100.0
            val = float(np.clip(val, -100.0, 100.0))
        else:
            val = (action[i] / 1.7) * 100.0
            val = float(np.clip(val, 0.0, 100.0))
        motor_cmd[f"{name}.pos"] = val
    return motor_cmd


def _smooth_action(
    current: dict[str, float],
    target: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """Blend target toward current positions (alpha in [0,1])."""
    if alpha <= 0:
        return target
    smoothed = {}
    for key, tgt in target.items():
        cur = current.get(key, tgt)
        smoothed[key] = float(cur + alpha * (tgt - cur))
    return smoothed


def _load_dataset_stats(dataset_repo_id: str | None, dataset_root: str | None) -> dict | None:
    if not dataset_repo_id:
        return None
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    return ds_meta.stats


def _ensure_tokenizer(policy: SmolVLAPolicy) -> None:
    if getattr(policy, "tokenizer", None) is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    policy.tokenizer = tokenizer


def _build_processors(
    policy: SmolVLAPolicy,
    dataset_stats: dict | None,
    policy_repo: str,
) -> tuple[Any, Any]:
    # Prefer loading processors from the pretrained repo, fallback to building from config.
    pre_overrides = {"device_processor": {"device": policy.config.device}}
    try:
        return make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_repo,
            dataset_stats=dataset_stats,
            preprocessor_overrides=pre_overrides,
        )
    except Exception:
        return make_pre_post_processors(
            policy_cfg=policy.config,
            dataset_stats=dataset_stats,
            preprocessor_overrides=pre_overrides,
        )


def _select_image_key(policy: SmolVLAPolicy) -> str:
    image_keys = list(policy.config.image_features.keys())
    if "observation.images.wrist" in image_keys:
        return "observation.images.wrist"
    if image_keys:
        return image_keys[0]
    raise ValueError("Policy config has no image features. Cannot run vision policy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolVLA on a physical SO-101 arm")
    parser.add_argument("--policy-repo", required=True, help="Hugging Face repo id for the policy")
    parser.add_argument("--dataset-repo-id", default=None, help="HF dataset repo id for stats")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset path for stats")
    parser.add_argument("--config-path", default=None, help="Optional config.json to fill defaults")

    parser.add_argument("--robot-port", default=None, help="Serial port for the robot")
    parser.add_argument("--robot-id", default=None, help="Robot id (used for calibration files)")
    parser.add_argument("--camera-index", type=int, default=None, help="OpenCV camera index")
    parser.add_argument("--camera-capture-width", type=int, default=None, help="Camera capture width")
    parser.add_argument("--camera-capture-height", type=int, default=None, help="Camera capture height")
    parser.add_argument("--camera-target-width", type=int, default=None, help="Model input width")
    parser.add_argument("--camera-target-height", type=int, default=None, help="Model input height")
    parser.add_argument("--camera-fps", type=int, default=None, help="Camera FPS")

    parser.add_argument("--task", default=None, help="Task instruction string")
    parser.add_argument("--device", default=None, help="torch device: cuda, cpu, mps")
    parser.add_argument("--control-hz", type=float, default=10.0, help="Control loop frequency")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="Action smoothing factor (0-1)")
    parser.add_argument("--max-relative-target", type=float, default=10.0, help="Safety cap per step")

    args = parser.parse_args()
    _load_defaults_from_config(args)

    if args.robot_port is None:
        raise ValueError("robot port is required (set --robot-port or config.json)")
    if args.camera_index is None:
        raise ValueError("camera index is required (set --camera-index or config.json)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Find calibration file with fallback patterns (matching record_single_arm.py)
    found_robot_id = _find_calibration_file(args.robot_id)
    if found_robot_id is not None:
        args.robot_id = found_robot_id
    else:
        logging.warning("No calibration file found. Calibration will be prompted on connect.")

    device = _select_device(args.device)
    logging.info("Using device: %s", device)

    # Robot + camera setup
    cam_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        fps=args.camera_fps or 30,
        width=args.camera_capture_width or 640,
        height=args.camera_capture_height or 480,
    )
    robot_cfg = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        cameras={"wrist": cam_cfg},
        max_relative_target=args.max_relative_target,
        use_degrees=False,
    )
    robot = SO101Follower(robot_cfg)

    # Model + processors
    policy = SmolVLAPolicy.from_pretrained(args.policy_repo)
    policy.to(device)
    policy.eval()
    policy.reset()
    policy.config.device = device.type
    _ensure_tokenizer(policy)
    image_key = _select_image_key(policy)
    if image_key != "observation.images.wrist":
        logging.info("Mapping wrist camera to policy image key: %s", image_key)

    dataset_stats = _load_dataset_stats(args.dataset_repo_id, args.dataset_root)
    preprocessor, postprocessor = _build_processors(policy, dataset_stats, args.policy_repo)

    # Control loop
    control_dt = 1.0 / max(args.control_hz, 1.0)
    logging.info("Starting control loop at %.2f Hz", args.control_hz)

    robot.connect()
    try:
        while True:
            loop_start = time.time()
            obs = robot.get_observation()

            rgb = obs["wrist"]
            if args.camera_target_width and args.camera_target_height:
                tgt_w = args.camera_target_width
                tgt_h = args.camera_target_height
                if rgb.shape[1] != tgt_w or rgb.shape[0] != tgt_h:
                    rgb = cv2.resize(rgb, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            state = _motor_obs_to_state_radians(obs)

            observation = {
                image_key: rgb,
                "observation.state": state,
            }

            # Prepare for inference
            model_obs = prepare_observation_for_inference(
                observation,
                device=device,
                task=args.task,
                robot_type="so101_follower",
            )
            model_obs = preprocessor(model_obs)

            with torch.inference_mode():
                action = policy.select_action(model_obs)
                action = postprocessor(action)

            if torch.is_tensor(action):
                action_np = action.detach().cpu().numpy().squeeze()
            else:
                action_np = np.asarray(action)

            target_cmd = _action_radians_to_motor(action_np)
            current_cmd = {f"{name}.pos": float(obs[f"{name}.pos"]) for name in JOINT_NAMES}
            cmd = _smooth_action(current_cmd, target_cmd, args.smooth_alpha)

            robot.send_action(cmd)

            elapsed = time.time() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        logging.info("Stopping inference loop")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
