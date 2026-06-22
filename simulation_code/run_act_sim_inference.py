#!/usr/bin/env python3
"""
Run deterministic ACT inference in the SO-101 MuJoCo pick-place simulation.

The policy is loaded from a LeRobot ACT checkpoint trained by
`train_act_on_data.py`. The policy observation contract intentionally matches
the physical dataset: wrist image plus 6D joint state.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR
    / "outputs"
    / "train"
    / "act_so101_corrected_30_b32_20260621_160923"
    / "checkpoints"
    / "026020"
    / "pretrained_model"
)

np = None
torch = None
imageio = None
ACTPolicy = None
make_pre_post_processors = None
SO101PickPlaceEnv = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="LeRobot ACT pretrained_model path.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps-per-episode", type=int, default=500, help="Maximum MuJoCo env steps per episode.")
    parser.add_argument("--steps-per-action", type=int, default=1, help="Repeat each selected ACT action this many env steps.")
    parser.add_argument("--device", default="cuda", help="Torch device for ACT inference.")
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer for live visual inspection.")
    parser.add_argument("--headless", action="store_true", help="Force EGL headless MuJoCo rendering before imports.")
    parser.add_argument("--randomize-block-reset", action="store_true", help="Randomize the block pose at episode reset.")
    parser.add_argument(
        "--curriculum-fixed-block",
        action="store_true",
        help="Use the fixed default block pose from the simulation curriculum.",
    )
    parser.add_argument("--block-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Fixed block position.")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed. Episode index is added when set.")
    parser.add_argument("--save-video", type=Path, default=None, help="Directory for rollout MP4 files.")
    parser.add_argument(
        "--video-cameras",
        default="all",
        choices=("top", "wrist", "side", "all"),
        help="Camera stream to save when --save-video is set.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Saved rollout video FPS.")
    parser.add_argument("--verbose", action="store_true", help="Print extra per-step debug information.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 0:
        raise ValueError("--episodes must be >= 0")
    if args.max_steps_per_episode < 1:
        raise ValueError("--max-steps-per-episode must be >= 1")
    if args.steps_per_action < 1:
        raise ValueError("--steps-per-action must be >= 1")
    if args.randomize_block_reset and args.block_pos is not None:
        raise ValueError("--randomize-block-reset and --block-pos are mutually exclusive")


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
        raise FileNotFoundError(
            f"ACT checkpoint is missing required files at {checkpoint}: {', '.join(missing)}"
        )

    empty = [name for name in required if (checkpoint / name).stat().st_size == 0]
    if empty:
        raise RuntimeError(f"ACT checkpoint has empty files at {checkpoint}: {', '.join(empty)}")


def load_dependencies(headless: bool) -> None:
    """Import heavy dependencies after CLI parsing and before MuJoCo env creation."""
    global np, torch, imageio, ACTPolicy, make_pre_post_processors, SO101PickPlaceEnv

    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        import numpy as _np
        import torch as _torch
    except ImportError as exc:
        raise RuntimeError(
            "ACT simulation inference requires the LeRobot/simulation Python environment. "
            "Activate the `lerobot` conda environment first."
        ) from exc

    try:
        import imageio.v2 as _imageio
    except ImportError:
        _imageio = None

    try:
        from lerobot.policies.act.modeling_act import ACTPolicy as _ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors as _make_pre_post_processors
    except ImportError as exc:
        raise RuntimeError(
            "Could not import LeRobot ACTPolicy. Activate the environment with the local LeRobot install."
        ) from exc

    try:
        from so101_gym_env import SO101PickPlaceEnv as _SO101PickPlaceEnv
    except ImportError as exc:
        raise RuntimeError("Could not import SO101PickPlaceEnv from the simulation_code directory.") from exc

    np = _np
    torch = _torch
    imageio = _imageio
    ACTPolicy = _ACTPolicy
    make_pre_post_processors = _make_pre_post_processors
    SO101PickPlaceEnv = _SO101PickPlaceEnv


def resolve_device(device_name: str) -> "torch.device":
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return device


def load_policy_and_processors(checkpoint: Path, device: "torch.device") -> tuple[Any, Any, Any]:
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


def adapt_sim_observation(obs: dict[str, Any]) -> dict[str, "torch.Tensor"]:
    if "observation.images.camera2" not in obs:
        raise KeyError("Simulation observation is missing observation.images.camera2 for the wrist camera.")
    if "observation.state" not in obs:
        raise KeyError("Simulation observation is missing observation.state.")

    wrist = obs["observation.images.camera2"]
    wrist_tensor = torch.as_tensor(wrist, dtype=torch.float32)
    if wrist_tensor.max() > 1.0:
        wrist_tensor = wrist_tensor / 255.0
    if wrist_tensor.ndim != 3 or wrist_tensor.shape[-1] != 3:
        raise ValueError(f"Expected wrist image shape HWC RGB, got {tuple(wrist_tensor.shape)}")

    state_tensor = torch.as_tensor(obs["observation.state"], dtype=torch.float32)
    if state_tensor.shape[-1] != 6:
        raise ValueError(f"Expected 6D joint state, got shape {tuple(state_tensor.shape)}")

    return {
        "observation.images.wrist": wrist_tensor.permute(2, 0, 1).contiguous(),
        "observation.state": state_tensor,
    }


def make_env(args: argparse.Namespace) -> Any:
    randomize_block = bool(args.randomize_block_reset)
    if args.curriculum_fixed_block or args.block_pos is not None:
        randomize_block = False

    return SO101PickPlaceEnv(
        render_mode="human" if args.render else None,
        image_size=256,
        max_episode_steps=args.max_steps_per_episode,
        randomize_block=randomize_block,
        vla_normalize=False,
        model_type="act",
    )


def reset_options(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.block_pos is not None:
        return {"block_pos": tuple(float(value) for value in args.block_pos)}
    return None


def select_action(policy: Any, preprocessor: Any, postprocessor: Any, obs: dict[str, Any], env: Any) -> Any:
    policy_obs = preprocessor(adapt_sim_observation(obs))
    with torch.no_grad():
        action = policy.select_action(policy_obs)
        action = postprocessor(action)
    action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.clip(action_np, env.joint_limits_low, env.joint_limits_high)


def video_frame(obs: dict[str, Any], cameras: str) -> Any:
    camera_keys = {
        "top": "observation.images.camera1",
        "wrist": "observation.images.camera2",
        "side": "observation.images.camera3",
    }
    if cameras == "all":
        frames = [obs[camera_keys[name]] for name in ("top", "wrist", "side")]
        return np.concatenate(frames, axis=1)
    return obs[camera_keys[cameras]]


def save_video(frames: list[Any], output_path: Path, fps: int) -> None:
    if not frames:
        return
    if imageio is None:
        raise RuntimeError("imageio is not installed; install it or omit --save-video.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def as_float(info: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = info.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_episode(
    env: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    args: argparse.Namespace,
    episode_idx: int,
) -> dict[str, Any]:
    seed = None if args.seed is None else args.seed + episode_idx
    obs, info = env.reset(seed=seed, options=reset_options(args))
    policy.reset()

    total_return = 0.0
    steps = 0
    success = False
    final_distance = as_float(info, "distance_to_block")
    final_block_height = as_float(info, "block_height")
    max_block_height = final_block_height
    contact_count = 0
    grasp_count = 0
    lift_count = 0
    frames = []

    if args.save_video is not None:
        frames.append(video_frame(obs, args.video_cameras))

    while steps < args.max_steps_per_episode:
        action = select_action(policy, preprocessor, postprocessor, obs, env)
        if args.verbose:
            print(f"episode={episode_idx + 1} step={steps} action={np.round(action, 4).tolist()}")

        for _ in range(args.steps_per_action):
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_return += float(reward)
            success = success or bool(info.get("success", False))
            final_distance = as_float(info, "distance_to_block", final_distance)
            final_block_height = as_float(info, "block_height", final_block_height)
            max_block_height = max(max_block_height, final_block_height)
            contact_count += int(bool(info.get("contacted", info.get("contact", False))))
            grasp_count += int(bool(info.get("gripped", info.get("grasp", False))))
            lift_count += int(bool(info.get("block_lifted", info.get("lifted", False))))

            if args.render:
                env.render()
            if args.save_video is not None:
                frames.append(video_frame(obs, args.video_cameras))

            if terminated or truncated or steps >= args.max_steps_per_episode:
                break

        if terminated or truncated or steps >= args.max_steps_per_episode:
            break

    if args.save_video is not None:
        save_video(frames, args.save_video / f"act_episode_{episode_idx + 1:03d}.mp4", args.video_fps)

    return {
        "return": total_return,
        "success": success,
        "steps": steps,
        "final_distance": final_distance,
        "final_block_height": final_block_height,
        "max_block_height": max_block_height,
        "contact_count": contact_count,
        "grasp_count": grasp_count,
        "lift_count": lift_count,
    }


def mean_metric(metrics: list[dict[str, Any]], key: str) -> float:
    if not metrics:
        return float("nan")
    return float(np.mean([float(item[key]) for item in metrics]))


def print_summary(metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        print("No episodes requested; checkpoint and dependencies loaded successfully.")
        return

    success_rate = float(np.mean([float(item["success"]) for item in metrics]))
    print("\nSummary")
    print(f"  episodes: {len(metrics)}")
    print(f"  success_rate: {success_rate:.3f}")
    print(f"  mean_return: {mean_metric(metrics, 'return'):.3f}")
    print(f"  mean_steps: {mean_metric(metrics, 'steps'):.1f}")
    print(f"  mean_final_distance: {mean_metric(metrics, 'final_distance'):.4f}")
    print(f"  mean_max_block_height: {mean_metric(metrics, 'max_block_height'):.4f}")
    print(f"  mean_contact_count: {mean_metric(metrics, 'contact_count'):.1f}")
    print(f"  mean_grasp_count: {mean_metric(metrics, 'grasp_count'):.1f}")
    print(f"  mean_lift_count: {mean_metric(metrics, 'lift_count'):.1f}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    validate_checkpoint(checkpoint)

    headless = args.headless or not args.render
    load_dependencies(headless=headless)
    device = resolve_device(args.device)

    print(f"Loading ACT checkpoint: {checkpoint}")
    print(f"Device: {device}")
    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint, device)
    print(
        "ACT config: "
        f"chunk_size={getattr(policy.config, 'chunk_size', None)} "
        f"n_action_steps={getattr(policy.config, 'n_action_steps', None)} "
        "with checkpoint pre/postprocessors"
    )
    env = make_env(args)

    metrics: list[dict[str, Any]] = []
    try:
        for episode_idx in range(args.episodes):
            episode_metrics = run_episode(env, policy, preprocessor, postprocessor, args, episode_idx)
            metrics.append(episode_metrics)
            print(
                f"episode {episode_idx + 1:03d}: "
                f"return={episode_metrics['return']:.3f} "
                f"success={int(episode_metrics['success'])} "
                f"steps={episode_metrics['steps']} "
                f"final_distance={episode_metrics['final_distance']:.4f} "
                f"max_block_height={episode_metrics['max_block_height']:.4f}"
            )
    finally:
        env.close()

    print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
