#!/usr/bin/env python3
"""
Run deterministic ACT PPO inference in the SO-101 MuJoCo pick-place simulation.

This script loads PPO training checkpoints produced by `train_act_in_sim.py`.
For supervised LeRobot ACT `pretrained_model` directories, use
`run_act_sim_inference.py` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path, PosixPath
from typing import Any

import torch
import torch.serialization

import train_act_in_sim as sim


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INIT_CHECKPOINT = (
    SCRIPT_DIR
    / "outputs"
    / "train"
    / "act_so101_corrected_30_b32_20260621_160923"
    / "checkpoints"
    / "026020"
    / "pretrained_model"
)
DEFAULT_RESUME = SCRIPT_DIR / "act_sim_ppo_checkpoint.pt"

imageio = None
np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME, help="ACT PPO .pt checkpoint path.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_INIT_CHECKPOINT,
        help="LeRobot ACT pretrained_model path used to initialize the PPO wrapper.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps-per-episode", type=int, default=300, help="Maximum MuJoCo env steps per episode.")
    parser.add_argument("--steps-per-action", type=int, default=1, help="Repeat each selected ACT action this many env steps.")
    parser.add_argument("--chunk-size", type=int, default=30, help="ACT action chunk size used by the PPO checkpoint.")
    parser.add_argument("--log-std-init", type=float, default=-2.0, help="Initial PPO log std; overwritten by checkpoint.")
    parser.add_argument("--policy-lr", type=float, default=1e-5, help="Optimizer LR needed only to restore checkpoint state.")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Optimizer LR needed only to restore checkpoint state.")
    parser.add_argument("--device", default="cuda", help="Torch device for ACT PPO inference.")
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer for live visual inspection.")
    parser.add_argument("--headless", action="store_true", help="Force EGL headless MuJoCo rendering before imports.")
    parser.add_argument("--randomize-block-reset", action="store_true", help="Randomize the block pose at episode reset.")
    parser.add_argument("--block-dist-range", type=float, nargs=2, default=(0.22, 0.26), metavar=("MIN", "MAX"), help="Randomized block distance range used with --randomize-block-reset.")
    parser.add_argument("--block-angle-range", type=float, nargs=2, default=(-10.0, 10.0), metavar=("MIN", "MAX"), help="Randomized block angle range in degrees used with --randomize-block-reset.")
    parser.add_argument(
        "--curriculum-fixed-block",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Delay after each sim step. Defaults to 0.02 with --render and 0.0 otherwise.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra per-step debug information.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 0:
        raise ValueError("--episodes must be >= 0")
    if args.max_steps_per_episode < 1:
        raise ValueError("--max-steps-per-episode must be >= 1")
    if args.steps_per_action < 1:
        raise ValueError("--steps-per-action must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.step_delay is not None and args.step_delay < 0:
        raise ValueError("--step-delay must be >= 0")
    if args.randomize_block_reset and args.block_pos is not None:
        raise ValueError("--randomize-block-reset and --block-pos are mutually exclusive")
    if args.block_dist_range[0] > args.block_dist_range[1]:
        raise ValueError("--block-dist-range MIN must be <= MAX")
    if args.block_angle_range[0] > args.block_angle_range[1]:
        raise ValueError("--block-angle-range MIN must be <= MAX")


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return device


def import_video_dependencies() -> None:
    global imageio, np
    try:
        import imageio.v2 as _imageio
    except ImportError:
        _imageio = None
    import numpy as _np

    imageio = _imageio
    np = _np


def block_position(args: argparse.Namespace) -> tuple[float, float, float]:
    if args.block_pos is not None:
        return tuple(float(value) for value in args.block_pos)
    return sim.block_position(args)


def make_env(args: argparse.Namespace) -> Any:
    return sim.make_sequential_env(args, block_position(args))


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


def build_policy(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    act_policy = sim.load_act_policy(args.init_checkpoint, device)
    policy = sim.ACTGaussianPPOPolicy(
        act_policy,
        action_dim=6,
        chunk_size=args.chunk_size,
        log_std_init=args.log_std_init,
    ).to(device)
    critic = sim.PrivilegedCritic(input_dim=16).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    torch.serialization.add_safe_globals([PosixPath])
    start_episode, total_chunks, total_env_steps = sim.load_checkpoint(
        args.resume,
        policy,
        critic,
        policy_optimizer,
        critic_optimizer,
        device,
    )
    print(
        "Loaded ACT PPO checkpoint: "
        f"{args.resume} "
        f"start_episode={start_episode} "
        f"total_chunks={total_chunks} "
        f"total_env_steps={total_env_steps}"
    )
    return policy, critic


def run_episode(env: Any, policy: Any, device: torch.device, args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    seed = None if args.seed is None else args.seed + episode_idx
    obs, info = env.reset(seed=seed)

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
    step_delay = args.step_delay if args.step_delay is not None else (0.02 if args.render else 0.0)

    if args.save_video is not None:
        frames.append(video_frame(obs, args.video_cameras))

    with torch.no_grad():
        while steps < args.max_steps_per_episode:
            policy_obs = sim.adapt_sim_observation(obs, device)
            mean_chunk = policy.mean_chunk(policy_obs, require_grad=False).squeeze(0).detach().cpu().numpy()
            if args.verbose:
                print(f"episode={episode_idx + 1} step={steps} chunk_shape={mean_chunk.shape}")

            done = False
            for action in mean_chunk:
                clipped_action = np.clip(action, env.joint_limits_low, env.joint_limits_high)
                obs, reward, terminated, truncated, info = sim.step_action(env, clipped_action, args.steps_per_action)
                steps += args.steps_per_action
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
                if step_delay > 0:
                    time.sleep(step_delay)

                done = terminated or truncated or steps >= args.max_steps_per_episode
                if done:
                    break

            if done or steps >= args.max_steps_per_episode:
                break

    if args.save_video is not None:
        save_video(frames, args.save_video / f"act_ppo_episode_{episode_idx + 1:03d}.mp4", args.video_fps)

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
    args.resume = args.resume.expanduser().resolve()
    args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    if args.save_video is not None:
        args.save_video = args.save_video.expanduser().resolve()
    if not args.resume.is_file():
        raise FileNotFoundError(f"ACT PPO checkpoint not found: {args.resume}")
    if not args.init_checkpoint.exists():
        raise FileNotFoundError(f"ACT init checkpoint not found: {args.init_checkpoint}")

    headless = args.headless or not args.render
    args.no_render = not args.render
    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    sim.load_training_dependencies()
    sim.define_model_classes()
    import_video_dependencies()
    device = resolve_device(args.device)

    print(f"Loading ACT init checkpoint: {args.init_checkpoint}")
    print(f"Device: {device}")
    policy, _critic = build_policy(args, device)
    policy.eval()

    env = make_env(args)
    metrics: list[dict[str, Any]] = []
    try:
        for episode_idx in range(args.episodes):
            episode_metrics = run_episode(env, policy, device, args, episode_idx)
            metrics.append(episode_metrics)
            print(
                f"episode {episode_idx + 1:03d}: "
                f"return={episode_metrics['return']:.3f} "
                f"success={int(episode_metrics['success'])} "
                f"steps={episode_metrics['steps']} "
                f"final_distance={episode_metrics['final_distance']:.4f} "
                f"max_block_height={episode_metrics['max_block_height']:.4f} "
                f"contact_count={episode_metrics['contact_count']} "
                f"grasp_count={episode_metrics['grasp_count']} "
                f"lift_count={episode_metrics['lift_count']}"
            )
    finally:
        env.close()

    print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
