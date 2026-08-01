#!/usr/bin/env python3
"""
Run deterministic ACT PPO inference in the SO-101 MuJoCo pick-place simulation.

This script loads PPO training checkpoints produced by `train_act_in_sim.py`.
For supervised LeRobot ACT `pretrained_model` directories, use
`run_act_sim_inference.py` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path, PosixPath
from typing import Any

import torch
import torch.serialization

import train_act_in_sim as sim


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INIT_CHECKPOINT = sim.DEFAULT_INIT_CHECKPOINT
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
    parser.add_argument(
        "--reward-profile",
        choices=tuple(sim.ACT_REWARD_PROFILES),
        default="baseline",
        help="Reward profile used for evaluation metrics and termination semantics.",
    )
    parser.add_argument("--policy-lr", type=float, default=1e-5, help="Optimizer LR needed only to restore checkpoint state.")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Optimizer LR needed only to restore checkpoint state.")
    parser.add_argument("--device", default="cuda", help="Torch device for ACT PPO inference.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions from the PPO policy distribution instead of using the mean action.")
    parser.add_argument("--stochastic-scale", type=float, default=1.0, help="Multiplier for learned PPO action std when --stochastic is set.")
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer for live visual inspection.")
    parser.add_argument("--headless", action="store_true", help="Force EGL headless MuJoCo rendering before imports.")
    parser.add_argument("--randomize-block-reset", action="store_true", help="Randomize the block pose at episode reset.")
    parser.add_argument(
        "--randomize-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vary scene lighting and surface brightness on reset (enabled by default).",
    )
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
    parser.add_argument("--output-json", type=Path, default=None, help="Write per-episode and aggregate metrics as JSON.")
    parser.add_argument("--reset-bank", type=Path, default=None)
    parser.add_argument(
        "--evaluation-reset-stage", choices=("normal", "pregrasp", "grasped"), default="normal"
    )
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
    if args.stochastic_scale < 0:
        raise ValueError("--stochastic-scale must be >= 0")
    if args.randomize_block_reset and args.block_pos is not None:
        raise ValueError("--randomize-block-reset and --block-pos are mutually exclusive")
    if args.block_dist_range[0] > args.block_dist_range[1]:
        raise ValueError("--block-dist-range MIN must be <= MAX")
    if args.block_angle_range[0] > args.block_angle_range[1]:
        raise ValueError("--block-angle-range MIN must be <= MAX")
    if args.evaluation_reset_stage != "normal" and args.reset_bank is None:
        raise ValueError("--reset-bank is required for pregrasp/grasped evaluation")


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
    normalization_stats = sim.load_act_normalization_stats(args.init_checkpoint, device)
    policy = sim.ACTGaussianPPOPolicy(
        act_policy,
        action_dim=6,
        chunk_size=args.chunk_size,
        log_std_init=args.log_std_init,
        normalization_stats=normalization_stats,
    ).to(device)
    critic = sim.PrivilegedCritic(input_dim=16).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    torch.serialization.add_safe_globals([PosixPath])
    checkpoint = sim.load_branch_checkpoint(args.resume, policy, critic, device)
    start_episode = int(checkpoint.get("episode", 0)) + 1
    total_chunks = int(checkpoint.get("total_chunks", 0))
    total_env_steps = int(checkpoint.get("total_env_steps", 0))
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
    options = None
    if args.evaluation_reset_stage != "normal":
        bank = sim.load_reset_bank(args.reset_bank)
        states = bank[args.evaluation_reset_stage]
        options = {"reset_state": states[episode_idx % len(states)]}
    obs, info = env.reset(seed=seed, options=options)

    total_return = 0.0
    steps = 0
    success = False
    final_distance = as_float(info, "distance_to_block")
    final_block_height = as_float(info, "block_height")
    final_block_height_gain = as_float(info, "block_height_gain", 0.0)
    max_block_height = final_block_height
    max_block_height_gain = final_block_height_gain
    contact_count = 0
    grasp_count = 0
    force_grasp_count = 0
    face_grasp_count = 0
    force_only_grasp_count = 0
    interior_face_contact_count = 0
    corner_only_contact_count = 0
    fixed_interior_face_contact_count = 0
    moving_interior_face_contact_count = 0
    bilateral_interior_face_contact_count = 0
    face_corner_rejection_count = 0
    face_alignment_total = 0.0
    face_opposition_total = 0.0
    face_jaw_axis_alignment_total = 0.0
    face_diagnostic_steps = 0
    pregrasp_face_axis_alignment_total = 0.0
    pregrasp_face_guidance_score_total = 0.0
    face_axis_guidance_progress_total = 0.0
    pregrasp_face_height_error_total = 0.0
    pregrasp_face_depth_error_total = 0.0
    pregrasp_face_lateral_error_total = 0.0
    interior_face_contact_score_total = 0.0
    face_contact_quality_total = 0.0
    bilateral_opposition_quality_total = 0.0
    micro_lift_count = 0
    lift_count = 0
    consecutive_grasp_steps = 0
    max_consecutive_grasp_steps = 0
    max_block_displacement = 0.0
    max_strict_grasp_streak = 0
    strict_lift = False
    strict_success = False
    action_clip_count = 0
    action_value_count = 0
    grasped_vertical_lift_reward_total = 0.0
    lift_side_push_penalty_total = 0.0
    frames = []
    step_delay = args.step_delay if args.step_delay is not None else (0.02 if args.render else 0.0)

    if args.save_video is not None:
        frames.append(video_frame(obs, args.video_cameras))

    with torch.no_grad():
        while steps < args.max_steps_per_episode:
            policy_obs = sim.adapt_sim_observation(obs, device)
            action_chunk = policy.mean_chunk(policy_obs, require_grad=False)
            if args.stochastic:
                std = policy.log_std.exp().view(1, 1, -1).expand_as(action_chunk)
                action_chunk = action_chunk + torch.randn_like(action_chunk) * std * float(args.stochastic_scale)
            action_chunk_np = action_chunk.squeeze(0).detach().cpu().numpy()
            if args.verbose:
                mode = "stochastic" if args.stochastic else "deterministic"
                print(f"episode={episode_idx + 1} step={steps} mode={mode} chunk_shape={action_chunk_np.shape}")

            done = False
            for action in action_chunk_np:
                clipped_action, clip_mask = sim.clip_mujoco_qpos(action)
                action_clip_count += int(clip_mask.sum())
                action_value_count += int(clip_mask.size)
                if args.verbose and clip_mask.any():
                    print(
                        f"  unclipped_mujoco_target={np.round(action, 4).tolist()} "
                        f"clip_mask={clip_mask.astype(int).tolist()}"
                    )
                obs, reward, terminated, truncated, info = sim.step_action(env, clipped_action, args.steps_per_action)
                steps += args.steps_per_action
                total_return += float(reward)
                success = success or bool(info.get("success", False))
                final_distance = as_float(info, "distance_to_block", final_distance)
                final_block_height = as_float(info, "block_height", final_block_height)
                final_block_height_gain = as_float(info, "block_height_gain", final_block_height_gain)
                max_block_height = max(max_block_height, final_block_height)
                max_block_height_gain = max(max_block_height_gain, final_block_height_gain)
                contact_count += int(bool(info.get("contacted", info.get("contact", False))))
                grasp_count += int(bool(info.get("gripped", info.get("grasp", False))))
                force_grasp_count += int(bool(info.get("force_gripped", False)))
                face_grasp_count += int(bool(info.get("face_gripped", False)))
                force_only_grasp_count += int(
                    bool(info.get("force_only_grasp", False))
                )
                interior_face_contact_count += int(
                    bool(info.get("interior_face_contact", False))
                )
                corner_only_contact_count += int(
                    bool(info.get("corner_only_contact", False))
                )
                fixed_interior_face_contact_count += int(
                    bool(info.get("fixed_interior_face_contact", False))
                )
                moving_interior_face_contact_count += int(
                    bool(info.get("moving_interior_face_contact", False))
                )
                bilateral_interior_face_contact_count += int(
                    bool(info.get("bilateral_interior_face_contact", False))
                )
                face_corner_rejection_count += int(
                    bool(info.get("face_corner_rejection", False))
                )
                face_alignment_total += as_float(info, "face_alignment", 0.0)
                face_opposition_total += as_float(info, "face_opposition", 0.0)
                face_jaw_axis_alignment_total += as_float(
                    info, "face_jaw_axis_alignment", 0.0
                )
                face_diagnostic_steps += 1
                pregrasp_face_axis_alignment_total += as_float(
                    info, "pregrasp_face_axis_alignment", 0.0
                )
                pregrasp_face_guidance_score_total += as_float(
                    info, "pregrasp_face_guidance_score", 0.0
                )
                face_axis_guidance_progress_total += as_float(
                    info, "face_axis_guidance_progress", 0.0
                )
                pregrasp_face_height_error_total += as_float(
                    info, "pregrasp_face_height_error", 0.0
                )
                pregrasp_face_depth_error_total += as_float(
                    info, "pregrasp_face_depth_error", 0.0
                )
                pregrasp_face_lateral_error_total += as_float(
                    info, "pregrasp_face_lateral_error", 0.0
                )
                interior_face_contact_score_total += as_float(
                    info, "interior_face_contact_score", 0.0
                )
                face_contact_quality_total += as_float(
                    info, "face_contact_quality", 0.0
                )
                bilateral_opposition_quality_total += as_float(
                    info, "bilateral_opposition_quality", 0.0
                )
                if bool(info.get("gripped", info.get("grasp", False))):
                    consecutive_grasp_steps += 1
                    max_consecutive_grasp_steps = max(max_consecutive_grasp_steps, consecutive_grasp_steps)
                else:
                    consecutive_grasp_steps = 0
                micro_lift_count += int(bool(info.get("micro_lifted", False)))
                lift_count += int(bool(info.get("block_lifted", info.get("lifted", False))))
                grasped_vertical_lift_reward_total += as_float(info, "grasped_vertical_lift_reward", 0.0)
                lift_side_push_penalty_total += as_float(info, "lift_side_push_penalty", 0.0)
                max_block_displacement = max(max_block_displacement, as_float(info, "block_displacement", 0.0))
                max_strict_grasp_streak = max(
                    max_strict_grasp_streak, int(info.get("strict_grasp_streak", 0))
                )
                strict_success = strict_success or bool(info.get("strict_lift_success", False))
                strict_lift = strict_lift or bool(
                    info.get("independent_strict_lift_crossed", False)
                )

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
        "final_block_height_gain": final_block_height_gain,
        "max_block_height": max_block_height,
        "max_block_height_gain": max_block_height_gain,
        "contact_count": contact_count,
        "grasp_count": grasp_count,
        "force_grasp_count": force_grasp_count,
        "face_grasp_count": face_grasp_count,
        "force_only_grasp_count": force_only_grasp_count,
        "interior_face_contact_count": interior_face_contact_count,
        "corner_only_contact_count": corner_only_contact_count,
        "fixed_interior_face_contact_count": fixed_interior_face_contact_count,
        "moving_interior_face_contact_count": moving_interior_face_contact_count,
        "bilateral_interior_face_contact_count": bilateral_interior_face_contact_count,
        "face_corner_rejection_count": face_corner_rejection_count,
        "mean_face_alignment": face_alignment_total / max(1, face_diagnostic_steps),
        "mean_face_opposition": face_opposition_total / max(1, face_diagnostic_steps),
        "mean_face_jaw_axis_alignment": (
            face_jaw_axis_alignment_total / max(1, face_diagnostic_steps)
        ),
        "mean_pregrasp_face_axis_alignment": (
            pregrasp_face_axis_alignment_total / max(1, face_diagnostic_steps)
        ),
        "mean_pregrasp_face_guidance_score": (
            pregrasp_face_guidance_score_total / max(1, face_diagnostic_steps)
        ),
        "mean_face_axis_guidance_progress": (
            face_axis_guidance_progress_total / max(1, face_diagnostic_steps)
        ),
        "mean_pregrasp_face_height_error": (
            pregrasp_face_height_error_total / max(1, face_diagnostic_steps)
        ),
        "mean_pregrasp_face_depth_error": (
            pregrasp_face_depth_error_total / max(1, face_diagnostic_steps)
        ),
        "mean_pregrasp_face_lateral_error": (
            pregrasp_face_lateral_error_total / max(1, face_diagnostic_steps)
        ),
        "mean_interior_face_contact_score": (
            interior_face_contact_score_total / max(1, face_diagnostic_steps)
        ),
        "mean_face_contact_quality": (
            face_contact_quality_total / max(1, face_diagnostic_steps)
        ),
        "mean_bilateral_opposition_quality": (
            bilateral_opposition_quality_total / max(1, face_diagnostic_steps)
        ),
        "micro_lift_count": micro_lift_count,
        "lift_count": lift_count,
        "sustained_grasp": max_strict_grasp_streak >= 5,
        "max_consecutive_grasp_steps": max_consecutive_grasp_steps,
        "max_strict_grasp_streak": max_strict_grasp_streak,
        "strict_lift": strict_lift,
        "strict_success": strict_success,
        "max_block_displacement": max_block_displacement,
        "action_clip_rate": action_clip_count / max(1, action_value_count),
        "grasped_vertical_lift_reward": grasped_vertical_lift_reward_total,
        "lift_side_push_penalty": lift_side_push_penalty_total,
    }


def mean_metric(metrics: list[dict[str, Any]], key: str) -> float:
    if not metrics:
        return float("nan")
    return float(np.mean([float(item[key]) for item in metrics]))


def build_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"episodes": 0}
    return {
        "episodes": len(metrics),
        "success_rate": float(np.mean([float(item["success"]) for item in metrics])),
        "strict_success_rate": float(np.mean([float(item["strict_success"]) for item in metrics])),
        "sustained_grasp_rate": float(np.mean([float(item["sustained_grasp"]) for item in metrics])),
        "strict_lift_rate": float(np.mean([float(item["strict_lift"]) for item in metrics])),
        "mean_return": mean_metric(metrics, "return"),
        "mean_steps": mean_metric(metrics, "steps"),
        "mean_final_distance": mean_metric(metrics, "final_distance"),
        "mean_max_block_height": mean_metric(metrics, "max_block_height"),
        "mean_max_block_height_gain": mean_metric(metrics, "max_block_height_gain"),
        "mean_contact_count": mean_metric(metrics, "contact_count"),
        "mean_grasp_count": mean_metric(metrics, "grasp_count"),
        "mean_force_grasp_count": mean_metric(metrics, "force_grasp_count"),
        "mean_face_grasp_count": mean_metric(metrics, "face_grasp_count"),
        "mean_force_only_grasp_count": mean_metric(
            metrics, "force_only_grasp_count"
        ),
        "mean_interior_face_contact_count": mean_metric(
            metrics, "interior_face_contact_count"
        ),
        "mean_corner_only_contact_count": mean_metric(
            metrics, "corner_only_contact_count"
        ),
        "mean_fixed_interior_face_contact_count": mean_metric(
            metrics, "fixed_interior_face_contact_count"
        ),
        "mean_moving_interior_face_contact_count": mean_metric(
            metrics, "moving_interior_face_contact_count"
        ),
        "mean_bilateral_interior_face_contact_count": mean_metric(
            metrics, "bilateral_interior_face_contact_count"
        ),
        "mean_face_corner_rejection_count": mean_metric(
            metrics, "face_corner_rejection_count"
        ),
        "mean_face_alignment": mean_metric(metrics, "mean_face_alignment"),
        "mean_face_opposition": mean_metric(metrics, "mean_face_opposition"),
        "mean_face_jaw_axis_alignment": mean_metric(
            metrics, "mean_face_jaw_axis_alignment"
        ),
        "mean_pregrasp_face_axis_alignment": mean_metric(
            metrics, "mean_pregrasp_face_axis_alignment"
        ),
        "mean_pregrasp_face_guidance_score": mean_metric(
            metrics, "mean_pregrasp_face_guidance_score"
        ),
        "mean_face_axis_guidance_progress": mean_metric(
            metrics, "mean_face_axis_guidance_progress"
        ),
        "mean_pregrasp_face_height_error": mean_metric(
            metrics, "mean_pregrasp_face_height_error"
        ),
        "mean_pregrasp_face_depth_error": mean_metric(
            metrics, "mean_pregrasp_face_depth_error"
        ),
        "mean_pregrasp_face_lateral_error": mean_metric(
            metrics, "mean_pregrasp_face_lateral_error"
        ),
        "mean_interior_face_contact_score": mean_metric(
            metrics, "mean_interior_face_contact_score"
        ),
        "mean_face_contact_quality": mean_metric(
            metrics, "mean_face_contact_quality"
        ),
        "mean_bilateral_opposition_quality": mean_metric(
            metrics, "mean_bilateral_opposition_quality"
        ),
        "mean_max_consecutive_grasp_steps": mean_metric(metrics, "max_consecutive_grasp_steps"),
        "mean_micro_lift_count": mean_metric(metrics, "micro_lift_count"),
        "mean_lift_count": mean_metric(metrics, "lift_count"),
        "mean_max_block_displacement": mean_metric(metrics, "max_block_displacement"),
        "mean_action_clip_rate": mean_metric(metrics, "action_clip_rate"),
        "mean_grasped_vertical_lift_reward": mean_metric(metrics, "grasped_vertical_lift_reward"),
        "mean_lift_side_push_penalty": mean_metric(metrics, "lift_side_push_penalty"),
    }


def print_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        print("No episodes requested; checkpoint and dependencies loaded successfully.")
        return build_summary(metrics)

    summary = build_summary(metrics)
    print("\nSummary")
    print(f"  episodes: {summary['episodes']}")
    print(f"  success_rate: {summary['success_rate']:.3f}")
    print(f"  sustained_grasp_rate: {summary['sustained_grasp_rate']:.3f}")
    print(f"  strict_lift_rate: {summary['strict_lift_rate']:.3f}")
    print(f"  mean_return: {mean_metric(metrics, 'return'):.3f}")
    print(f"  mean_steps: {mean_metric(metrics, 'steps'):.1f}")
    print(f"  mean_final_distance: {mean_metric(metrics, 'final_distance'):.4f}")
    print(f"  mean_max_block_height: {mean_metric(metrics, 'max_block_height'):.4f}")
    print(f"  mean_max_block_height_gain: {mean_metric(metrics, 'max_block_height_gain'):.4f}")
    print(f"  mean_contact_count: {mean_metric(metrics, 'contact_count'):.1f}")
    print(f"  mean_grasp_count: {mean_metric(metrics, 'grasp_count'):.1f}")
    print(f"  mean_force_grasp_count: {summary['mean_force_grasp_count']:.1f}")
    print(f"  mean_face_grasp_count: {summary['mean_face_grasp_count']:.1f}")
    print(
        f"  mean_force_only_grasp_count: "
        f"{summary['mean_force_only_grasp_count']:.1f}"
    )
    print(
        "  mean_interior_face_contact_count: "
        f"{summary['mean_interior_face_contact_count']:.1f}"
    )
    print(
        "  mean_corner_only_contact_count: "
        f"{summary['mean_corner_only_contact_count']:.1f}"
    )
    print(
        "  mean_fixed_interior_face_contact_count: "
        f"{summary['mean_fixed_interior_face_contact_count']:.1f}"
    )
    print(
        "  mean_moving_interior_face_contact_count: "
        f"{summary['mean_moving_interior_face_contact_count']:.1f}"
    )
    print(
        "  mean_bilateral_interior_face_contact_count: "
        f"{summary['mean_bilateral_interior_face_contact_count']:.1f}"
    )
    print(
        "  mean_face_corner_rejection_count: "
        f"{summary['mean_face_corner_rejection_count']:.1f}"
    )
    print(f"  mean_face_alignment: {summary['mean_face_alignment']:.3f}")
    print(f"  mean_face_opposition: {summary['mean_face_opposition']:.3f}")
    print(
        "  mean_face_jaw_axis_alignment: "
        f"{summary['mean_face_jaw_axis_alignment']:.3f}"
    )
    print(
        "  mean_pregrasp_face_axis_alignment: "
        f"{summary['mean_pregrasp_face_axis_alignment']:.3f}"
    )
    print(
        "  mean_pregrasp_face_guidance_score: "
        f"{summary['mean_pregrasp_face_guidance_score']:.3f}"
    )
    print(
        "  mean_face_axis_guidance_progress: "
        f"{summary['mean_face_axis_guidance_progress']:.4f}"
    )
    print(
        "  mean_interior_face_contact_score: "
        f"{summary['mean_interior_face_contact_score']:.3f}"
    )
    print(
        "  mean_face_contact_quality: "
        f"{summary['mean_face_contact_quality']:.3f}"
    )
    print(
        "  mean_bilateral_opposition_quality: "
        f"{summary['mean_bilateral_opposition_quality']:.3f}"
    )
    print(f"  mean_micro_lift_count: {mean_metric(metrics, 'micro_lift_count'):.1f}")
    print(f"  mean_lift_count: {mean_metric(metrics, 'lift_count'):.1f}")
    print(f"  mean_grasped_vertical_lift_reward: {mean_metric(metrics, 'grasped_vertical_lift_reward'):.3f}")
    print(f"  mean_lift_side_push_penalty: {mean_metric(metrics, 'lift_side_push_penalty'):.3f}")
    print(f"  mean_max_block_displacement: {mean_metric(metrics, 'max_block_displacement'):.4f}")
    print(f"  mean_action_clip_rate: {mean_metric(metrics, 'action_clip_rate'):.4f}")
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.resume = args.resume.expanduser().resolve()
    args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    if args.save_video is not None:
        args.save_video = args.save_video.expanduser().resolve()
    if args.output_json is not None:
        args.output_json = args.output_json.expanduser().resolve()
    args.reward_kwargs = sim.resolve_reward_profile(args.reward_profile)
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
    if args.seed is not None:
        torch.manual_seed(args.seed)

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
                f"max_block_height_gain={episode_metrics['max_block_height_gain']:.4f} "
                f"contact_count={episode_metrics['contact_count']} "
                f"grasp_count={episode_metrics['grasp_count']} "
                f"lift_count={episode_metrics['lift_count']}"
            )
    finally:
        env.close()

    summary = print_summary(metrics)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"summary": summary, "episodes": metrics}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote evaluation JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
