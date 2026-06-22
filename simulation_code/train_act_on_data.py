#!/usr/bin/env python3
"""
Train an ACT policy from the physical SO-101 demonstration dataset.

This script is intentionally a thin wrapper around the official LeRobot
training entry point. It validates the local dataset, then launches
`lerobot-train` with the dataset path this LeRobot fork expects.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = PROJECT_DIR / "imitation-learning" / "datasets" / "so101_pickplace_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "train" / "act_so101_physical"
DEFAULT_REPO_ID = "local/so101_pickplace_v1"
DEFAULT_FAST_TARGET_EPOCHS = 20.0
DEFAULT_SAVE_EVERY_EPOCHS = 5.0
DEFAULT_CORRECTED_BATCH_SIZE = 32
DEFAULT_CORRECTED_CHUNK_SIZE = 30
DEFAULT_CORRECTED_N_ACTION_STEPS = 30


class DatasetValidationError(RuntimeError):
    """Raised when the local LeRobot dataset is not usable for ACT training."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT on the physical SO-101 dataset via LeRobot")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default="act_so101_physical")
    parser.add_argument(
        "--performance-profile",
        "--profile",
        dest="performance_profile",
        choices=("baseline", "fast", "corrected-act"),
        default="corrected-act",
        help=(
            "baseline preserves LeRobot defaults; fast enables AMP and epoch-based step defaults; "
            "corrected-act adds the dataset/action preflight and shorter ACT horizons."
        ),
    )
    parser.add_argument("--steps", type=int, default=None, help="Optional LeRobot training step override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional LeRobot batch-size override")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="LeRobot DataLoader worker count. Default 0 avoids multiprocessing issues on WSL-mounted workspaces.",
    )
    parser.add_argument(
        "--target-epochs",
        type=float,
        default=None,
        help="Compute steps from dataset frames and batch size when --steps is omitted.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=float,
        default=None,
        help="Compute LeRobot save_freq from dataset frames and batch size.",
    )
    parser.add_argument("--log-freq", type=int, default=None, help="Optional LeRobot log_freq override")
    parser.add_argument("--eval-freq", type=int, default=None, help="Optional LeRobot eval_freq override")
    parser.add_argument("--video-backend", default=None, help="Optional LeRobot dataset.video_backend override")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="ACT policy.chunk_size override. corrected-act defaults to 30.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="ACT policy.n_action_steps override. corrected-act defaults to 30.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip parquet-level action/state audit. Not recommended for real training.",
    )
    parser.add_argument(
        "--use-file-system-sharing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch LeRobot through a wrapper that sets PyTorch multiprocessing sharing_strategy=file_system.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--resume", action="store_true", help="Resume from output checkpoint using LeRobot config")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved command")
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--use-amp", dest="use_amp", action="store_true", default=None)
    amp.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    wandb = parser.add_mutually_exclusive_group()
    wandb.add_argument("--wandb", dest="wandb_enabled", action="store_true", default=True)
    wandb.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON in {path}: {exc}") from exc


def validate_dataset(dataset_path: Path) -> dict:
    dataset_path = dataset_path.resolve()
    info_path = dataset_path / "meta" / "info.json"
    stats_path = dataset_path / "meta" / "stats.json"
    tasks_path = dataset_path / "meta" / "tasks.parquet"

    info = _load_json(info_path)
    _load_json(stats_path)
    if not tasks_path.exists():
        raise DatasetValidationError(f"Missing task metadata: {tasks_path}")

    expected = {
        "total_episodes": 100,
        "total_frames": 41631,
        "fps": 30,
        "robot_type": "so101_follower",
    }
    for key, value in expected.items():
        actual = info.get(key)
        if actual != value:
            raise DatasetValidationError(f"Expected {key}={value!r}, found {actual!r} in {info_path}")

    features = info.get("features", {})
    required_features = {
        "action": [6],
        "observation.state": [6],
        "observation.images.wrist": [256, 256, 3],
    }
    for key, shape in required_features.items():
        feature = features.get(key)
        if feature is None:
            raise DatasetValidationError(f"Dataset is missing feature {key!r}")
        if feature.get("shape") != shape:
            raise DatasetValidationError(f"Feature {key!r} has shape {feature.get('shape')!r}, expected {shape!r}")

    data_files = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    video_files = sorted((dataset_path / "videos" / "observation.images.wrist").glob("chunk-*/*.mp4"))
    episode_files = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not data_files:
        raise DatasetValidationError(f"No parquet data files found under {dataset_path / 'data'}")
    if not video_files:
        raise DatasetValidationError(
            f"No wrist-camera videos found under {dataset_path / 'videos' / 'observation.images.wrist'}"
        )
    if not episode_files:
        raise DatasetValidationError(f"No episode metadata parquet files found under {dataset_path / 'meta' / 'episodes'}")

    return {
        "path": str(dataset_path),
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "fps": info["fps"],
        "robot_type": info["robot_type"],
        "data_files": len(data_files),
        "wrist_videos": len(video_files),
        "episode_files": len(episode_files),
    }


def audit_action_contract(dataset_path: Path, horizons: tuple[int, ...] = (1, 30, 100)) -> dict:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DatasetValidationError(
            "Dataset action/state preflight requires pandas, pyarrow, and numpy. "
            "Activate the `lerobot` conda environment before training, or pass --skip-preflight "
            "only if you intentionally want to bypass this audit."
        ) from exc

    rows = []
    for path in sorted((dataset_path / "data").glob("chunk-*/*.parquet")):
        rows.append(pd.read_parquet(path, columns=["action", "observation.state", "episode_index", "frame_index"]))
    if not rows:
        raise DatasetValidationError(f"No parquet files found under {dataset_path / 'data'}")

    df = pd.concat(rows, ignore_index=True).sort_values(["episode_index", "frame_index"])
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    abs_action_state = np.abs(actions - states)

    episode_lengths = df.groupby("episode_index", sort=True).size().to_numpy()
    start_indices = df.groupby("episode_index", sort=True).head(1).index.to_numpy()
    end_indices = df.groupby("episode_index", sort=True).tail(1).index.to_numpy()

    future_delta_mean_abs = {}
    for horizon in horizons:
        deltas = []
        for _episode, group in df.groupby("episode_index", sort=True):
            idx = group.index.to_numpy()
            if len(idx) <= horizon:
                continue
            deltas.append(np.abs(actions[idx[horizon:]] - actions[idx[:-horizon]]))
        if deltas:
            future_delta_mean_abs[str(horizon)] = np.concatenate(deltas, axis=0).mean(axis=0).round(6).tolist()

    gripper = actions[:, 5]
    return {
        "rows": int(len(df)),
        "episodes": int(df["episode_index"].nunique()),
        "action_equals_state": bool(np.allclose(actions, states, atol=1e-7)),
        "mean_abs_action_minus_state": abs_action_state.mean(axis=0).round(8).tolist(),
        "max_abs_action_minus_state": abs_action_state.max(axis=0).round(8).tolist(),
        "future_delta_mean_abs": future_delta_mean_abs,
        "episode_length_min_mean_max": [
            int(episode_lengths.min()),
            float(np.round(episode_lengths.mean(), 2)),
            int(episode_lengths.max()),
        ],
        "start_state_mean": states[start_indices].mean(axis=0).round(6).tolist(),
        "end_state_mean": states[end_indices].mean(axis=0).round(6).tolist(),
        "gripper_action_min_mean_max": [
            float(np.round(gripper.min(), 6)),
            float(np.round(gripper.mean(), 6)),
            float(np.round(gripper.max(), 6)),
        ],
    }


def ensure_lerobot_available() -> None:
    if importlib.util.find_spec("lerobot") is None and shutil.which("lerobot-train") is None:
        raise RuntimeError(
            "LeRobot is not installed in this Python environment.\n"
            "Install it before running training, for example:\n"
            "  pip install 'lerobot[feetech]'\n"
            "or install the project version you used for the existing SO-101 setup."
        )


def _effective_batch_size(args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return args.batch_size
    if args.performance_profile == "corrected-act":
        return DEFAULT_CORRECTED_BATCH_SIZE
    return 8


def _steps_for_epochs(total_frames: int, batch_size: int, epochs: float) -> int:
    return max(1, math.ceil((total_frames * epochs) / batch_size))


def apply_performance_defaults(args: argparse.Namespace, summary: dict) -> None:
    batch_size = _effective_batch_size(args)

    if args.performance_profile in {"fast", "corrected-act"}:
        if args.use_amp is None:
            args.use_amp = True
        if args.target_epochs is None and args.steps is None:
            args.target_epochs = DEFAULT_FAST_TARGET_EPOCHS
        if args.save_every_epochs is None:
            args.save_every_epochs = DEFAULT_SAVE_EVERY_EPOCHS
        if args.performance_profile == "corrected-act":
            if args.batch_size is None:
                args.batch_size = DEFAULT_CORRECTED_BATCH_SIZE
            if args.chunk_size is None:
                args.chunk_size = DEFAULT_CORRECTED_CHUNK_SIZE
            if args.n_action_steps is None:
                args.n_action_steps = DEFAULT_CORRECTED_N_ACTION_STEPS
    elif args.use_amp is None:
        args.use_amp = False

    if args.steps is None and args.target_epochs is not None:
        args.steps = _steps_for_epochs(summary["total_frames"], batch_size, args.target_epochs)

    if args.save_every_epochs is not None:
        args.save_freq = _steps_for_epochs(summary["total_frames"], batch_size, args.save_every_epochs)
    else:
        args.save_freq = None


def _lerobot_train_command(args: argparse.Namespace) -> list[str]:
    if args.use_file_system_sharing:
        launcher = SCRIPT_DIR / "lerobot_train_filesystem.py"
        if launcher.exists():
            return [sys.executable, str(launcher)]
    lerobot_train = Path(sys.executable).with_name("lerobot-train")
    return [str(lerobot_train) if lerobot_train.exists() else "lerobot-train"]


def build_lerobot_command(args: argparse.Namespace, dataset_root: Path) -> list[str]:
    lerobot_train_cmd = _lerobot_train_command(args)
    if args.resume:
        config_path = args.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        return [
            *lerobot_train_cmd,
            f"--config_path={config_path}",
            "--resume=true",
        ]

    command = [
        *lerobot_train_cmd,
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=act",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        f"--wandb.enable={str(args.wandb_enabled).lower()}",
        "--policy.push_to_hub=false",
        f"--num_workers={args.num_workers}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
    ]
    if args.steps is not None:
        command.append(f"--steps={args.steps}")
    if args.batch_size is not None:
        command.append(f"--batch_size={args.batch_size}")
    if args.save_freq is not None:
        command.append(f"--save_freq={args.save_freq}")
    if args.log_freq is not None:
        command.append(f"--log_freq={args.log_freq}")
    if args.eval_freq is not None:
        command.append(f"--eval_freq={args.eval_freq}")
    if args.video_backend is not None:
        command.append(f"--dataset.video_backend={args.video_backend}")
    if args.chunk_size is not None:
        command.append(f"--policy.chunk_size={args.chunk_size}")
    if args.n_action_steps is not None:
        command.append(f"--policy.n_action_steps={args.n_action_steps}")
    return command


def main() -> int:
    args = parse_args()
    args.dataset_path = args.dataset_path.resolve()
    args.output_dir = args.output_dir.resolve()

    summary = validate_dataset(args.dataset_path)
    audit_summary = None
    if not args.skip_preflight:
        audit_summary = audit_action_contract(args.dataset_path)
    apply_performance_defaults(args, summary)
    if args.dry_run:
        args.wandb_enabled = False
    dataset_root = args.dataset_path
    command = build_lerobot_command(args, dataset_root)

    print("Validated physical SO-101 dataset:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if audit_summary is not None:
        print("\nDataset action/state preflight:")
        for key, value in audit_summary.items():
            print(f"  {key}: {value}")
    print("\nResolved training defaults:")
    print(f"  performance_profile: {args.performance_profile}")
    print(f"  batch_size: {_effective_batch_size(args)}")
    print(f"  use_amp: {args.use_amp}")
    print(f"  target_epochs: {args.target_epochs}")
    print(f"  steps: {args.steps if args.steps is not None else 'LeRobot default'}")
    print(f"  save_freq: {args.save_freq if args.save_freq is not None else 'LeRobot default'}")
    print("\nResolved LeRobot command:")
    print("  " + " \\\n    ".join(command))

    if args.dry_run:
        print("\nDry run complete; training was not launched.")
        return 0

    ensure_lerobot_available()
    return subprocess.call(command, cwd=SCRIPT_DIR)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetValidationError as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
