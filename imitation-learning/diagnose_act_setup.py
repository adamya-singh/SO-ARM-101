#!/usr/bin/env python3
"""
Diagnose the SO-101 ACT dataset, checkpoint, and physical inference logs.

This is an audit tool, not a trainer. It is designed to answer whether the
offline ACT setup has a coherent action/state contract before another long run
or physical motion test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "so101_pickplace_v1"
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


class DiagnosticError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--physical-log", type=Path, default=None, help="Optional run_act_physical_inference CSV log.")
    parser.add_argument("--samples", type=int, default=16, help="Recorded frames to use for checkpoint replay.")
    parser.add_argument("--horizon", type=int, default=30, help="Future-action horizon for replay diagnostics.")
    parser.add_argument(
        "--action-lead-steps",
        type=int,
        default=None,
        help="Action lead used for replay targets. Defaults to checkpoint config, or 0 for old checkpoints.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for checkpoint replay.")
    parser.add_argument("--skip-policy-replay", action="store_true", help="Only run metadata/parquet/log diagnostics.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DiagnosticError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"Invalid JSON in {path}: {exc}") from exc


def require_parquet_deps() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DiagnosticError(
            "Parquet diagnostics require numpy, pandas, and pyarrow. Activate the conda env first:\n"
            "  source /home/win10ubuntu/miniforge3/etc/profile.d/conda.sh\n"
            "  conda activate lerobot"
        ) from exc
    return np, pd


def load_dataset_frame(dataset_path: Path) -> tuple[Any, Any, Any]:
    np, pd = require_parquet_deps()
    data_files = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise DiagnosticError(f"No parquet files found under {dataset_path / 'data'}")
    frames = [
        pd.read_parquet(path, columns=["action", "observation.state", "episode_index", "frame_index", "index"])
        for path in data_files
    ]
    df = pd.concat(frames, ignore_index=True).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return np, pd, df


def validate_metadata(dataset_path: Path) -> dict[str, Any]:
    info = load_json(dataset_path / "meta" / "info.json")
    stats = load_json(dataset_path / "meta" / "stats.json")
    features = info.get("features", {})
    required = {
        "action": [6],
        "observation.state": [6],
        "observation.images.wrist": [256, 256, 3],
    }
    errors = []
    for key, shape in required.items():
        feature = features.get(key)
        if feature is None:
            errors.append(f"missing feature {key}")
        elif feature.get("shape") != shape:
            errors.append(f"{key} shape {feature.get('shape')} != {shape}")
    return {
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "features_ok": not errors,
        "feature_errors": errors,
        "stats_keys": sorted(stats.keys()),
        "data_files": len(list((dataset_path / "data").glob("chunk-*/*.parquet"))),
        "wrist_videos": len(list((dataset_path / "videos" / "observation.images.wrist").glob("chunk-*/*.mp4"))),
    }


def action_state_audit(dataset_path: Path, horizons: tuple[int, ...]) -> tuple[dict[str, Any], Any]:
    np, _pd, df = load_dataset_frame(dataset_path)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    abs_diff = np.abs(actions - states)
    ep_lengths = df.groupby("episode_index", sort=True).size().to_numpy()
    start_idx = df.groupby("episode_index", sort=True).head(1).index.to_numpy()
    end_idx = df.groupby("episode_index", sort=True).tail(1).index.to_numpy()

    future = {}
    for horizon in horizons:
        deltas = []
        for _episode, group in df.groupby("episode_index", sort=True):
            idx = group.index.to_numpy()
            if len(idx) > horizon:
                deltas.append(np.abs(actions[idx[horizon:]] - actions[idx[:-horizon]]))
        if deltas:
            future[str(horizon)] = np.concatenate(deltas, axis=0).mean(axis=0).round(6).tolist()

    gripper = actions[:, 5]
    audit = {
        "rows": int(len(df)),
        "episodes": int(df["episode_index"].nunique()),
        "action_equals_state": bool(np.allclose(actions, states, atol=1e-7)),
        "mean_abs_action_minus_state": abs_diff.mean(axis=0).round(8).tolist(),
        "max_abs_action_minus_state": abs_diff.max(axis=0).round(8).tolist(),
        "future_delta_mean_abs": future,
        "episode_length_min_mean_max": [
            int(ep_lengths.min()),
            float(np.round(ep_lengths.mean(), 2)),
            int(ep_lengths.max()),
        ],
        "start_state_mean": states[start_idx].mean(axis=0).round(6).tolist(),
        "end_state_mean": states[end_idx].mean(axis=0).round(6).tolist(),
        "gripper_action_min_mean_max": [
            float(np.round(gripper.min(), 6)),
            float(np.round(gripper.mean(), 6)),
            float(np.round(gripper.max(), 6)),
        ],
        "gripper_motion_mean_abs_1f": float(np.round(np.abs(np.diff(gripper)).mean(), 6)),
    }
    return audit, df


def checkpoint_audit(checkpoint: Path) -> dict[str, Any]:
    required = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json", "train_config.json"]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    empty = [name for name in required if (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == 0]
    if missing:
        return {"exists": False, "missing": missing, "empty": empty}

    config = load_json(checkpoint / "config.json")
    return {
        "exists": True,
        "missing": missing,
        "empty": empty,
        "chunk_size": config.get("chunk_size"),
        "n_action_steps": config.get("n_action_steps"),
        "action_lead_steps": config.get("action_lead_steps", 0),
        "n_obs_steps": config.get("n_obs_steps"),
        "input_features": config.get("input_features"),
        "output_features": config.get("output_features"),
        "normalization_mapping": config.get("normalization_mapping"),
        "vision_backbone": config.get("vision_backbone"),
        "use_vae": config.get("use_vae"),
        "kl_weight": config.get("kl_weight"),
    }


def select_valid_indices(df: Any, samples: int, horizon: int, action_lead_steps: int) -> list[int]:
    valid = []
    for _episode, group in df.groupby("episode_index", sort=True):
        idx = group.index.to_numpy()
        required_future = action_lead_steps + horizon
        if len(idx) > required_future:
            valid.extend(idx[:-required_future].tolist())
    if not valid:
        return []
    if samples >= len(valid):
        return valid
    step = max(1, len(valid) // samples)
    return valid[::step][:samples]


def add_batch_dim(batch: dict[str, Any], torch: Any) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.unsqueeze(0) if value.ndim in {1, 3} else value
        else:
            result[key] = value
    return result


def run_policy_replay(
    dataset_path: Path,
    checkpoint: Path,
    df: Any,
    samples: int,
    horizon: int,
    action_lead_steps: int,
    device_name: str,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:
        raise DiagnosticError(
            "Policy replay requires torch and LeRobot. Activate the `lerobot` conda environment first."
        ) from exc

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ds = LeRobotDataset(repo_id="local/so101_pickplace_v1", root=str(dataset_path))
    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    indices = select_valid_indices(df, samples, horizon, action_lead_steps)
    if not indices:
        return {
            "samples": 0,
            "action_lead_steps": action_lead_steps,
            "error": f"No valid samples with horizon={horizon} and action_lead_steps={action_lead_steps}",
        }

    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    one_step_errors = []
    horizon_errors = []
    predicted_actions = []
    blank_deltas = []

    for idx in indices:
        sample = ds[int(idx)]
        obs = {
            "observation.images.wrist": sample["observation.images.wrist"],
            "observation.state": sample["observation.state"],
        }
        model_obs = preprocessor(obs)
        model_obs = add_batch_dim(model_obs, torch)
        with torch.inference_mode():
            try:
                pred_chunk = policy.predict_action_chunk(model_obs)
                pred_chunk = postprocessor(pred_chunk)
                pred = pred_chunk.detach().cpu().numpy().reshape(-1, 6)
            except Exception:
                policy.reset()
                pred_one = postprocessor(policy.select_action(model_obs))
                pred = pred_one.detach().cpu().numpy().reshape(1, 6)

            blank_obs = dict(obs)
            blank_obs["observation.images.wrist"] = torch.zeros_like(obs["observation.images.wrist"])
            blank_model_obs = add_batch_dim(preprocessor(blank_obs), torch)
            policy.reset()
            blank_action = postprocessor(policy.select_action(blank_model_obs)).detach().cpu().numpy().reshape(-1)

        pred0 = pred[0]
        predicted_actions.append(pred0)
        blank_deltas.append(np.abs(pred0 - blank_action).mean())
        target_start = idx + action_lead_steps
        one_step_errors.append(np.abs(pred0 - actions[target_start]).mean())
        gt_horizon = actions[target_start : target_start + min(horizon, len(pred))]
        pred_horizon = pred[: len(gt_horizon)]
        horizon_errors.append(np.abs(pred_horizon - gt_horizon).mean())

    predicted = np.asarray(predicted_actions)
    return {
        "samples": len(indices),
        "device": str(device),
        "action_lead_steps": action_lead_steps,
        "one_step_mean_abs_error": float(np.round(np.mean(one_step_errors), 6)),
        "horizon_mean_abs_error": float(np.round(np.mean(horizon_errors), 6)),
        "predicted_action_mean": predicted.mean(axis=0).round(6).tolist(),
        "predicted_action_std": predicted.std(axis=0).round(6).tolist(),
        "predicted_gripper_min_mean_max": [
            float(np.round(predicted[:, 5].min(), 6)),
            float(np.round(predicted[:, 5].mean(), 6)),
            float(np.round(predicted[:, 5].max(), 6)),
        ],
        "blank_image_mean_abs_action_delta": float(np.round(np.mean(blank_deltas), 6)),
    }


def physical_log_audit(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"rows": 0}
    try:
        import numpy as np
    except ImportError as exc:
        raise DiagnosticError("Physical log audit requires numpy.") from exc

    action_cols = [name for name in rows[0] if name.startswith("action_rad_")]
    state_cols = [name for name in rows[0] if name.startswith("state_")]
    actions = np.asarray([[float(row[col]) for col in action_cols] for row in rows], dtype=np.float32)
    states = np.asarray([[float(row[col]) for col in state_cols] for row in rows], dtype=np.float32)
    action_delta = np.abs(np.diff(actions, axis=0)) if len(actions) > 1 else np.zeros((0, actions.shape[1]), dtype=np.float32)
    state_delta = states[-1] - states[0]
    return {
        "rows": len(rows),
        "mean_abs_action_step_delta": action_delta.mean(axis=0).round(6).tolist() if len(action_delta) else [0.0] * actions.shape[1],
        "state_start_to_end_delta": state_delta.round(6).tolist(),
        "action_gripper_min_mean_max": [
            float(np.round(actions[:, 5].min(), 6)),
            float(np.round(actions[:, 5].mean(), 6)),
            float(np.round(actions[:, 5].max(), 6)),
        ],
    }


def print_section(title: str, data: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in data.items():
        print(f"{key}: {value}")


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    metadata = validate_metadata(dataset_path)
    action_audit, df = action_state_audit(dataset_path, horizons=(1, args.horizon, 100))
    ckpt = checkpoint_audit(checkpoint)
    action_lead_steps = args.action_lead_steps
    if action_lead_steps is None:
        action_lead_steps = int(ckpt.get("action_lead_steps", 0) or 0)
    if action_lead_steps < 0:
        raise DiagnosticError(f"--action-lead-steps must be non-negative, got {action_lead_steps}")

    print_section("Dataset Metadata", metadata)
    print_section("Action/State Contract", action_audit)
    print_section("Checkpoint", ckpt)

    if args.physical_log is not None:
        print_section("Physical Inference Log", physical_log_audit(args.physical_log.expanduser().resolve()))

    if not args.skip_policy_replay and ckpt.get("exists") and not ckpt.get("missing") and not ckpt.get("empty"):
        print_section(
            "Offline Policy Replay",
            run_policy_replay(
                dataset_path,
                checkpoint,
                df,
                args.samples,
                args.horizon,
                action_lead_steps,
                args.device,
            ),
        )
    elif args.skip_policy_replay:
        print("\nOffline Policy Replay\n---------------------\nskipped by --skip-policy-replay")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"diagnose_act_setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"diagnose_act_setup failed unexpectedly: {exc}", file=sys.stderr)
        raise SystemExit(1)
