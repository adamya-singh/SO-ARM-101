"""Current-pose and train-mean future-target baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from so_arm101_v2.contracts import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    PHYSICAL_NORMALIZED_HIGH,
    PHYSICAL_NORMALIZED_LOW,
    act_to_mujoco_qpos,
    act_to_physical_normalized,
    physical_normalized_to_act,
)
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.data.targets import (
    FutureTargetSpec,
    build_future_target_index,
    gather_future_state_targets,
)

if TYPE_CHECKING:
    from so_arm101_v2.data import DatasetInventory, SplitManifest


@dataclass(frozen=True)
class BaselineEvaluation:
    payload: dict[str, Any]

    @property
    def dataset_digest(self) -> str:
        return str(self.payload["dataset_digest"])

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def _vector_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def compute_pose_metrics(
    current: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    act_mae = np.mean(np.abs(difference), axis=0)
    act_rmse = np.sqrt(np.mean(np.square(difference), axis=0))
    act_l2 = np.linalg.norm(difference, axis=1)

    target_mujoco = act_to_mujoco_qpos(target).astype(np.float64)
    predicted_mujoco = act_to_mujoco_qpos(prediction).astype(np.float64)
    mujoco_difference = predicted_mujoco - target_mujoco

    hard_clipped = np.clip(prediction, ACT_DATASET_LOW, ACT_DATASET_HIGH).astype(np.float32)
    hard_mask = hard_clipped != prediction
    current_physical = np.clip(
        act_to_physical_normalized(current),
        PHYSICAL_NORMALIZED_LOW,
        PHYSICAL_NORMALIZED_HIGH,
    )
    requested_physical = act_to_physical_normalized(prediction)
    hard_physical = np.clip(
        requested_physical, PHYSICAL_NORMALIZED_LOW, PHYSICAL_NORMALIZED_HIGH
    )
    limited_physical = current_physical + np.clip(
        hard_physical - current_physical, -20.0, 20.0
    )
    relative_mask = limited_physical != hard_physical
    executed_act = physical_normalized_to_act(limited_physical)
    executed_difference = executed_act.astype(np.float64) - target.astype(np.float64)

    return {
        "act_mae_per_joint": act_mae.tolist(),
        "act_rmse_per_joint": act_rmse.tolist(),
        "act_l2": _vector_stats(act_l2),
        "normalized_mse": float(np.mean(np.square(
            (prediction - target) / ((ACT_DATASET_HIGH - ACT_DATASET_LOW) / 2.0)
        ))),
        "mujoco_mae_per_joint": np.mean(np.abs(mujoco_difference), axis=0).tolist(),
        "mujoco_rmse_per_joint": np.sqrt(
            np.mean(np.square(mujoco_difference), axis=0)
        ).tolist(),
        "hard_clip_sample_fraction": float(np.mean(np.any(hard_mask, axis=1))),
        "hard_clip_per_joint_fraction": np.mean(hard_mask, axis=0).tolist(),
        "relative_limit_sample_fraction": float(np.mean(np.any(relative_mask, axis=1))),
        "relative_limit_per_joint_fraction": np.mean(relative_mask, axis=0).tolist(),
        "executed_act_mae_per_joint": np.mean(
            np.abs(executed_difference), axis=0
        ).tolist(),
        "executed_act_rmse_per_joint": np.sqrt(
            np.mean(np.square(executed_difference), axis=0)
        ).tolist(),
        "executed_act_l2": _vector_stats(np.linalg.norm(executed_difference, axis=1)),
    }


def baseline_resource_name(dataset_digest: str) -> str:
    return f"so101_pickplace_v1.baselines.v1.{dataset_digest[:12]}.json"


def evaluate_baselines(
    inventory: "DatasetInventory",
    manifests: Mapping[str, "SplitManifest"],
    *,
    leads: tuple[int, ...] = (1, 3, 5),
    output_dir: str | Path | None = None,
) -> BaselineEvaluation:
    """Evaluate baselines with means fitted only on the training split."""
    if leads != (1, 3, 5):
        raise ValueError("baseline report v1 is fixed to leads (1, 3, 5)")
    if set(manifests) != {"train", "validation", "test"}:
        raise ValueError("train, validation, and test manifests are required")
    for manifest in manifests.values():
        if manifest.dataset_digest != inventory.dataset_digest:
            raise ValueError("split manifest dataset digest mismatch")

    lead_reports: dict[str, Any] = {}
    for lead in leads:
        subsets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split, manifest in manifests.items():
            mask = np.isin(inventory.episode_indices, manifest.episode_ids)
            states = inventory.states[mask]
            index = build_future_target_index(
                inventory.episode_indices[mask],
                inventory.frame_indices[mask],
                inventory.timestamps[mask],
                FutureTargetSpec(lead),
                fps=inventory.fps,
            )
            subsets[split] = (
                states[index.anchor_indices], gather_future_state_targets(states, index)
            )
        mean_target = np.mean(subsets["train"][1], axis=0, dtype=np.float64).astype(np.float32)
        split_reports = {}
        for split, (current, target) in subsets.items():
            split_reports[split] = {
                "samples": len(target),
                "current_pose": compute_pose_metrics(current, target, current),
                "train_mean_target": compute_pose_metrics(
                    current, target, np.broadcast_to(mean_target, target.shape)
                ),
            }
        lead_reports[str(lead)] = {
            "lead_seconds": lead / inventory.fps,
            "train_mean_target": mean_target.tolist(),
            "splits": split_reports,
        }

    coordinate = inventory.coordinate_contract
    calibration = inventory.calibration
    body: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": inventory.dataset_id,
        "dataset_digest": inventory.dataset_digest,
        "joint_order": inventory.joints["order"],
        "coordinate_contract_sha256": coordinate["sha256"],
        "calibration_sha256": calibration["sha256"],
        "split_content_sha256": {
            name: manifest.content_sha256 for name, manifest in sorted(manifests.items())
        },
        "mean_fit_split": "train",
        "max_relative_target": 20.0,
        "leads": lead_reports,
    }
    body["content_sha256"] = content_sha256(body)
    result = BaselineEvaluation(body)
    if output_dir is not None:
        write_immutable_json(
            Path(output_dir) / baseline_resource_name(inventory.dataset_digest),
            result.to_dict(),
        )
    return result


__all__ = [
    "BaselineEvaluation",
    "baseline_resource_name",
    "compute_pose_metrics",
    "evaluate_baselines",
]
