"""Closed-loop policy wrapper for privileged oracle-clone checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from so_arm101_v2.data._serialization import content_sha256

from so_arm101_v2.learning.oracle_distillation import (
    OracleCloneKind,
    build_oracle_clone_model,
    build_oracle_features,
    oracle_feature_schema,
)


class OracleCloneCheckpointPolicy:
    requires_pixels = False

    def __init__(self, checkpoint: str | Path) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("oracle clone inference requires the 'learn' extra") from exc
        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported oracle clone checkpoint schema")
        self.kind = OracleCloneKind(payload["model_kind"])
        self.input_dim = int(payload["input_dim"])
        expected_input_dim = {
            OracleCloneKind.PHASE_STATE: 10,
            OracleCloneKind.FEEDBACK_STATE: 25,
            OracleCloneKind.PHASE_DYNAMICS: 26,
            OracleCloneKind.PHASE_DYNAMICS_CONTACT: 29,
            OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2: 57,
        }[self.kind]
        if self.input_dim != expected_input_dim:
            raise ValueError("oracle clone input dimension does not match feature schema")
        self.hidden_width = int(payload.get("hidden_width", 128))
        self.extras_mean = np.asarray(payload["extras_mean"], dtype=np.float32)
        self.extras_std = np.asarray(payload["extras_std"], dtype=np.float32)
        self.feature_schema = tuple(payload.get("feature_schema", oracle_feature_schema(self.kind)))
        if self.feature_schema != oracle_feature_schema(self.kind):
            raise ValueError("oracle clone feature schema metadata is malformed")
        self.history_length = int(payload.get("history_length", 1))
        self.maximum_delta = np.asarray(payload["maximum_act_delta_per_step"], dtype=np.float32)
        self.teacher_horizon = int(payload["teacher_horizon"])
        if (
            self.maximum_delta.shape != (6,)
            or self.teacher_horizon != 450
            or not isinstance(payload.get("manifest_content_sha256"), str)
            or not isinstance(payload.get("report_content_sha256"), str)
        ):
            raise ValueError("oracle clone checkpoint metadata is malformed")
        report_path = checkpoint_path.with_name("report.json")
        if not report_path.is_file():
            raise FileNotFoundError(f"oracle clone report does not exist: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stated = report.get("content_sha256")
        report_body = dict(report)
        report_body.pop("content_sha256", None)
        if stated != content_sha256(report_body) or stated != payload["report_content_sha256"]:
            raise ValueError("oracle clone report content hash mismatch")
        if (
            report.get("passed") is not True
            or report.get("closed_loop_eligible") is not True
            or payload.get("closed_loop_eligible") is not True
            or report.get("training_row_mode") != "full"
            or payload.get("training_row_mode") != "full"
            or (
                report.get("learning_rate_schedule", "fixed") == "decay_10k_20k"
                and (
                    not (
                        report.get("prefix_parity", {}).get("required") is True
                        and report.get("prefix_parity", {}).get("passed") is True
                    )
                    and not (
                        report.get("recovery_augmentation") is not None
                        and report.get("prefix_parity", {}).get("applicable") is False
                        and isinstance(report.get("prefix_parity", {}).get("reference_content_sha256"), str)
                    )
                    and not (
                        self.kind in (
                            OracleCloneKind.PHASE_DYNAMICS,
                            OracleCloneKind.PHASE_DYNAMICS_CONTACT,
                            OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
                        )
                        and report.get("recovery_augmentation") is not None
                        and report.get("prefix_parity", {}).get("required") is False
                        and report.get("prefix_parity", {}).get("applicable") is False
                        and report.get("prefix_parity", {}).get("reference_content_sha256") is None
                    )
                    and not (
                        report.get("correction_augmentation") is not None
                        and report.get("prefix_parity", {}).get("applicable") is False
                        and isinstance(
                            report.get("prefix_parity", {}).get("reference_content_sha256"), str
                        )
                    )
                )
            )
        ):
            raise ValueError("oracle clone checkpoint is not eligible for closed-loop evaluation")
        if (
            int(report.get("hidden_width", 128)) != self.hidden_width
            or report.get("manifest_content_sha256") != payload.get("manifest_content_sha256")
            or report.get("training_row_indices") != payload.get("training_row_indices")
            or tuple(report.get("feature_schema", oracle_feature_schema(self.kind))) != self.feature_schema
            or int(report.get("history_length", 1)) != self.history_length
        ):
            raise ValueError("oracle clone checkpoint and report metadata disagree")
        contact_kinds = (
            OracleCloneKind.PHASE_DYNAMICS_CONTACT,
            OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
        )
        if self.kind in contact_kinds:
            report_observability = report.get("observability_annotations")
            checkpoint_observability = payload.get("observability_annotations")
            if (
                not isinstance(report_observability, dict)
                or not isinstance(checkpoint_observability, dict)
                or not isinstance(report_observability.get("manifest_content_sha256"), str)
                or report_observability != checkpoint_observability
                or report.get("contact_timing") != "pre_action_before_current_command_selection"
                or payload.get("contact_timing") != report.get("contact_timing")
            ):
                raise ValueError("oracle clone observability metadata disagree")
        report_correction = report.get("correction_augmentation")
        checkpoint_correction = payload.get("correction_augmentation")
        if report_correction is not None or checkpoint_correction is not None:
            if (
                not isinstance(report_correction, dict)
                or report_correction != checkpoint_correction
                or not isinstance(report_correction.get("manifest_content_sha256"), str)
            ):
                raise ValueError("oracle clone correction metadata disagree")
        expected_history = (
            2 if self.kind is OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2 else 1
        )
        if self.history_length != expected_history or (
            expected_history == 2
            and (
                report.get("history_padding") != "repeat_current_at_episode_reset"
                or payload.get("history_padding") != report.get("history_padding")
            )
        ):
            raise ValueError("oracle clone history metadata disagree")
        self.report_content_sha256 = stated
        self.manifest_content_sha256 = payload["manifest_content_sha256"]
        self.model = build_oracle_clone_model(self.input_dim, self.hidden_width)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.torch = torch
        self.action_index = 0
        self.seed = int(payload["config"]["seed"])

    @property
    def policy_id(self) -> str:
        return f"{self.kind.value}.seed{self.seed}"

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del image
        if adapter is None:
            raise ValueError("oracle clone policy requires a MuJoCo adapter")
        snapshot = adapter.privileged_state()
        if np.max(np.abs(snapshot.current_act - np.asarray(current_act, dtype=np.float32))) > 1e-5:
            raise RuntimeError("adapter state changed during oracle clone observation")
        arrays: dict[str, np.ndarray] = {
            "current_act": snapshot.current_act[None],
            "robot_qvel": snapshot.robot_qvel[None],
            "cube_position": snapshot.cube_position[None],
            "cube_quaternion_wxyz": snapshot.cube_quaternion_wxyz[None],
            "cube_linear_velocity": snapshot.cube_linear_velocity[None],
            "cube_angular_velocity": snapshot.cube_angular_velocity[None],
            "progress": np.asarray([
                min(self.action_index, self.teacher_horizon - 1) / (self.teacher_horizon - 1)
            ], dtype=np.float32),
        }
        observability: dict[str, np.ndarray] | None = None
        if self.kind in (
            OracleCloneKind.PHASE_DYNAMICS_CONTACT,
            OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
        ):
            observability = {
                "contact_flags": adapter.privileged_contact_state().as_array()[None],
            }
        if self.kind is OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2:
            assert observability is not None
            previous = adapter.previous_privileged_state()
            for name in previous.__dataclass_fields__:
                observability[f"previous_{name}"] = np.asarray(
                    getattr(previous, name), dtype=np.float32,
                )[None]
            observability["previous_contact_flags"] = (
                adapter.previous_privileged_contact_state().as_array()[None]
            )
        features, _, _ = build_oracle_features(
            self.kind, arrays, observability=observability,
            extras_mean=self.extras_mean, extras_std=self.extras_std,
        )
        if features.shape != (1, self.input_dim):
            raise ValueError("oracle clone feature schema does not match checkpoint")
        with self.torch.inference_mode():
            normalized_delta = self.model(self.torch.from_numpy(features)).numpy()[0]
        self.action_index += 1
        return np.asarray(
            snapshot.current_act + normalized_delta.astype(np.float32) * self.maximum_delta,
            dtype=np.float32,
        )


__all__ = ["OracleCloneCheckpointPolicy"]
