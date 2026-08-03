"""Small deterministic residual policies for privileged oracle distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES, evaluate_physical_command, load_task_contract
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.simulation.oracle import load_oracle_demonstrations
from so_arm101_v2.simulation.recovery import load_oracle_recovery_examples

from .tiny_model import normalize_act


class OracleCloneKind(str, Enum):
    PHASE_STATE = "phase_state"
    FEEDBACK_STATE = "feedback_state"
    PHASE_DYNAMICS = "phase_dynamics"
    PHASE_DYNAMICS_CONTACT = "phase_dynamics_contact"
    PHASE_DYNAMICS_CONTACT_HISTORY2 = "phase_dynamics_contact_history2"


class OracleTrainingRows(str, Enum):
    FULL = "full"
    MEMORIZATION32 = "memorization32"
    MEMORIZATION64 = "memorization64"
    MEMORIZATION128 = "memorization128"
    MEMORIZATION256 = "memorization256"


class OracleLearningRateSchedule(str, Enum):
    FIXED = "fixed"
    DECAY_10K_20K = "decay_10k_20k"


ORACLE_MEMORIZATION32_ROWS = (
    0, 34, 70, 89, 110, 127, 145, 147, 159, 175, 189, 205, 229, 255, 269, 285,
    299, 315, 332, 350, 357, 365, 375, 386, 394, 403, 410, 419, 421, 424, 436, 449,
)


@dataclass(frozen=True)
class OracleDistillationConfig:
    seed: int = 101
    learning_rate: float = 1e-3
    max_steps: int = 10_000
    normalized_mse_threshold: float = 1e-6
    max_act_error_threshold: float = 0.01
    baseline_improvement_factor: float = 100.0
    hidden_width: int = 128
    training_rows: str = OracleTrainingRows.FULL.value
    lr_schedule: str = OracleLearningRateSchedule.FIXED.value
    recovery_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.seed < 0 or self.learning_rate <= 0 or self.max_steps <= 0:
            raise ValueError("invalid oracle distillation configuration")
        if not np.isfinite(self.recovery_loss_weight) or not 0.0 <= self.recovery_loss_weight <= 1.0:
            raise ValueError("recovery_loss_weight must be finite and between 0 and 1")
        if self.hidden_width not in (128, 256):
            raise ValueError("oracle clone hidden_width must be 128 or 256")
        try:
            OracleTrainingRows(self.training_rows)
        except ValueError as exc:
            raise ValueError("unsupported oracle training_rows mode") from exc
        try:
            schedule = OracleLearningRateSchedule(self.lr_schedule)
        except ValueError as exc:
            raise ValueError("unsupported oracle learning-rate schedule") from exc
        if schedule is OracleLearningRateSchedule.DECAY_10K_20K and (
            self.seed != 101
            or self.learning_rate != 1e-3
            or self.max_steps != 30_000
            or self.hidden_width != 256
            or self.normalized_mse_threshold != 1e-6
            or self.max_act_error_threshold != 0.01
            or self.baseline_improvement_factor != 100.0
        ):
            raise ValueError(
                "decay_10k_20k requires seed 101, width 256, lr 1e-3, "
                "30,000 steps, and the immutable gate thresholds"
            )


@dataclass(frozen=True)
class OracleDistillationResult:
    kind: OracleCloneKind
    passed: bool
    closed_loop_eligible: bool
    steps: int
    normalized_mse: float
    max_act_error: float
    directory: Path
    checkpoint: Path
    report_json: Path
    report_html: Path


@dataclass(frozen=True)
class OracleResidualAnalysisResult:
    directory: Path
    report_json: Path
    report_html: Path


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("oracle distillation requires the 'learn' extra") from exc
    return torch


def build_oracle_clone_model(input_dim: int, hidden_width: int = 128) -> Any:
    torch = _torch()
    if hidden_width not in (128, 256):
        raise ValueError("oracle clone hidden_width must be 128 or 256")

    class OracleCloneNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_width),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_width, 6),
            )
            with torch.no_grad():
                self.network[-1].weight.zero_()
                self.network[-1].bias.zero_()

        def forward(self, features: Any) -> Any:
            return self.network(features)

    return OracleCloneNetwork()


def oracle_training_row_indices(mode: OracleTrainingRows | str, source_rows: int) -> np.ndarray:
    """Resolve the immutable training-row selection against a full oracle capture."""
    mode = OracleTrainingRows(mode)
    if source_rows <= 0:
        raise ValueError("oracle source must contain at least one row")
    if mode is OracleTrainingRows.FULL:
        return np.arange(source_rows, dtype=np.int64)
    if source_rows != 450:
        raise ValueError("memorization rungs require the canonical 450-row oracle capture")
    target_count = {
        OracleTrainingRows.MEMORIZATION32: 32,
        OracleTrainingRows.MEMORIZATION64: 64,
        OracleTrainingRows.MEMORIZATION128: 128,
        OracleTrainingRows.MEMORIZATION256: 256,
    }[mode]
    selected = set(ORACLE_MEMORIZATION32_ROWS)
    if len(selected) != 32 or min(selected) < 0 or max(selected) >= source_rows:
        raise RuntimeError("memorization32 row contract is invalid")
    while len(selected) < target_count:
        ordered = sorted(selected)
        gaps = [
            (right - left, -left, (left + right) // 2)
            for left, right in zip(ordered, ordered[1:])
            if right - left > 1
        ]
        if not gaps:
            raise RuntimeError("cannot construct requested memorization rung")
        selected.add(max(gaps)[2])
    return np.asarray(sorted(selected), dtype=np.int64)


def count_oracle_command_safety_violations(
    current_act: np.ndarray, predicted_act: np.ndarray
) -> int:
    current_values = np.asarray(current_act, dtype=np.float32)
    predicted_values = np.asarray(predicted_act, dtype=np.float32)
    if current_values.shape != predicted_values.shape or current_values.ndim != 2 or current_values.shape[1] != 6:
        raise ValueError("oracle safety rows must be matching [N, 6] arrays")
    violations = 0
    for current, predicted in zip(current_values, predicted_values, strict=True):
        evaluation = evaluate_physical_command(current, predicted)
        violations += int(any((
            np.any(evaluation.act_clip_mask),
            np.any(evaluation.mujoco_clip_mask),
            np.any(evaluation.physical_clip_mask),
            np.any(evaluation.relative_limit_mask),
        )))
    return violations


_DYNAMIC_EXTRA_FIELDS = (
    "robot_qvel", "cube_position", "cube_quaternion_wxyz",
    "cube_linear_velocity", "cube_angular_velocity",
)


def _raw_extras(kind: OracleCloneKind, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if kind is OracleCloneKind.PHASE_STATE:
        return np.asarray(arrays["cube_position"], dtype=np.float32)
    return np.concatenate(
        tuple(np.asarray(arrays[name], dtype=np.float32) for name in _DYNAMIC_EXTRA_FIELDS),
        axis=1,
    ).astype(np.float32)


def oracle_feature_schema(kind: OracleCloneKind | str) -> tuple[str, ...]:
    kind = OracleCloneKind(kind)
    if kind is OracleCloneKind.PHASE_STATE:
        return ("current_act[6]", "cube_position[3]", "progress[1]")
    dynamics = (
        "current_act[6]", "robot_qvel[6]", "cube_position[3]",
        "cube_quaternion_wxyz[4]", "cube_linear_velocity[3]",
        "cube_angular_velocity[3]",
    )
    if kind is OracleCloneKind.FEEDBACK_STATE:
        return dynamics
    current = dynamics + ("contact_flags[3]",)
    if kind is OracleCloneKind.PHASE_DYNAMICS:
        return dynamics + ("progress[1]",)
    if kind is OracleCloneKind.PHASE_DYNAMICS_CONTACT:
        return current + ("progress[1]",)
    return current + tuple(f"previous_{name}" for name in current) + ("progress[1]",)


def build_oracle_features(
    kind: OracleCloneKind | str,
    arrays: Mapping[str, np.ndarray],
    *,
    observability: Mapping[str, np.ndarray] | None = None,
    extras_mean: np.ndarray | None = None,
    extras_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact training features and return the stored normalization."""
    kind = OracleCloneKind(kind)
    current = normalize_act(np.asarray(arrays["current_act"], dtype=np.float32))
    extras = _raw_extras(kind, arrays)
    mean = extras.mean(axis=0, dtype=np.float64).astype(np.float32) if extras_mean is None else np.asarray(extras_mean, dtype=np.float32)
    std = extras.std(axis=0, dtype=np.float64).astype(np.float32) if extras_std is None else np.asarray(extras_std, dtype=np.float32)
    std = np.maximum(std, np.float32(1e-6))
    if mean.shape != (extras.shape[1],) or std.shape != (extras.shape[1],):
        raise ValueError("oracle feature normalization shape mismatch")
    normalized_extras = (extras - mean) / std
    parts = [current, normalized_extras]
    if kind is OracleCloneKind.PHASE_STATE:
        parts.append(np.asarray(arrays["progress"], dtype=np.float32).reshape(-1, 1))
    elif kind in (
        OracleCloneKind.PHASE_DYNAMICS,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
    ):
        if kind is not OracleCloneKind.PHASE_DYNAMICS:
            if observability is None or "contact_flags" not in observability:
                raise ValueError(f"{kind.value} requires aligned observability annotations")
            contacts = np.asarray(observability["contact_flags"], dtype=np.float32)
            if contacts.shape != (current.shape[0], 3) or not np.all(np.isin(contacts, (0.0, 1.0))):
                raise ValueError("contact flags must be binary [N, 3]")
            parts.append(contacts)
        if kind is OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2:
            assert observability is not None
            previous_current = normalize_act(
                np.asarray(observability["previous_current_act"], dtype=np.float32)
            )
            previous_extras = np.concatenate(
                tuple(
                    np.asarray(observability[f"previous_{name}"], dtype=np.float32)
                    for name in _DYNAMIC_EXTRA_FIELDS
                ),
                axis=1,
            ).astype(np.float32)
            if previous_current.shape != current.shape or previous_extras.shape != extras.shape:
                raise ValueError("two-frame observability shape mismatch")
            previous_contacts = np.asarray(
                observability["previous_contact_flags"], dtype=np.float32
            )
            if (
                previous_contacts.shape != (current.shape[0], 3)
                or not np.all(np.isin(previous_contacts, (0.0, 1.0)))
            ):
                raise ValueError("previous contact flags must be binary [N, 3]")
            parts.extend((previous_current, (previous_extras - mean) / std, previous_contacts))
        parts.append(np.asarray(arrays["progress"], dtype=np.float32).reshape(-1, 1))
    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("oracle features contain nonfinite values")
    return features, mean, std


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable oracle model artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _svg_series(target: np.ndarray, prediction: np.ndarray, boundaries: list[int]) -> str:
    width, height = 760, 130
    low = float(min(target.min(), prediction.min()))
    high = float(max(target.max(), prediction.max()))
    span = max(high - low, 1e-8)

    def points(values: np.ndarray) -> str:
        return " ".join(
            f"{index * (width - 20) / max(len(values) - 1, 1) + 10:.2f},"
            f"{height - 10 - (float(value) - low) * (height - 20) / span:.2f}"
            for index, value in enumerate(values)
        )

    markers = "".join(
        f'<line x1="{boundary * (width - 20) / max(len(target) - 1, 1) + 10:.2f}" y1="5" '
        f'x2="{boundary * (width - 20) / max(len(target) - 1, 1) + 10:.2f}" y2="{height - 5}" stroke="#394657"/>'
        for boundary in boundaries if boundary < len(target)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">{markers}'
        f'<polyline fill="none" stroke="#70b7ff" stroke-width="2" points="{points(target)}"/>'
        f'<polyline fill="none" stroke="#ffb86b" stroke-width="1.5" points="{points(prediction)}"/>'
        "</svg>"
    )


def _diagnostic_html(report: Mapping[str, Any], targets: np.ndarray, predictions: np.ndarray) -> str:
    boundaries: list[int] = []
    offset = 0
    for episode in report["episodes"]:
        boundaries.extend(offset + int(value) for value in episode["waypoint_boundaries"])
        offset += int(episode["rows"])
    boundaries = sorted(set(boundaries))
    full_trajectory = report["training_row_mode"] == OracleTrainingRows.FULL.value
    plot_boundaries = boundaries if full_trajectory else []
    plots = "".join(
        f"<h3>{html.escape(name)}</h3>{_svg_series(targets[:, index], predictions[:, index], plot_boundaries)}"
        for index, name in enumerate(JOINT_NAMES)
    )
    source_indices: list[int | str] = [int(value) for value in report["training_row_indices"]]
    augmentation = report.get("recovery_augmentation")
    if augmentation is not None:
        source_indices.extend(f"recovery:{name}" for name in augmentation["anchors"])
    if full_trajectory:
        diagnostic_indices = sorted({0, len(targets) - 1, *(min(value, len(targets) - 1) for value in boundaries)})
        plot_note = "Vertical lines are teacher stage boundaries and are not model inputs."
    else:
        diagnostic_indices = list(range(len(targets)))
        plot_note = "Subset points are ordered by source row; stage boundaries are listed in the table."
    rows = "".join(
        "<tr>"
        f"<td>{source_indices[index]}</td><td>{html.escape(str(report['row_scenarios'][index]))}</td>"
        f"<td>{html.escape(np.array2string(targets[index], precision=4))}</td>"
        f"<td>{html.escape(np.array2string(predictions[index], precision=4))}</td>"
        f"<td>{float(np.max(np.abs(targets[index] - predictions[index]))):.7f}</td>"
        "</tr>"
        for index in diagnostic_indices
    )
    status = "PASS" if report["passed"] else "FAIL"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Oracle distillation</title>
<style>body{{font-family:system-ui;background:#10151c;color:#e7edf5;margin:24px}}svg{{background:#18212c;max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #445;padding:5px;font-family:monospace}}.pass{{color:#75d18a}}.fail{{color:#ff6b6b}}</style></head>
<body><h1>Privileged oracle distillation: {html.escape(report['model_kind'])}</h1>
<h2 class="{'pass' if report['passed'] else 'fail'}">{status}</h2>
<p>Training rows {report['rows']} of {report['source_rows']} · mode {html.escape(report['training_row_mode'])} · hidden width {report['hidden_width']} · steps {report['steps']} · normalized delta MSE {report['normalized_mse']:.10g} · max ACT error {report['max_act_error']:.10g}</p>
<p>Cumulative absolute ACT error by joint: {html.escape(str(report['cumulative_absolute_act_error_per_joint']))}. Worst row: {report['worst_error']['row']} / {html.escape(report['worst_error']['joint'])}.</p>
<p><span style="color:#70b7ff">teacher</span> / <span style="color:#ffb86b">prediction</span>. {plot_note}</p>
{plots}<h2>Fixed diagnostic examples</h2><table><tr><th>source row</th><th>scenario</th><th>teacher ACT</th><th>predicted ACT</th><th>max error</th></tr>{rows}</table>
<p>Privileged simulation diagnostic only; this is not a deployment policy.</p></body></html>"""


def _result(directory: Path, report: Mapping[str, Any]) -> OracleDistillationResult:
    return OracleDistillationResult(
        kind=OracleCloneKind(report["model_kind"]),
        passed=bool(report["passed"]),
        closed_loop_eligible=bool(report.get("closed_loop_eligible", False)),
        steps=int(report["steps"]),
        normalized_mse=float(report["normalized_mse"]),
        max_act_error=float(report["max_act_error"]),
        directory=directory,
        checkpoint=directory / "model.pt",
        report_json=directory / "report.json",
        report_html=directory / "report.html",
    )


def _load_hashed_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stated = report.get("content_sha256")
    body = dict(report)
    body.pop("content_sha256", None)
    if not isinstance(stated, str) or stated != content_sha256(body):
        raise ValueError(f"report content hash mismatch: {report_path}")
    return report


def _validate_prefix_reference(
    report_path: str | Path,
    *,
    manifest: Mapping[str, Any],
    kind: OracleCloneKind,
    config: OracleDistillationConfig,
    source_rows: int,
    row_indices: np.ndarray,
) -> dict[str, Any]:
    report = _load_hashed_report(report_path)
    reference_config = report.get("config", {})
    expected = {
        "manifest_content_sha256": manifest["content_sha256"],
        "collection_digest": manifest["collection_digest"],
        "model_kind": kind.value,
        "source_rows": source_rows,
        "training_row_indices": row_indices.tolist(),
        "hidden_width": config.hidden_width,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"prefix reference {field} mismatch")
    reference_values = {
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "max_steps": 10_000,
        "normalized_mse_threshold": config.normalized_mse_threshold,
        "max_act_error_threshold": config.max_act_error_threshold,
        "baseline_improvement_factor": config.baseline_improvement_factor,
        "hidden_width": config.hidden_width,
        "training_rows": config.training_rows,
    }
    for field, value in reference_values.items():
        if reference_config.get(field) != value:
            raise ValueError(f"prefix reference config {field} mismatch")
    if reference_config.get("lr_schedule", OracleLearningRateSchedule.FIXED.value) != "fixed":
        raise ValueError("prefix reference must use the fixed learning rate")
    if report.get("steps") != 10_000 or not report.get("loss_trace"):
        raise ValueError("prefix reference must contain a complete 10,000-step trace")
    return report


def oracle_learning_rate(schedule: OracleLearningRateSchedule | str, step: int) -> float:
    """Return the immutable learning rate for a one-based optimizer step."""
    schedule = OracleLearningRateSchedule(schedule)
    if step <= 0:
        raise ValueError("optimizer step must be positive")
    if schedule is OracleLearningRateSchedule.FIXED or step <= 10_000:
        return 1e-3
    if step <= 20_000:
        return 1e-4
    return 1e-5


def oracle_recovery_weighted_loss(
    prediction: Any,
    targets: Any,
    *,
    nominal_rows: int,
    recovery_loss_weight: float,
) -> Any:
    """Apply the fixed row-weighted recovery objective to scalar element losses."""
    if prediction.shape != targets.shape or len(prediction.shape) != 2:
        raise ValueError("weighted recovery loss requires matching rank-two tensors")
    total_rows = int(prediction.shape[0])
    recovery_rows = total_rows - nominal_rows
    if nominal_rows <= 0 or recovery_rows < 0:
        raise ValueError("invalid nominal row boundary")
    if not np.isfinite(recovery_loss_weight) or recovery_loss_weight < 0:
        raise ValueError("recovery loss weight must be finite and nonnegative")
    squared_error = (prediction - targets).square()
    if recovery_rows == 0:
        # Keep the established nominal-only optimizer trajectory byte-for-byte.
        return squared_error.mean()
    return (
        squared_error[:nominal_rows].sum()
        + recovery_loss_weight * squared_error[nominal_rows:].sum()
    ) / (nominal_rows + recovery_loss_weight * recovery_rows)


def distill_oracle_policy(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    kind: OracleCloneKind | str,
    config: OracleDistillationConfig | None = None,
    prefix_parity_report: str | Path | None = None,
    recovery_manifest_path: str | Path | None = None,
    observability_manifest_path: str | Path | None = None,
) -> OracleDistillationResult:
    config = config or OracleDistillationConfig()
    kind = OracleCloneKind(kind)
    manifest, arrays = load_oracle_demonstrations(manifest_path)
    source_rows = int(np.asarray(arrays["action_index"]).shape[0])
    row_indices = oracle_training_row_indices(config.training_rows, source_rows)
    schedule = OracleLearningRateSchedule(config.lr_schedule)
    recovery_manifest: dict[str, Any] | None = None
    recovery_arrays: dict[str, np.ndarray] | None = None
    observability_manifest: dict[str, Any] | None = None
    nominal_observability: dict[str, np.ndarray] | None = None
    recovery_observability: dict[str, np.ndarray] | None = None
    observability_kinds = {
        OracleCloneKind.PHASE_DYNAMICS_CONTACT,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
    }
    if observability_manifest_path is not None:
        from so_arm101_v2.simulation.observability import load_observability_annotations

        observability_manifest, nominal_observability, recovery_observability = (
            load_observability_annotations(observability_manifest_path)
        )
        if observability_manifest.get("oracle_manifest_content_sha256") != manifest["content_sha256"]:
            raise ValueError("observability and oracle manifests disagree")
    elif kind in observability_kinds:
        raise ValueError(f"{kind.value} requires an observability manifest")
    if observability_manifest is not None and kind not in observability_kinds:
        raise ValueError("observability manifests are only valid for contact/history clone kinds")
    if recovery_manifest_path is not None:
        recovery_kinds = {
            OracleCloneKind.PHASE_STATE,
            OracleCloneKind.PHASE_DYNAMICS,
            OracleCloneKind.PHASE_DYNAMICS_CONTACT,
            OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
        }
        if kind not in recovery_kinds or config.training_rows != OracleTrainingRows.FULL.value:
            raise ValueError("recovery augmentation requires a phase-aware model with all nominal rows")
        recovery_manifest, recovery_arrays = load_oracle_recovery_examples(recovery_manifest_path)
        if recovery_manifest.get("source_manifest_content_sha256") != manifest["content_sha256"]:
            raise ValueError("recovery examples do not derive from the nominal oracle manifest")
        if (
            observability_manifest is not None
            and observability_manifest.get("recovery_manifest_content_sha256")
            != recovery_manifest["content_sha256"]
        ):
            raise ValueError("observability and recovery manifests disagree")
    prefix_reference: dict[str, Any] | None = None
    if schedule is OracleLearningRateSchedule.DECAY_10K_20K:
        if prefix_parity_report is None:
            if recovery_manifest is None or kind in (
                OracleCloneKind.PHASE_STATE, OracleCloneKind.FEEDBACK_STATE,
            ):
                raise ValueError("decay_10k_20k requires --prefix-parity-report")
        else:
            prefix_reference = _validate_prefix_reference(
                prefix_parity_report,
                manifest=manifest,
                kind=kind,
                config=config,
                source_rows=source_rows,
                row_indices=row_indices,
            )
    elif prefix_parity_report is not None:
        raise ValueError("prefix parity reports are only valid with decay_10k_20k")
    config_identity = asdict(config)
    # Preserve the identities of the established nominal and implicit equal-weight
    # controls. New ablation weights are explicit, content-addressed identities.
    if recovery_manifest is None or config.recovery_loss_weight == 1.0:
        config_identity.pop("recovery_loss_weight")
    identity = {
        "manifest_content_sha256": manifest["content_sha256"],
        "collection_digest": manifest["collection_digest"],
        "model_kind": kind.value,
        "optimizer": "adam_full_batch",
        "source_rows": source_rows,
        "training_row_indices": row_indices.tolist(),
        "config": config_identity,
        "prefix_reference_content_sha256": (
            prefix_reference["content_sha256"] if prefix_reference is not None else None
        ),
        "recovery_manifest_content_sha256": (
            recovery_manifest["content_sha256"] if recovery_manifest is not None else None
        ),
        "recovery_collection_digest": (
            recovery_manifest["collection_digest"] if recovery_manifest is not None else None
        ),
        "recovery_rows": (
            int(recovery_manifest["arrays"]["rows"]) if recovery_manifest is not None else 0
        ),
        "observability_manifest_content_sha256": (
            observability_manifest["content_sha256"]
            if observability_manifest is not None else None
        ),
        "observability_collection_digest": (
            observability_manifest["collection_digest"]
            if observability_manifest is not None else None
        ),
    }
    run_digest = content_sha256(identity)
    directory = Path(output_dir) / "models" / kind.value / run_digest[:16]
    existing = directory / "report.json"
    if existing.is_file():
        existing_report = _load_hashed_report(existing)
        if (
            existing_report.get("run_digest") != run_digest
            or any(existing_report.get(field) != value for field, value in identity.items())
        ):
            raise FileExistsError(f"immutable oracle run identity differs: {directory}")
        if not (directory / "model.pt").is_file() or not (directory / "report.html").is_file():
            raise FileNotFoundError(f"immutable oracle run is missing sidecars: {directory}")
        return _result(directory, existing_report)

    nominal_features, extras_mean, extras_std = build_oracle_features(
        kind, arrays, observability=nominal_observability,
    )
    if int(nominal_features.shape[0]) != source_rows:
        raise ValueError("oracle feature and action row counts disagree")
    features_parts = [nominal_features[row_indices]]
    current_parts = [np.asarray(arrays["current_act"], dtype=np.float32)[row_indices]]
    target_parts = [np.asarray(arrays["executed_act"], dtype=np.float32)[row_indices]]
    maximum_delta = np.asarray(
        load_task_contract("fixed_cube_pickup_v1").safety.maximum_act_delta_per_step,
        dtype=np.float32,
    )
    delta_parts = [np.asarray(arrays["executed_delta_act"], dtype=np.float32)[row_indices]]
    if recovery_arrays is not None:
        recovery_features, _, _ = build_oracle_features(
            kind, recovery_arrays, observability=recovery_observability,
            extras_mean=extras_mean, extras_std=extras_std,
        )
        features_parts.append(recovery_features)
        current_parts.append(np.asarray(recovery_arrays["current_act"], dtype=np.float32))
        target_parts.append(np.asarray(recovery_arrays["executed_act"], dtype=np.float32))
        delta_parts.append(np.asarray(recovery_arrays["executed_delta_act"], dtype=np.float32))
    features_np = np.concatenate(features_parts, axis=0)
    current_act = np.concatenate(current_parts, axis=0)
    targets_act = np.concatenate(target_parts, axis=0)
    target_delta_normalized = np.concatenate(delta_parts, axis=0) / maximum_delta
    baseline_mse = float(np.mean(np.square(target_delta_normalized)))

    torch = _torch()
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    np.random.seed(config.seed)
    features = torch.from_numpy(features_np)
    targets = torch.from_numpy(target_delta_normalized)
    model = build_oracle_clone_model(features_np.shape[1], config.hidden_width)
    with torch.inference_mode():
        initial_output = model(features).numpy()
    if not np.array_equal(initial_output, np.zeros_like(initial_output)):
        raise RuntimeError("oracle residual head did not initialize to hold")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_trace: list[dict[str, float | int]] = []
    normalized_mse = float("inf")
    max_act_error = float("inf")
    predictions_delta = np.zeros_like(target_delta_normalized)
    predictions_act = current_act.copy()
    safety_violations = 0
    passed = False
    steps = 0
    first_gradient_norms: list[float] = []
    prefix_parity_passed: bool | None = None
    nominal_rows = len(row_indices)
    recovery_rows = 0 if recovery_arrays is None else int(recovery_features.shape[0])
    recovery_weight = float(config.recovery_loss_weight if recovery_rows else 0.0)
    nominal_mse = float("inf")
    nominal_max_act_error = float("inf")
    recovery_mse: float | None = None
    recovery_max_act_error: float | None = None
    weighted_objective = float("inf")
    nominal_baseline_mse = float(np.mean(np.square(target_delta_normalized[:nominal_rows])))
    recovery_baseline_mse = (
        None if not recovery_rows
        else float(np.mean(np.square(target_delta_normalized[nominal_rows:])))
    )
    best_observed = {
        "step": 0,
        "normalized_mse": float("inf"),
        "max_act_error": float("inf"),
    }
    for step in range(1, config.max_steps + 1):
        active_learning_rate = oracle_learning_rate(schedule, step)
        for group in optimizer.param_groups:
            group["lr"] = active_learning_rate
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = oracle_recovery_weighted_loss(
            prediction, targets, nominal_rows=nominal_rows,
            recovery_loss_weight=recovery_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("oracle distillation loss became nonfinite")
        loss.backward()
        if step == 1:
            first_gradient_norms = [
                float(parameter.grad.norm().item())
                for parameter in model.parameters() if parameter.grad is not None
            ]
        optimizer.step()
        with torch.inference_mode():
            predictions_delta = model(features).numpy().astype(np.float32)
        residual = predictions_delta - target_delta_normalized
        nominal_mse = float(np.mean(np.square(residual[:nominal_rows])))
        recovery_mse = (
            None if not recovery_rows else float(np.mean(np.square(residual[nominal_rows:])))
        )
        weighted_objective = (
            float(np.mean(np.square(residual))) if not recovery_rows else float(
                (np.square(residual[:nominal_rows]).sum(dtype=np.float64)
                 + recovery_weight * np.square(residual[nominal_rows:]).sum(dtype=np.float64))
                / (nominal_rows + recovery_weight * recovery_rows)
            )
        )
        normalized_mse = nominal_mse
        predictions_act = current_act + predictions_delta * maximum_delta
        nominal_max_act_error = float(
            np.max(np.abs(predictions_act[:nominal_rows] - targets_act[:nominal_rows]))
        )
        recovery_max_act_error = (
            None if not recovery_rows else float(
                np.max(np.abs(predictions_act[nominal_rows:] - targets_act[nominal_rows:]))
            )
        )
        max_act_error = nominal_max_act_error
        steps = step
        if normalized_mse < best_observed["normalized_mse"]:
            best_observed = {
                "step": step,
                "normalized_mse": normalized_mse,
                "max_act_error": max_act_error,
            }
        if step == 1 or step % 100 == 0:
            loss_trace.append({"step": step, "normalized_mse": normalized_mse})
        if step == 10_000 and prefix_reference is not None and recovery_manifest is None:
            prefix_parity_passed = bool(
                loss_trace == prefix_reference["loss_trace"]
                and normalized_mse == prefix_reference["normalized_mse"]
                and max_act_error == prefix_reference["max_act_error"]
            )
            if not prefix_parity_passed:
                raise RuntimeError("scheduled run failed exact 10,000-step prefix parity")
        numerical_gate_passed = bool(
            normalized_mse <= config.normalized_mse_threshold
            and max_act_error <= config.max_act_error_threshold
            and normalized_mse <= nominal_baseline_mse / config.baseline_improvement_factor
        )
        if numerical_gate_passed:
            safety_violations = count_oracle_command_safety_violations(
                current_act[:nominal_rows], predictions_act[:nominal_rows]
            )
            passed = bool(
                safety_violations == 0
                and first_gradient_norms
                and all(np.isfinite(first_gradient_norms))
                and np.all(np.isfinite(predictions_delta))
                and (prefix_reference is None or recovery_manifest is not None or prefix_parity_passed is True)
            )
            if passed:
                break
    if not loss_trace or loss_trace[-1]["step"] != steps:
        loss_trace.append({"step": steps, "normalized_mse": normalized_mse})

    # Always evaluate final predicted commands, even when the numerical gate failed.
    safety_violations = count_oracle_command_safety_violations(
        current_act[:nominal_rows], predictions_act[:nominal_rows]
    )
    numerical_gate_passed = bool(
        normalized_mse <= config.normalized_mse_threshold
        and max_act_error <= config.max_act_error_threshold
        and normalized_mse <= nominal_baseline_mse / config.baseline_improvement_factor
    )
    passed = bool(
        numerical_gate_passed
        and safety_violations == 0
        and first_gradient_norms
        and all(np.isfinite(first_gradient_norms))
        and np.all(np.isfinite(predictions_delta))
        and (prefix_reference is None or recovery_manifest is not None or prefix_parity_passed is True)
    )

    selected_scenario_indices = np.asarray(arrays["scenario_index"], dtype=np.int64)[row_indices]
    row_scenarios = [manifest["episodes"][int(index)]["scenario_id"] for index in selected_scenario_indices]
    if recovery_manifest is not None:
        row_scenarios.extend(f"recovery:{item['name']}" for item in recovery_manifest["records"])
    absolute_error = np.abs(predictions_act - targets_act)
    worst_flat = int(np.argmax(absolute_error))
    worst_row, worst_joint = np.unravel_index(worst_flat, absolute_error.shape)
    if worst_row < len(row_indices):
        worst_source: int | str = int(row_indices[worst_row])
    else:
        assert recovery_manifest is not None
        worst_source = f"recovery:{recovery_manifest['records'][worst_row - len(row_indices)]['name']}"
    report: dict[str, Any] = {
        "schema_version": 1,
        **identity,
        "run_digest": run_digest,
        "rows": int(features_np.shape[0]),
        "source_rows": source_rows,
        "training_row_mode": config.training_rows,
        "training_row_indices": row_indices.tolist(),
        "input_dim": int(features_np.shape[1]),
        "hidden_width": config.hidden_width,
        "learning_rate_schedule": schedule.value,
        "learning_rate_boundaries": (
            [{"first_step": 1, "learning_rate": 1e-3}]
            if schedule is OracleLearningRateSchedule.FIXED
            else [
                {"first_step": 1, "learning_rate": 1e-3},
                {"first_step": 10_001, "learning_rate": 1e-4},
                {"first_step": 20_001, "learning_rate": 1e-5},
            ]
        ),
        "active_learning_rate_at_end": oracle_learning_rate(schedule, steps),
        "best_observed": best_observed,
        "prefix_parity": {
            "required": prefix_reference is not None and recovery_manifest is None,
            "applicable": recovery_manifest is None,
            "passed": prefix_parity_passed,
            "reference_content_sha256": (
                prefix_reference["content_sha256"] if prefix_reference is not None else None
            ),
            "validated_step": 10_000 if prefix_parity_passed is True else None,
            "reason": (
                "not comparable because the recovery rows intentionally change the training objective"
                if recovery_manifest is not None else None
            ),
        },
        "architecture": [
            f"linear_{features_np.shape[1]}_{config.hidden_width}", "relu",
            f"linear_{config.hidden_width}_{config.hidden_width}", "relu",
            f"linear_{config.hidden_width}_6_zero_init",
        ],
        "feature_schema": list(oracle_feature_schema(kind)),
        "history_length": (
            2 if kind is OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2 else 1
        ),
        "history_padding": (
            "repeat_current_at_episode_reset"
            if kind is OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2 else None
        ),
        "contact_timing": (
            "pre_action_before_current_command_selection"
            if kind in observability_kinds else None
        ),
        "target": "executed_act_minus_current_act_divided_by_maximum_act_delta_per_step",
        "normalization": {
            "extras_mean": extras_mean.tolist(),
            "extras_std": extras_std.tolist(),
            "std_floor": 1e-6,
        },
        "steps": steps,
        "passed": passed,
        "closed_loop_eligible": bool(
            passed
            and config.training_rows == OracleTrainingRows.FULL.value
            and (prefix_reference is None or recovery_manifest is not None or prefix_parity_passed is True)
        ),
        "numerical_gate_passed": numerical_gate_passed,
        "normalized_mse": normalized_mse,
        "max_act_error": max_act_error,
        "cumulative_absolute_act_error_per_joint": np.sum(absolute_error, axis=0, dtype=np.float64).tolist(),
        "worst_error": {
            "row": worst_source,
            "training_position": int(worst_row),
            "joint": JOINT_NAMES[int(worst_joint)],
            "absolute_act_error": float(absolute_error[worst_row, worst_joint]),
        },
        "zero_delta_baseline_normalized_mse": baseline_mse,
        "nominal_metrics": {
            "rows": nominal_rows,
            "unweighted_normalized_mse": nominal_mse,
            "maximum_act_error": nominal_max_act_error,
            "zero_delta_baseline_normalized_mse": nominal_baseline_mse,
            "zero_delta_improvement_factor": nominal_baseline_mse / nominal_mse,
            "predicted_command_safety_violations": safety_violations,
        },
        "recovery_metrics": None if not recovery_rows else {
            "rows": recovery_rows,
            "loss_weight": recovery_weight,
            "unweighted_normalized_mse": recovery_mse,
            "maximum_act_error": recovery_max_act_error,
            "zero_delta_baseline_normalized_mse": recovery_baseline_mse,
        },
        "weighted_objective": {
            "value": weighted_objective,
            "formula": "(sum_nominal_element_squared_errors + w * sum_recovery_element_squared_errors) / (450 + 8w)",
            "recovery_loss_weight": recovery_weight,
            "denominator": nominal_rows + recovery_weight * recovery_rows,
        },
        "training_safety_violations": safety_violations,
        "gradient_norms_at_step_1": first_gradient_norms,
        "loss_trace": loss_trace,
        "episodes": manifest["episodes"],
        "row_scenarios": row_scenarios,
        "recovery_augmentation": None if recovery_manifest is None else {
            "manifest_content_sha256": recovery_manifest["content_sha256"],
            "collection_digest": recovery_manifest["collection_digest"],
            "rows": int(recovery_manifest["arrays"]["rows"]),
            "anchors": [item["name"] for item in recovery_manifest["records"]],
            "normalization": "unchanged_nominal_statistics",
        },
        "observability_annotations": None if observability_manifest is None else {
            "manifest_content_sha256": observability_manifest["content_sha256"],
            "collection_digest": observability_manifest["collection_digest"],
            "contact_fields": observability_manifest["contact_fields"],
            "history_length": observability_manifest["history_length"],
            "history_padding": observability_manifest["history_padding"],
        },
        "joint_order": list(JOINT_NAMES),
        "claim": "privileged_pipeline_memorization_only_not_closed_loop_success",
    }
    report["content_sha256"] = content_sha256(report)
    checkpoint_payload = {
        "schema_version": 1,
        **identity,
        "input_dim": int(features_np.shape[1]),
        "hidden_width": config.hidden_width,
        "training_row_mode": config.training_rows,
        "training_row_indices": row_indices.tolist(),
        "source_rows": source_rows,
        "closed_loop_eligible": report["closed_loop_eligible"],
        "extras_mean": torch.from_numpy(extras_mean.copy()),
        "extras_std": torch.from_numpy(extras_std.copy()),
        "maximum_act_delta_per_step": torch.from_numpy(maximum_delta.copy()),
        "teacher_horizon": int(manifest["teacher_horizon"]),
        "state_dict": model.state_dict(),
        "report_content_sha256": report["content_sha256"],
        "recovery_augmentation": report["recovery_augmentation"],
        "feature_schema": report["feature_schema"],
        "history_length": report["history_length"],
        "history_padding": report["history_padding"],
        "contact_timing": report["contact_timing"],
        "observability_annotations": report["observability_annotations"],
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint_payload, checkpoint_buffer)
    _write_immutable_bytes(directory / "model.pt", checkpoint_buffer.getvalue())
    write_immutable_json(directory / "report.json", report)
    _write_immutable_bytes(
        directory / "report.html",
        _diagnostic_html(report, targets_act, predictions_act).encode("utf-8"),
    )
    return _result(directory, report)


def _aggregate_residual_rows(
    positions: np.ndarray,
    normalized_error: np.ndarray,
    act_error: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(positions, dtype=np.int64)
    if values.size == 0:
        return {"rows": 0}
    normalized = normalized_error[values]
    raw = act_error[values]
    return {
        "rows": int(values.size),
        "normalized_mse": float(np.mean(np.square(normalized))),
        "maximum_act_error": float(np.max(np.abs(raw))),
        "mean_row_normalized_mse": float(
            np.mean(np.mean(np.square(normalized), axis=1))
        ),
        "training_positions": values.tolist(),
    }


def _residual_analysis_html(report: Mapping[str, Any]) -> str:
    stage_rows = "".join(
        "<tr>"
        f"<td>{item['stage']}</td><td>{item['start']}</td><td>{item['end_exclusive']}</td>"
        f"<td>{item['normalized_mse']:.8g}</td><td>{item['maximum_act_error']:.8g}</td>"
        "</tr>"
        for item in report["per_stage"]
    )
    joint_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['joint'])}</td><td>{item['normalized_mse']:.8g}</td>"
        f"<td>{item['maximum_act_error']:.8g}</td><td>{item['cumulative_absolute_act_error']:.8g}</td>"
        "</tr>"
        for item in report["per_joint"]
    )
    worst_rows = "".join(
        "<tr>"
        f"<td>{item['source_row']}</td><td>{item['stage']}</td>"
        f"<td>{item['normalized_mse']:.8g}</td><td>{item['maximum_act_error']:.8g}</td>"
        "</tr>"
        for item in sorted(
            report["per_row"], key=lambda item: item["normalized_mse"], reverse=True
        )[:20]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Oracle residual analysis</title>
<style>body{{font-family:system-ui;background:#10151c;color:#e7edf5;margin:24px}}table{{border-collapse:collapse;margin-bottom:24px}}td,th{{border:1px solid #445;padding:5px;font-family:monospace}}</style></head>
<body><h1>Oracle residual analysis</h1>
<p>Checkpoint <code>{html.escape(report['checkpoint_sha256'])}</code><br>Training report <code>{html.escape(report['training_report_content_sha256'])}</code><br>Manifest <code>{html.escape(report['manifest_content_sha256'])}</code></p>
<p>Possible label conflicts: {len(report['possible_label_conflicts'])}. A conflict requires feature L2 distance ≤ {report['conflict_rule']['maximum_feature_l2_distance']} and raw target-delta L∞ difference ≥ {report['conflict_rule']['minimum_target_delta_linf_difference']}.</p>
<h2>Stages</h2><table><tr><th>stage</th><th>start</th><th>end</th><th>normalized MSE</th><th>max ACT error</th></tr>{stage_rows}</table>
<h2>Joints</h2><table><tr><th>joint</th><th>normalized MSE</th><th>max ACT error</th><th>cumulative abs error</th></tr>{joint_rows}</table>
<h2>Worst rows</h2><table><tr><th>source row</th><th>stage</th><th>normalized MSE</th><th>max ACT error</th></tr>{worst_rows}</table>
<p>This is immutable offline diagnostic evidence, not closed-loop eligibility.</p></body></html>"""


def analyze_oracle_residuals(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> OracleResidualAnalysisResult:
    """Hash-validate and analyze an oracle clone without requiring promotion."""
    analysis_version = "oracle_residual_v2"
    checkpoint_path = Path(checkpoint_path)
    report = _load_hashed_report(checkpoint_path.with_name("report.json"))
    manifest, arrays = load_oracle_demonstrations(manifest_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    torch = _torch()
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported oracle clone checkpoint schema")
    if payload.get("report_content_sha256") != report["content_sha256"]:
        raise ValueError("checkpoint and training report hash mismatch")
    if (
        payload.get("manifest_content_sha256") != manifest["content_sha256"]
        or report.get("manifest_content_sha256") != manifest["content_sha256"]
    ):
        raise ValueError("checkpoint, report, and manifest hashes disagree")
    metadata_fields = ("model_kind", "training_row_indices", "source_rows")
    if any(payload.get(field) != report.get(field) for field in metadata_fields):
        raise ValueError("checkpoint and training report metadata disagree")
    hidden_width = int(payload.get("hidden_width", 128))
    if hidden_width != int(report.get("hidden_width", 128)):
        raise ValueError("checkpoint and report hidden widths disagree")

    kind = OracleCloneKind(payload["model_kind"])
    row_indices = np.asarray(payload["training_row_indices"], dtype=np.int64)
    features_all, _, _ = build_oracle_features(
        kind,
        arrays,
        extras_mean=np.asarray(payload["extras_mean"], dtype=np.float32),
        extras_std=np.asarray(payload["extras_std"], dtype=np.float32),
    )
    if int(payload["source_rows"]) != len(features_all):
        raise ValueError("checkpoint source-row count mismatch")
    features = features_all[row_indices]
    maximum_delta = np.asarray(payload["maximum_act_delta_per_step"], dtype=np.float32)
    current_act = np.asarray(arrays["current_act"], dtype=np.float32)[row_indices]
    targets_act = np.asarray(arrays["executed_act"], dtype=np.float32)[row_indices]
    target_delta_act = np.asarray(arrays["executed_delta_act"], dtype=np.float32)[row_indices]
    targets_normalized = target_delta_act / maximum_delta
    model = build_oracle_clone_model(int(payload["input_dim"]), hidden_width)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.inference_mode():
        prediction_normalized = model(torch.from_numpy(features)).numpy().astype(np.float32)
    predictions_act = current_act + prediction_normalized * maximum_delta
    normalized_error = prediction_normalized - targets_normalized
    act_error = predictions_act - targets_act

    boundaries: list[int] = []
    offset = 0
    for episode in manifest["episodes"]:
        boundaries.extend(offset + int(value) for value in episode["waypoint_boundaries"])
        offset += int(episode["rows"])
    boundaries = sorted(set(boundaries))
    stage_for_source: dict[int, int] = {}
    per_stage: list[dict[str, Any]] = []
    for stage, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        positions = np.flatnonzero((row_indices >= start) & (row_indices < end))
        stage_for_source.update((int(row_indices[position]), stage) for position in positions)
        per_stage.append({
            "stage": stage,
            "start": start,
            "end_exclusive": end,
            **_aggregate_residual_rows(positions, normalized_error, act_error),
        })

    per_row = [
        {
            "source_row": int(source_row),
            "training_position": position,
            "stage": stage_for_source.get(int(source_row)),
            "normalized_mse": float(np.mean(np.square(normalized_error[position]))),
            "maximum_act_error": float(np.max(np.abs(act_error[position]))),
            "normalized_error": normalized_error[position].tolist(),
            "act_error": act_error[position].tolist(),
        }
        for position, source_row in enumerate(row_indices)
    ]
    per_joint = [
        {
            "joint": joint,
            "normalized_mse": float(np.mean(np.square(normalized_error[:, index]))),
            "maximum_act_error": float(np.max(np.abs(act_error[:, index]))),
            "cumulative_absolute_act_error": float(
                np.sum(np.abs(act_error[:, index]), dtype=np.float64)
            ),
        }
        for index, joint in enumerate(JOINT_NAMES)
    ]
    boundary_windows = []
    for boundary in boundaries[1:-1]:
        positions = np.flatnonzero(
            (row_indices >= max(0, boundary - 5)) & (row_indices <= boundary + 5)
        )
        boundary_windows.append({
            "boundary": boundary,
            "radius_rows": 5,
            **_aggregate_residual_rows(positions, normalized_error, act_error),
        })

    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest_position = np.argmin(distances, axis=1)
    nearest_feature = []
    for position, neighbor in enumerate(nearest_position):
        nearest_feature.append({
            "source_row": int(row_indices[position]),
            "nearest_source_row": int(row_indices[neighbor]),
            "feature_l2_distance": float(distances[position, neighbor]),
            "target_delta_linf_difference": float(
                np.max(np.abs(target_delta_act[position] - target_delta_act[neighbor]))
            ),
        })
    upper = np.triu_indices(len(features), k=1)
    pair_distances = distances[upper]
    pair_target_differences = np.max(
        np.abs(target_delta_act[upper[0]] - target_delta_act[upper[1]]), axis=1
    )
    conflict_mask = (pair_distances <= 0.005) & (pair_target_differences >= 0.01)
    possible_conflicts = [
        {
            "source_row_a": int(row_indices[upper[0][index]]),
            "source_row_b": int(row_indices[upper[1][index]]),
            "feature_l2_distance": float(pair_distances[index]),
            "target_delta_linf_difference": float(pair_target_differences[index]),
        }
        for index in np.flatnonzero(conflict_mask)
    ]
    target_difference_statistics = {
        "nearest_pair_min": float(min(item["target_delta_linf_difference"] for item in nearest_feature)),
        "nearest_pair_mean": float(np.mean([item["target_delta_linf_difference"] for item in nearest_feature])),
        "nearest_pair_max": float(max(item["target_delta_linf_difference"] for item in nearest_feature)),
    }
    identity = {
        "analysis_version": analysis_version,
        "checkpoint_sha256": checkpoint_sha256,
        "training_report_content_sha256": report["content_sha256"],
        "manifest_content_sha256": manifest["content_sha256"],
        "conflict_rule": {
            "maximum_feature_l2_distance": 0.005,
            "minimum_target_delta_linf_difference": 0.01,
        },
    }
    analysis_digest = content_sha256(identity)
    directory = Path(output_dir) / "analyses" / kind.value / analysis_digest[:16]
    analysis: dict[str, Any] = {
        "schema_version": 1,
        **identity,
        "analysis_digest": analysis_digest,
        "model_kind": kind.value,
        "hidden_width": hidden_width,
        "rows": int(len(row_indices)),
        "source_rows": int(len(features_all)),
        "training_row_mode": report["training_row_mode"],
        "training_row_indices": row_indices.tolist(),
        "overall": _aggregate_residual_rows(
            np.arange(len(row_indices)), normalized_error, act_error
        ),
        "per_row": per_row,
        "per_stage": per_stage,
        "per_joint": per_joint,
        "boundary_windows": boundary_windows,
        "nearest_feature": nearest_feature,
        "target_difference_statistics": target_difference_statistics,
        "possible_label_conflicts": possible_conflicts,
        "claim": "immutable_offline_residual_diagnosis_not_promotion",
    }
    analysis["content_sha256"] = content_sha256(analysis)
    write_immutable_json(directory / "analysis.json", analysis)
    _write_immutable_bytes(
        directory / "analysis.html", _residual_analysis_html(analysis).encode("utf-8")
    )
    return OracleResidualAnalysisResult(
        directory=directory,
        report_json=directory / "analysis.json",
        report_html=directory / "analysis.html",
    )


__all__ = [
    "OracleCloneKind",
    "OracleTrainingRows",
    "OracleLearningRateSchedule",
    "ORACLE_MEMORIZATION32_ROWS",
    "OracleDistillationConfig",
    "OracleDistillationResult",
    "OracleResidualAnalysisResult",
    "build_oracle_clone_model",
    "build_oracle_features",
    "oracle_feature_schema",
    "count_oracle_command_safety_violations",
    "oracle_training_row_indices",
    "oracle_learning_rate",
    "oracle_recovery_weighted_loss",
    "distill_oracle_policy",
    "analyze_oracle_residuals",
]
