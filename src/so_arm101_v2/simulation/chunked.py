"""Closed-loop execution and promotion gate for chunked-action clones."""

from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.learning.numerics import (
    AUTO,
    NumericsSpec,
    numerics_identity,
    resolve_default_numerics,
)

from .policy_specs import PolicySpec

from .observability import _evaluation_passed, _load_hashed_json, _rollout_summary
from .oracle import load_oracle_demonstrations
from .suites import load_simulation_suite

CHUNK_HORIZON_LADDER = (10, 30, 90)


@dataclass(frozen=True)
class ChunkedGateResult:
    directory: Path
    report_json: Path
    status: str


class ChunkedClonePolicy:
    """Execute a chunked clone: one privileged observation per H actions."""

    requires_pixels = False

    def __init__(self, checkpoint: str | Path) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("chunked inference requires the 'learn' extra") from exc
        from so_arm101_v2.learning.chunked import build_chunked_clone_model
        from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act

        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported chunked clone checkpoint schema")
        self.chunk_horizon = int(payload["chunk_horizon"])
        if payload.get("model_kind") != f"chunked_h{self.chunk_horizon}":
            raise ValueError("chunked clone checkpoint kind metadata is malformed")
        self.input_dim = int(payload["input_dim"])
        self.hidden_width = int(payload["hidden_width"])
        self.teacher_horizon = int(payload["teacher_horizon"])
        self.config_seed = int(payload.get("config", {}).get("seed", 101))
        if self.input_dim != 10 or self.teacher_horizon not in (450, 480):
            raise ValueError("chunked clone checkpoint metadata is malformed")
        report = _load_hashed_json(
            checkpoint_path.with_name("report.json"), label="chunked clone report"
        )
        if (
            report["content_sha256"] != payload.get("report_content_sha256")
            or report.get("manifest_content_sha256") != payload.get("manifest_content_sha256")
            or report.get("run_digest") != payload.get("run_digest")
        ):
            raise ValueError("chunked clone checkpoint and report disagree")
        self.report_content_sha256 = report["content_sha256"]
        self.manifest_content_sha256 = payload["manifest_content_sha256"]
        self.extras_mean = np.asarray(payload["extras_mean"], dtype=np.float32)
        self.extras_std = np.asarray(payload["extras_std"], dtype=np.float32)
        self.saturation_mode = str(payload.get("saturation_mode", "none"))
        self._has_saturation_key = "saturation_mode" in payload
        self.decoder_eta = payload.get("decoder_eta")
        self.margin_act = payload.get("margin_act")
        if self.saturation_mode == "feasible_chain_v1" and (
            self.decoder_eta is None or self.margin_act is None
        ):
            raise ValueError("feasible-chain checkpoint is missing decoder parameters")
        self.model = build_chunked_clone_model(
            self.input_dim, self.hidden_width, self.chunk_horizon
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.torch = torch
        self._normalize_act = normalize_act
        self._denormalize_act = denormalize_act
        self.action_index = 0
        self._buffer: list[np.ndarray] = []

    @property
    def policy_id(self) -> str:
        base = f"chunked_h{self.chunk_horizon}.seed{self.config_seed}"
        if self._has_saturation_key:
            base = f"{base}.{self.saturation_mode}"
        return base

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._buffer = []

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del image
        from so_arm101_v2.learning.oracle_distillation import (
            OracleCloneKind,
            build_oracle_features,
        )

        if not self._buffer:
            if adapter is None:
                raise ValueError("chunked clone policy requires a MuJoCo adapter")
            snapshot = adapter.privileged_state()
            if np.max(np.abs(snapshot.current_act - np.asarray(current_act, dtype=np.float32))) > 1e-5:
                raise RuntimeError("adapter state changed during chunked clone observation")
            arrays = {
                "current_act": snapshot.current_act[None],
                "cube_position": snapshot.cube_position[None],
                "progress": np.asarray([
                    min(self.action_index, self.teacher_horizon - 1)
                    / (self.teacher_horizon - 1)
                ], dtype=np.float32),
            }
            features, _, _ = build_oracle_features(
                OracleCloneKind.PHASE_STATE, arrays,
                extras_mean=self.extras_mean, extras_std=self.extras_std,
            )
            current_norm = self._normalize_act(snapshot.current_act)
            with self.torch.inference_mode():
                raw = self.model(self.torch.from_numpy(features))
                if self.saturation_mode == "feasible_chain_v1":
                    from so_arm101_v2.learning.chunked import decode_feasible_chunk

                    chunk_norm = decode_feasible_chunk(
                        raw.view(1, self.chunk_horizon, 6),
                        self.torch.from_numpy(current_norm[None, :]),
                        eta=float(self.decoder_eta),
                        margin_act=float(self.margin_act),
                    ).numpy()[0]
                else:
                    residual = raw.numpy()[0]
                    chunk_norm = current_norm[None, :] + residual.reshape(self.chunk_horizon, 6)
            commands = self._denormalize_act(chunk_norm.astype(np.float32))
            self._buffer = [np.asarray(row, dtype=np.float32) for row in commands]
        self.action_index += 1
        command = self._buffer.pop(0)
        return command


def resolve_chunked_gate_status(candidates: list[Mapping[str, Any]]) -> str:
    """Validate the ascending-ladder branch trace and derive the status."""
    if len(candidates) != len(CHUNK_HORIZON_LADDER):
        raise ValueError("chunked gate requires one record per ladder horizon")
    passed_horizon: int | None = None
    for candidate, horizon in zip(candidates, CHUNK_HORIZON_LADDER):
        if int(candidate.get("chunk_horizon", -1)) != horizon:
            raise ValueError("chunked candidate order does not match the ladder")
        state = candidate.get("state")
        if passed_horizon is not None:
            if state != "skipped_after_first_pass":
                raise ValueError("candidate ran after the first nominal pass")
            continue
        if state not in ("nominal_passed", "nominal_failed"):
            raise ValueError(f"invalid chunked candidate state {state!r}")
        if candidate.get("evaluation") is None:
            raise ValueError("evaluated chunked candidate lacks evaluation evidence")
        if state == "nominal_passed":
            passed_horizon = horizon
    if passed_horizon is not None:
        return f"passed_h{passed_horizon}"
    return "closed_loop_not_resolved"


def run_chunked_gate(
    model_path: str | Path,
    oracle_manifest_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
    numerics: NumericsSpec | None | str = AUTO,
) -> ChunkedGateResult:
    """Train the ascending chunk-horizon ladder and stop at the first safe pass."""
    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
    from .rollout import evaluate_closed_loop

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if (
        oracle_manifest.get("scenario_ids") != ["nominal"]
        or int(oracle_manifest.get("teacher_horizon", 0)) != 450
    ):
        raise ValueError("chunked gate requires the canonical nominal 450-row oracle capture")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("chunked gate requires the passing deterministic v3 preflight")

    identity = {
        "schema_version": 1,
        "experiment": "chunked_promotion_gate_v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "candidate_order": list(CHUNK_HORIZON_LADDER),
        "fixed_training_config": {
            "seed": 101,
            "hidden_width": 256,
            "learning_rate": 1e-3,
            "max_steps": 30_000,
            "data": "nominal_450_rows_only",
            "target": "normalized_absolute_act_residual_on_current_pose",
            "chunk_padding": "repeat_final_command",
        },
        "offline_role": "telemetry_only_promotion_by_closed_loop",
        "pass_rule": "three deterministic nominal successes with zero safety counts",
    }
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    gate_digest = content_sha256(identity)
    directory = output_dir / "chunked_gates" / gate_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="chunked gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable chunked gate differs: {directory}")
        return ChunkedGateResult(directory, report_path, str(existing["status"]))

    suite = load_simulation_suite("fixed_pick_place_v3")
    nominal_suite = replace(
        suite,
        suite_id="fixed_pick_place_v3.nominal_chunked_gate",
        scenarios=tuple(item for item in suite.scenarios if item.scenario_id == "nominal"),
        repeats=3,
    )
    candidates: list[dict[str, Any]] = []
    terminal: str | None = None
    for horizon in CHUNK_HORIZON_LADDER:
        if terminal is not None:
            candidates.append({
                "chunk_horizon": horizon, "state": "skipped_after_first_pass",
            })
            continue
        training = train_chunked_clone(
            oracle_manifest_path,
            output_dir,
            config=ChunkedCloneConfig(chunk_horizon=horizon),
            numerics=numerics,
        )
        training_report = _load_hashed_json(
            training.report_json, label="chunked training report"
        )
        probe = ChunkedClonePolicy(training.checkpoint)
        evaluation_identity = content_sha256({
            "gate_digest": gate_digest,
            "chunk_horizon": horizon,
            "checkpoint_sha256": hashlib.sha256(training.checkpoint.read_bytes()).hexdigest(),
            "training_report_content_sha256": training_report["content_sha256"],
            "preflight_report_content_sha256": preflight["content_sha256"],
        })
        evaluation = evaluate_closed_loop(
            model_path,
            nominal_suite,
            {probe.policy_id: PolicySpec(kind="chunked_clone", checkpoint=str(training.checkpoint.resolve()))},
            output_dir / "chunked_gate_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "gate_digest": gate_digest,
                "chunk_horizon": horizon,
                "checkpoint_sha256": hashlib.sha256(training.checkpoint.read_bytes()).hexdigest(),
                "offline_report_content_sha256": training_report["content_sha256"],
                "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
                "preflight_report_content_sha256": preflight["content_sha256"],
            },
        )
        evaluation_report = _load_hashed_json(
            evaluation.report_json, label="chunked evaluation report"
        )
        passed = _evaluation_passed(evaluation_report)
        candidates.append({
            "chunk_horizon": horizon,
            "state": "nominal_passed" if passed else "nominal_failed",
            "checkpoint_sha256": hashlib.sha256(training.checkpoint.read_bytes()).hexdigest(),
            "training_report": str(training.report_json.resolve()),
            "training_report_content_sha256": training_report["content_sha256"],
            "offline_telemetry": {
                "normalized_mse": float(training.normalized_mse),
                "max_act_error": float(training.max_act_error),
                "steps": int(training.steps),
            },
            "evaluation": {
                "report": str(evaluation.report_json.resolve()),
                "content_sha256": evaluation_report["content_sha256"],
                "passed": passed,
                "deterministic": bool(evaluation_report["deterministic"]),
                "rollouts": [_rollout_summary(item) for item in evaluation_report["rollouts"]],
            },
        })
        if passed:
            terminal = f"passed_h{horizon}"

    status = resolve_chunked_gate_status(candidates)
    if terminal is not None and terminal != status:
        raise RuntimeError("chunked gate branch trace disagrees with terminal status")
    next_actions = {
        "passed_h10": "broader_evaluation_of_passing_chunked_policy_only",
        "passed_h30": "broader_evaluation_of_passing_chunked_policy_only",
        "passed_h90": "broader_evaluation_of_passing_chunked_policy_only",
        "closed_loop_not_resolved": (
            "stop_and_write_new_proposal_correction_chunks_or_capacity_or_ensembling"
        ),
    }
    report = {
        **identity,
        "gate_digest": gate_digest,
        "status": status,
        "next_action": next_actions[status],
        "candidates": candidates,
        "training_runs_started": sum(
            item["state"] != "skipped_after_first_pass" for item in candidates
        ),
        "nominal_evaluations_started": sum(
            item.get("evaluation") is not None for item in candidates
        ),
        "partial_milestones_are_non_promoting": True,
        "broader_evaluation_run": False,
        "claim": "chunk_horizon_ladder_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return ChunkedGateResult(directory, report_path, status)


# Promotion-preference order for the saturation-attribution gate: the
# decoder family first (hard feasibility guarantee), then the penalty family
# (soft), then unmitigated corrections; within a family, single-ingredient
# before combined.  All candidates run; the earliest passer promotes.
SATURATION_CANDIDATE_ORDER = (
    "feasible_decoder_only",
    "corrections_and_feasible_decoder",
    "noise_penalty_only",
    "corrections_and_noise_penalty",
    "corrections_only",
)
_SATURATION_CANDIDATE_SPECS = {
    "feasible_decoder_only": ("feasible_chain_v1", False),
    "corrections_and_feasible_decoder": ("feasible_chain_v1", True),
    "noise_penalty_only": ("noise_penalty_v1", False),
    "corrections_and_noise_penalty": ("noise_penalty_v1", True),
    "corrections_only": ("none", True),
}
_SATURATION_FIXED = {
    "chunk_horizon": 90,
    "seed": 101,
    "hidden_width": 256,
    "learning_rate": 1e-3,
    "max_steps": 30_000,
    "decoder_eta": 0.9,
    "margin_act": 0.002,
    "noise_sigma": 0.05,
    "penalty_weight": 1.0,
}


def _saturation_candidate_config(mode: str) -> Any:
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig

    fixed = _SATURATION_FIXED
    base = {
        "chunk_horizon": fixed["chunk_horizon"],
        "seed": fixed["seed"],
        "hidden_width": fixed["hidden_width"],
        "learning_rate": fixed["learning_rate"],
        "max_steps": fixed["max_steps"],
    }
    if mode == "none":
        return ChunkedCloneConfig(**base)
    base.update({
        "saturation_mode": mode,
        "decoder_eta": fixed["decoder_eta"],
        "margin_act": fixed["margin_act"],
    })
    if mode == "noise_penalty_v1":
        base.update({
            "noise_sigma": fixed["noise_sigma"],
            "penalty_weight": fixed["penalty_weight"],
        })
    return ChunkedCloneConfig(**base)


def resolve_saturation_gate_status(candidates: list[Mapping[str, Any]]) -> str:
    """Validate the all-candidates branch trace and derive the status."""
    if len(candidates) != len(SATURATION_CANDIDATE_ORDER):
        raise ValueError("saturation gate requires one record per candidate")
    promoted: str | None = None
    for candidate, expected_id in zip(candidates, SATURATION_CANDIDATE_ORDER):
        if candidate.get("candidate_id") != expected_id:
            raise ValueError("saturation candidate order does not match the registry")
        state = candidate.get("state")
        if state not in ("nominal_passed", "nominal_failed"):
            raise ValueError(f"invalid saturation candidate state {state!r}")
        if candidate.get("evaluation") is None:
            raise ValueError("saturation candidate lacks evaluation evidence")
        if state == "nominal_passed" and promoted is None:
            promoted = expected_id
    if promoted is not None:
        return f"promoted_{promoted}"
    return "closed_loop_not_resolved"


@dataclass(frozen=True)
class _SaturationCandidateTask:
    candidate_id: str
    model_path: str
    oracle_manifest_path: str
    correction_manifest_path: str
    output_dir: str
    gate_digest: str
    oracle_manifest_content_sha256: str
    correction_manifest_content_sha256: str
    preflight_content_sha256: str
    record_video: bool
    eval_workers: int
    legacy_numerics: bool = False


def _saturation_nominal_suite() -> Any:
    suite = load_simulation_suite("fixed_pick_place_v3")
    return replace(
        suite,
        suite_id="fixed_pick_place_v3.nominal_saturation_gate",
        scenarios=tuple(item for item in suite.scenarios if item.scenario_id == "nominal"),
        repeats=3,
    )


def _run_saturation_candidate(task: _SaturationCandidateTask) -> dict[str, Any]:
    """Train and evaluate one saturation candidate; safe to run in a worker process.

    ``train_chunked_clone`` seeds and pins torch to one thread per call, and all
    artifacts land in digest-keyed directories, so results are byte-identical
    whether this runs in-process or in a spawn worker.
    """
    from so_arm101_v2.learning.chunked import train_chunked_clone

    from .rollout import evaluate_closed_loop

    output_dir = Path(task.output_dir)
    mode, use_corrections = _SATURATION_CANDIDATE_SPECS[task.candidate_id]
    nominal_suite = _saturation_nominal_suite()
    training = train_chunked_clone(
        task.oracle_manifest_path,
        output_dir,
        config=_saturation_candidate_config(mode),
        correction_manifest_path=(
            task.correction_manifest_path if use_corrections else None
        ),
        numerics=None if task.legacy_numerics else AUTO,
    )
    training_report = _load_hashed_json(
        training.report_json, label="saturation training report"
    )
    probe = ChunkedClonePolicy(training.checkpoint)
    checkpoint_sha = hashlib.sha256(training.checkpoint.read_bytes()).hexdigest()
    evaluation_identity = content_sha256({
        "gate_digest": task.gate_digest,
        "candidate_id": task.candidate_id,
        "checkpoint_sha256": checkpoint_sha,
        "training_report_content_sha256": training_report["content_sha256"],
        "preflight_report_content_sha256": task.preflight_content_sha256,
    })
    evaluation = evaluate_closed_loop(
        task.model_path,
        nominal_suite,
        {probe.policy_id: PolicySpec(kind="chunked_clone", checkpoint=str(training.checkpoint.resolve()))},
        output_dir / "saturation_gate_evaluations" / evaluation_identity[:16],
        environment_proven=True,
        record_video=task.record_video,
        workers=task.eval_workers,
        provenance={
            "gate_digest": task.gate_digest,
            "candidate_id": task.candidate_id,
            "checkpoint_sha256": checkpoint_sha,
            "offline_report_content_sha256": training_report["content_sha256"],
            "oracle_manifest_content_sha256": task.oracle_manifest_content_sha256,
            "correction_manifest_content_sha256": task.correction_manifest_content_sha256,
            "preflight_report_content_sha256": task.preflight_content_sha256,
        },
    )
    evaluation_report = _load_hashed_json(
        evaluation.report_json, label="saturation evaluation report"
    )
    passed = _evaluation_passed(evaluation_report)
    return {
        "candidate_id": task.candidate_id,
        "saturation_mode": mode,
        "corrections": use_corrections,
        "state": "nominal_passed" if passed else "nominal_failed",
        "checkpoint_sha256": checkpoint_sha,
        "training_report": str(training.report_json.resolve()),
        "training_report_content_sha256": training_report["content_sha256"],
        "offline_telemetry": {
            "normalized_mse": float(training.normalized_mse),
            "max_act_error": float(training.max_act_error),
            "steps": int(training.steps),
            "saturation": training_report.get("saturation"),
        },
        "evaluation": {
            "report": str(evaluation.report_json.resolve()),
            "content_sha256": evaluation_report["content_sha256"],
            "passed": passed,
            "deterministic": bool(evaluation_report["deterministic"]),
            "rollouts": [_rollout_summary(item) for item in evaluation_report["rollouts"]],
        },
    }


def run_saturation_gate(
    model_path: str | Path,
    oracle_manifest_path: str | Path,
    correction_manifest_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
    numerics: NumericsSpec | None | str = AUTO,
) -> ChunkedGateResult:
    """Train and evaluate all five saturation-attribution candidates."""
    from .correction import load_oracle_corrections

    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if (
        oracle_manifest.get("scenario_ids") != ["nominal"]
        or int(oracle_manifest.get("teacher_horizon", 0)) != 450
    ):
        raise ValueError("saturation gate requires the canonical nominal 450-row oracle capture")
    correction_manifest, _ = load_oracle_corrections(correction_manifest_path)
    if (
        correction_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("saturation gate manifests disagree")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("saturation gate requires the passing deterministic v3 preflight")

    identity = {
        "schema_version": 1,
        "experiment": "chunked_saturation_attribution_gate_v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "correction_manifest_content_sha256": correction_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "candidate_order": list(SATURATION_CANDIDATE_ORDER),
        "fixed_training_config": dict(_SATURATION_FIXED),
        "promotion_rule": (
            "all candidates run; promote the earliest passer in candidate_order "
            "(decoder family first, then penalty family, then unmitigated "
            "corrections; single-ingredient before combined within a family)"
        ),
        "early_stop": "none_all_candidates_required_for_attribution",
        "offline_role": "telemetry_only_promotion_by_closed_loop",
        "pass_rule": "three deterministic nominal successes with zero safety counts",
    }
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    gate_digest = content_sha256(identity)
    directory = output_dir / "saturation_gates" / gate_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="saturation gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable saturation gate differs: {directory}")
        return ChunkedGateResult(directory, report_path, str(existing["status"]))

    if workers is None:
        workers = 10 if record_video else 15
    if numerics is not None:
        # GPU regime: run candidates sequentially in-process (one CUDA
        # context, one warm inductor cache — trainings are minutes each) and
        # spend the whole budget on each candidate's CPU eval rollouts.
        candidate_workers = 1
        eval_workers = max(1, workers)
    else:
        candidate_workers = min(len(SATURATION_CANDIDATE_ORDER), max(1, workers))
        eval_workers = max(1, workers // candidate_workers) if workers > 1 else 1
    tasks = [
        _SaturationCandidateTask(
            candidate_id=candidate_id,
            model_path=str(model_path),
            oracle_manifest_path=str(oracle_manifest_path),
            correction_manifest_path=str(correction_manifest_path),
            output_dir=str(output_dir),
            gate_digest=gate_digest,
            oracle_manifest_content_sha256=oracle_manifest["content_sha256"],
            correction_manifest_content_sha256=correction_manifest["content_sha256"],
            preflight_content_sha256=preflight["content_sha256"],
            record_video=record_video,
            eval_workers=eval_workers,
            legacy_numerics=numerics is None,
        )
        for candidate_id in SATURATION_CANDIDATE_ORDER
    ]
    if candidate_workers <= 1:
        candidates = [_run_saturation_candidate(task) for task in tasks]
    else:
        pool = ProcessPoolExecutor(
            max_workers=candidate_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        try:
            futures = [pool.submit(_run_saturation_candidate, task) for task in tasks]
            # Collect in SATURATION_CANDIDATE_ORDER so the report is
            # byte-identical to the sequential gate.
            candidates = [future.result() for future in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    status = resolve_saturation_gate_status(candidates)
    report = {
        **identity,
        "gate_digest": gate_digest,
        "status": status,
        "next_action": (
            "broader_evaluation_of_promoted_policy_only"
            if status.startswith("promoted_")
            else "stop_and_write_new_proposal_replanning_cadence_or_smaller_final_chunk"
        ),
        "attribution": {
            item["candidate_id"]: item["state"] == "nominal_passed" for item in candidates
        },
        "candidates": candidates,
        "training_runs_started": len(candidates),
        "nominal_evaluations_started": len(candidates),
        "partial_milestones_are_non_promoting": True,
        "broader_evaluation_run": False,
        "claim": "saturation_attribution_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return ChunkedGateResult(directory, report_path, status)


__all__ = [
    "CHUNK_HORIZON_LADDER",
    "ChunkedClonePolicy",
    "ChunkedGateResult",
    "SATURATION_CANDIDATE_ORDER",
    "resolve_chunked_gate_status",
    "resolve_saturation_gate_status",
    "run_chunked_gate",
    "run_saturation_gate",
]
