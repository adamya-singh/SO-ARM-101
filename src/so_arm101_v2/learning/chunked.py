"""Chunked-action clones: the Phase-8 promotion rung after the tiny lane.

A chunked clone reads the 10 ``phase_state`` inputs once per chunk and emits
the next ``H`` absolute commands, executed open-loop through the unchanged
safety path.  Offline metrics are recorded as telemetry only: per the
promotion proposal (``notes/chunked-promotion-proposal.md``) the near-exact
memorization gate is retired, and promotion is decided solely by
deterministic nominal MuJoCo closed-loop evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from so_arm101_v2.contracts import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    JOINT_NAMES,
    effective_safe_act_bounds,
    load_task_contract,
)
from so_arm101_v2.data._serialization import (
    content_sha256,
    write_immutable_bytes,
    write_immutable_json,
)
from so_arm101_v2.simulation.oracle import load_oracle_demonstrations

from .numerics import (
    AUTO,
    NumericsSpec,
    apply_numerics,
    cpu_state_dict,
    noise_generator,
    numerics_identity,
    resolve_default_numerics,
)
from .oracle_distillation import OracleCloneKind, build_oracle_features
from .tiny_model import normalize_act

SATURATION_MODES = ("none", "feasible_chain_v1", "noise_penalty_v1")


# The oracle's own closest approach to the effective envelope is ~0.0035 ACT
# (gripper floor); the decoder margin must stay below it so every training
# target remains strictly inside the shrunk box.
_MAXIMUM_MARGIN_ACT = 0.0035


@dataclass(frozen=True)
class ChunkedCloneConfig:
    chunk_horizon: int
    seed: int = 101
    hidden_width: int = 256
    learning_rate: float = 1e-3
    max_steps: int = 30_000
    saturation_mode: str = "none"
    decoder_eta: float | None = None
    margin_act: float | None = None
    noise_sigma: float | None = None
    penalty_weight: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.chunk_horizon <= 480:
            raise ValueError("chunk_horizon must lie in [1, 480]")
        if self.seed < 0 or self.learning_rate <= 0 or self.max_steps <= 0:
            raise ValueError("invalid chunked clone configuration")
        if self.hidden_width not in (128, 256, 512, 1024):
            raise ValueError("chunked clone hidden_width must be 128, 256, 512, or 1024")
        if self.saturation_mode not in SATURATION_MODES:
            raise ValueError("unsupported saturation_mode")
        if self.saturation_mode == "none":
            if any(
                value is not None for value in (
                    self.decoder_eta, self.margin_act,
                    self.noise_sigma, self.penalty_weight,
                )
            ):
                raise ValueError("saturation parameters require a saturation mode")
            return
        if (
            self.decoder_eta is None or not 0 < self.decoder_eta <= 1
            or self.margin_act is None or not 0 <= self.margin_act < _MAXIMUM_MARGIN_ACT
        ):
            raise ValueError("saturation modes require decoder_eta in (0,1] and margin_act in [0, 0.0035)")
        if self.saturation_mode == "noise_penalty_v1":
            if (
                self.noise_sigma is None or self.noise_sigma <= 0
                or self.penalty_weight is None or self.penalty_weight <= 0
            ):
                raise ValueError("noise_penalty_v1 requires positive noise_sigma and penalty_weight")
        elif self.noise_sigma is not None or self.penalty_weight is not None:
            raise ValueError("noise parameters are only valid for noise_penalty_v1")


@dataclass(frozen=True)
class ChunkedCloneResult:
    chunk_horizon: int
    steps: int
    normalized_mse: float
    max_act_error: float
    directory: Path
    checkpoint: Path
    report_json: Path


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("chunked distillation requires the 'learn' extra") from exc
    return torch


def build_chunked_clone_model(input_dim: int, hidden_width: int, chunk_horizon: int) -> Any:
    torch = _torch()
    if hidden_width not in (128, 256, 512, 1024):
        raise ValueError("chunked clone hidden_width must be 128, 256, 512, or 1024")
    if not 1 <= chunk_horizon <= 480:
        raise ValueError("chunk_horizon must lie in [1, 480]")

    class ChunkedCloneNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_width),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_width, hidden_width),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_width, chunk_horizon * 6),
            )
            with torch.no_grad():
                self.network[-1].weight.zero_()
                self.network[-1].bias.zero_()

        def forward(self, features: Any) -> Any:
            return self.network(features)

    return ChunkedCloneNetwork()


def build_chunked_targets(
    arrays: Mapping[str, np.ndarray],
    chunk_horizon: int,
    *,
    episode_lengths: Sequence[int] | None = None,
) -> np.ndarray:
    """Normalized-act residual targets ``[rows, H, 6]`` with repeat-last padding.

    Target ``[t, j]`` is ``normalize_act(executed[min(t + j, last)]) -
    normalize_act(current[t])`` so a zero-initialized head predicts "hold the
    current pose" for the whole chunk.  ``episode_lengths`` bounds each row's
    lookahead to its own episode: padding repeats that episode's final
    command, never crossing into the next episode.
    """
    executed = np.asarray(arrays["executed_act"], dtype=np.float32)
    current = np.asarray(arrays["current_act"], dtype=np.float32)
    if executed.ndim != 2 or executed.shape[1] != 6 or executed.shape != current.shape:
        raise ValueError("chunked targets require matching [N, 6] act arrays")
    rows = executed.shape[0]
    if episode_lengths is None:
        last_index = np.full(rows, rows - 1, dtype=np.int64)
    else:
        lengths = [int(value) for value in episode_lengths]
        if any(value <= 0 for value in lengths) or sum(lengths) != rows:
            raise ValueError("episode lengths must be positive and sum to the row count")
        last_index = np.concatenate([
            np.full(length, offset + length - 1, dtype=np.int64)
            for offset, length in zip(np.cumsum([0, *lengths[:-1]]), lengths)
        ])
    executed_norm = normalize_act(executed)
    current_norm = normalize_act(current)
    indices = np.minimum(
        np.arange(rows)[:, None] + np.arange(chunk_horizon)[None, :],
        last_index[:, None],
    )
    return executed_norm[indices] - current_norm[:, None, :]


@functools.lru_cache(maxsize=8)
def _feasible_decode_constants(margin_act: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized-act box and per-step delta cap shared by decode and penalty."""
    low_act, high_act = effective_safe_act_bounds()
    low_norm = normalize_act(low_act + np.float32(margin_act))
    high_norm = normalize_act(high_act - np.float32(margin_act))
    maximum_delta = np.asarray(
        load_task_contract("fixed_cube_pickup_v1").safety.maximum_act_delta_per_step,
        dtype=np.float32,
    )
    delta_norm = (maximum_delta * 2.0 / (ACT_DATASET_HIGH - ACT_DATASET_LOW)).astype(np.float32)
    return low_norm, high_norm, delta_norm


def decode_feasible_chunk(
    raw: Any, current_norm: Any, *, eta: float, margin_act: float
) -> Any:
    """Decode raw chunk outputs into feasible normalized-act commands.

    Each command is a tanh-bounded step (at most ``eta`` of the per-step delta
    cap) from the previous command, clamped into the margin-shrunk effective
    box.  Zero raw output holds the current pose; every training target is
    exactly representable (oracle deltas are 6x under the cap and 0.004 inside
    the box), so the decode is near-identity on-distribution and saturates
    gracefully off-distribution instead of leaving the envelope.
    """
    torch = _torch()
    low_norm, high_norm, delta_norm = _feasible_decode_constants(margin_act)
    low = torch.from_numpy(low_norm).to(raw.device)
    high = torch.from_numpy(high_norm).to(raw.device)
    step = (torch.from_numpy(delta_norm) * float(eta)).to(raw.device)
    previous = current_norm
    commands = []
    for offset in range(raw.shape[1]):
        delta = step * torch.tanh(raw[:, offset, :] / step)
        previous = torch.clamp(previous + delta, low, high)
        commands.append(previous)
    return torch.stack(commands, dim=1)


def feasible_decode_telemetry(
    raw: np.ndarray, current_norm: np.ndarray, *, eta: float, margin_act: float
) -> dict[str, float]:
    """Convergence telemetry for the feasible decoder (never gates)."""
    low_norm, high_norm, delta_norm = _feasible_decode_constants(margin_act)
    step = delta_norm * np.float32(eta)
    previous = current_norm.copy()
    clamped = 0
    total = 0
    tanh_max = 0.0
    for offset in range(raw.shape[1]):
        activation = np.tanh(raw[:, offset, :] / step)
        tanh_max = max(tanh_max, float(np.max(np.abs(activation))))
        unclamped = previous + step * activation
        bounded = np.clip(unclamped, low_norm, high_norm)
        clamped += int(np.count_nonzero(bounded != unclamped))
        total += unclamped.size
        previous = bounded
    return {
        "clamp_active_fraction": clamped / total,
        "tanh_activation_max_abs": tanh_max,
    }


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    write_immutable_bytes(
        path, data, conflict_message=f"immutable chunked artifact differs: {path}"
    )


def _load_hashed_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    stated = report.get("content_sha256")
    body = dict(report)
    body.pop("content_sha256", None)
    if not isinstance(stated, str) or stated != content_sha256(body):
        raise ValueError(f"chunked report content hash mismatch: {path}")
    return report


def _chunked_step_loss(
    model: Any,
    features: Any,
    targets: Any,
    current_norm: Any,
    noisy: Any | None,
    chain_constants: tuple[Any, Any, Any] | None,
    penalty_constants: tuple[Any, Any, Any] | None,
    penalty_weight: float,
    chunk_horizon: int,
    rows: int,
) -> tuple[Any, Any | None]:
    """One full-batch training step's (loss, penalty), op-for-op the legacy math.

    Pure tensor function so regime v2 can run it through torch.compile as a
    single fused graph; every host sync (isfinite, .item(), grad norms) stays
    with the caller.  The feasible chain is inlined so its 90 serial
    iterations fuse instead of dispatching ~540 tiny kernels.
    """
    import torch

    raw = model(features)
    if chain_constants is not None:
        low, high, step_scaled = chain_constants
        chunk = raw.view(rows, chunk_horizon, 6)
        previous = current_norm
        commands = []
        for offset in range(chunk_horizon):
            delta = step_scaled * torch.tanh(chunk[:, offset, :] / step_scaled)
            previous = torch.clamp(previous + delta, low, high)
            commands.append(previous)
        decoded = torch.stack(commands, dim=1)
        residual = (decoded - current_norm[:, None, :]).reshape(rows, -1)
    else:
        residual = raw
    loss = (residual - targets).square().mean()
    penalty = None
    if penalty_constants is not None:
        penalty_low, penalty_high, penalty_step = penalty_constants
        raw_noisy = model(noisy).view(rows, chunk_horizon, 6)
        noisy_current = noisy[:, :6]
        commands = noisy_current[:, None, :] + raw_noisy
        box_violation = torch.relu(commands - penalty_high) + torch.relu(penalty_low - commands)
        deltas = torch.cat([
            commands[:, :1, :] - noisy_current[:, None, :],
            commands[:, 1:, :] - commands[:, :-1, :],
        ], dim=1)
        delta_violation = torch.relu(deltas.abs() - penalty_step)
        penalty = box_violation.mean() + delta_violation.mean()
        loss = loss + penalty_weight * penalty
    return loss, penalty


def train_chunked_clone(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: ChunkedCloneConfig,
    correction_manifest_path: str | Path | None = None,
    numerics: NumericsSpec | None | str = AUTO,
) -> ChunkedCloneResult:
    """Train one fixed chunked clone; offline metrics are telemetry, not a gate."""
    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()
    manifest, arrays = load_oracle_demonstrations(manifest_path)
    source_rows = int(np.asarray(arrays["action_index"]).shape[0])
    # Chunk lookahead never crosses an episode boundary: derive the nominal
    # block's episode lengths from the manifest (a single 450-row episode
    # yields [450], which is exactly the whole-block behavior).
    episode_lengths = [int(item["rows"]) for item in manifest["episodes"]]
    if sum(episode_lengths) != source_rows:
        raise ValueError("oracle manifest episode lengths disagree with its row count")
    correction_manifest: dict[str, Any] | None = None
    training_arrays: Mapping[str, np.ndarray] = arrays
    if correction_manifest_path is not None:
        from so_arm101_v2.simulation.correction import load_oracle_corrections

        correction_manifest, correction_arrays = load_oracle_corrections(correction_manifest_path)
        if correction_manifest.get("source_manifest_content_sha256") != manifest["content_sha256"]:
            raise ValueError("correction trajectories do not derive from the nominal oracle manifest")
        accepted = [item for item in correction_manifest["episodes"] if item["accepted"]]
        episode_lengths = episode_lengths + [int(item["rows"]) for item in accepted]
        training_arrays = {
            name: np.concatenate([
                np.asarray(arrays[name], dtype=np.float32),
                np.asarray(correction_arrays[name], dtype=np.float32),
            ])
            for name in ("current_act", "cube_position", "progress", "executed_act")
        }
    # Normalization statistics come from the source manifest's rows alone
    # (capture-wide for multi-scenario manifests), never from corrections.
    _, extras_mean, extras_std = build_oracle_features(OracleCloneKind.PHASE_STATE, arrays)
    features_np, _, _ = build_oracle_features(
        OracleCloneKind.PHASE_STATE, training_arrays,
        extras_mean=extras_mean, extras_std=extras_std,
    )
    targets_np = build_chunked_targets(
        training_arrays, config.chunk_horizon, episode_lengths=episode_lengths,
    )
    rows = int(features_np.shape[0])
    model_kind = f"chunked_h{config.chunk_horizon}"
    # Saturation keys enter the identity only when a mode is active, and
    # correction keys only when a manifest is supplied, so every legacy
    # chunked run digest stays byte-identical.
    config_identity = asdict(config)
    if config.saturation_mode == "none":
        for name in (
            "saturation_mode", "decoder_eta", "margin_act",
            "noise_sigma", "penalty_weight",
        ):
            config_identity.pop(name)
    identity = {
        "manifest_content_sha256": manifest["content_sha256"],
        "collection_digest": manifest["collection_digest"],
        "model_kind": model_kind,
        "optimizer": "adam_full_batch",
        "source_rows": source_rows,
        "config": config_identity,
        "target": "normalized_absolute_act_residual_on_current_pose",
        "chunk_padding": "repeat_final_command",
        "offline_role": "telemetry_only_promotion_by_closed_loop",
    }
    if correction_manifest is not None:
        identity["correction_manifest_content_sha256"] = correction_manifest["content_sha256"]
        identity["correction_collection_digest"] = correction_manifest["collection_digest"]
        identity["correction_rows"] = rows - source_rows
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    run_digest = content_sha256(identity)
    directory = Path(output_dir) / "models" / model_kind / run_digest[:16]
    existing = directory / "report.json"
    if existing.is_file():
        report = _load_hashed_report(existing)
        if report.get("run_digest") != run_digest or any(
            report.get(field) != value for field, value in identity.items()
        ):
            raise FileExistsError(f"immutable chunked run identity differs: {directory}")
        if not (directory / "model.pt").is_file():
            raise FileNotFoundError(f"immutable chunked run is missing sidecars: {directory}")
        return ChunkedCloneResult(
            chunk_horizon=config.chunk_horizon,
            steps=int(report["steps"]),
            normalized_mse=float(report["normalized_mse"]),
            max_act_error=float(report["max_act_error"]),
            directory=directory,
            checkpoint=directory / "model.pt",
            report_json=existing,
        )

    torch = _torch()
    device = apply_numerics(torch, numerics, seed=config.seed)
    features = torch.from_numpy(features_np).to(device)
    targets = torch.from_numpy(targets_np.reshape(rows, -1)).to(device)
    current_norm_np = normalize_act(np.asarray(training_arrays["current_act"], dtype=np.float32))
    current_norm = torch.from_numpy(current_norm_np).to(device)
    model = build_chunked_clone_model(
        features_np.shape[1], config.hidden_width, config.chunk_horizon
    ).to(device)
    with torch.inference_mode():
        initial_output = model(features).cpu().numpy()
    if not np.array_equal(initial_output, np.zeros_like(initial_output)):
        raise RuntimeError("chunked residual head did not initialize to hold")
    chain_mode = config.saturation_mode == "feasible_chain_v1"
    penalty_mode = config.saturation_mode == "noise_penalty_v1"
    if penalty_mode:
        low_norm, high_norm, delta_norm = _feasible_decode_constants(config.margin_act)
        penalty_low = torch.from_numpy(low_norm).to(device)
        penalty_high = torch.from_numpy(high_norm).to(device)
        penalty_step = (torch.from_numpy(delta_norm) * float(config.decoder_eta)).to(device)

    def forward_residual() -> Any:
        raw = model(features)
        if chain_mode:
            decoded = decode_feasible_chunk(
                raw.view(rows, config.chunk_horizon, 6), current_norm,
                eta=config.decoder_eta, margin_act=config.margin_act,
            )
            return (decoded - current_norm[:, None, :]).reshape(rows, -1)
        return raw

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_trace: list[dict[str, float | int]] = []
    first_gradient_norms: list[float] = []
    penalty_value: float | None = None
    if numerics is None:
        # Legacy CPU-eager loop, byte-for-byte: global-RNG noise, per-step
        # penalty sync, eager dispatch.  This lane pins every stored digest.
        for step in range(1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = (forward_residual() - targets).square().mean()
            if penalty_mode:
                noisy = features + torch.randn_like(features) * float(config.noise_sigma)
                raw_noisy = model(noisy).view(rows, config.chunk_horizon, 6)
                noisy_current = noisy[:, :6]
                commands = noisy_current[:, None, :] + raw_noisy
                box_violation = torch.relu(commands - penalty_high) + torch.relu(penalty_low - commands)
                deltas = torch.cat([
                    commands[:, :1, :] - noisy_current[:, None, :],
                    commands[:, 1:, :] - commands[:, :-1, :],
                ], dim=1)
                delta_violation = torch.relu(deltas.abs() - penalty_step)
                penalty = box_violation.mean() + delta_violation.mean()
                penalty_value = float(penalty.item())
                loss = loss + float(config.penalty_weight) * penalty
            if not torch.isfinite(loss):
                raise RuntimeError("chunked distillation loss became nonfinite")
            loss.backward()
            if step == 1:
                first_gradient_norms = [
                    float(parameter.grad.norm().item())
                    for parameter in model.parameters() if parameter.grad is not None
                ]
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == config.max_steps:
                loss_trace.append({"step": step, "normalized_mse": float(loss.item())})
    else:
        chain_constants = None
        if chain_mode:
            low_norm, high_norm, delta_norm = _feasible_decode_constants(config.margin_act)
            chain_constants = (
                torch.from_numpy(low_norm).to(device),
                torch.from_numpy(high_norm).to(device),
                (torch.from_numpy(delta_norm) * float(config.decoder_eta)).to(device),
            )
        penalty_constants = (penalty_low, penalty_high, penalty_step) if penalty_mode else None
        step_loss = _chunked_step_loss
        if numerics.compile == "inductor":
            step_loss = torch.compile(
                _chunked_step_loss, mode=numerics.compile_mode,
                fullgraph=True, dynamic=False,
            )
        noise_source = noise_generator(torch, config.seed) if penalty_mode else None
        penalty = None
        for step in range(1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            noisy = None
            if penalty_mode:
                # cpu_generator_v1: device-invariant noise stream (D3).
                noise = torch.randn(features.shape, generator=noise_source)
                noisy = features + noise.to(device) * float(config.noise_sigma)
            loss, penalty = step_loss(
                model, features, targets, current_norm, noisy,
                chain_constants, penalty_constants,
                float(config.penalty_weight or 0.0), config.chunk_horizon, rows,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("chunked distillation loss became nonfinite")
            loss.backward()
            if step == 1:
                first_gradient_norms = [
                    float(parameter.grad.norm().item())
                    for parameter in model.parameters() if parameter.grad is not None
                ]
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == config.max_steps:
                loss_trace.append({"step": step, "normalized_mse": float(loss.item())})
        if penalty_mode and penalty is not None:
            # Same reported value as the legacy per-step sync, one sync total.
            penalty_value = float(penalty.item())
    with torch.inference_mode():
        raw_final = model(features)
        predictions = forward_residual().cpu().numpy().astype(np.float32)
    residual = predictions - targets_np.reshape(rows, -1)
    normalized_mse = float(np.mean(np.square(residual)))
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("chunked predictions contain nonfinite values")
    decoder_telemetry: dict[str, float] | None = None
    if chain_mode:
        decoder_telemetry = feasible_decode_telemetry(
            raw_final.view(rows, config.chunk_horizon, 6).cpu().numpy().astype(np.float32),
            current_norm_np, eta=config.decoder_eta, margin_act=config.margin_act,
        )
    # Convert normalized-act residual error into ACT units per joint.
    act_scale = ((ACT_DATASET_HIGH - ACT_DATASET_LOW) * 0.5).astype(np.float32)
    act_error = np.abs(
        residual.reshape(rows, config.chunk_horizon, 6)
    ) * act_scale[None, None, :]
    max_act_error = float(np.max(act_error))
    per_offset_max = np.max(act_error, axis=(0, 2)).astype(np.float64).tolist()
    zero_baseline = float(np.mean(np.square(targets_np)))
    report: dict[str, Any] = {
        "schema_version": 1,
        **identity,
        "run_digest": run_digest,
        "rows": rows,
        "input_dim": int(features_np.shape[1]),
        "hidden_width": config.hidden_width,
        "chunk_horizon": config.chunk_horizon,
        "architecture": [
            f"linear_{features_np.shape[1]}_{config.hidden_width}", "relu",
            f"linear_{config.hidden_width}_{config.hidden_width}", "relu",
            f"linear_{config.hidden_width}_{config.chunk_horizon * 6}_zero_init",
        ],
        "normalization": {
            "extras_mean": extras_mean.tolist(),
            "extras_std": extras_std.tolist(),
            "std_floor": 1e-6,
        },
        "steps": config.max_steps,
        "normalized_mse": normalized_mse,
        "max_act_error": max_act_error,
        "per_offset_max_act_error": per_offset_max,
        "zero_hold_baseline_normalized_mse": zero_baseline,
        "zero_hold_improvement_factor": (
            zero_baseline / normalized_mse if normalized_mse > 0 else None
        ),
        "gradient_norms_at_step_1": first_gradient_norms,
        "loss_trace": loss_trace,
        "joint_order": list(JOINT_NAMES),
        "claim": "offline_telemetry_only_promotion_requires_closed_loop_evaluation",
    }
    if config.saturation_mode != "none":
        low_act, high_act = effective_safe_act_bounds()
        report["saturation"] = {
            "mode": config.saturation_mode,
            "decoder_eta": config.decoder_eta,
            "margin_act": config.margin_act,
            "noise_sigma": config.noise_sigma,
            "penalty_weight": config.penalty_weight,
            "effective_act_low": low_act.tolist(),
            "effective_act_high": high_act.tolist(),
            "final_penalty_value": penalty_value,
            "decoder_telemetry": decoder_telemetry,
        }
    if correction_manifest is not None:
        report["correction_augmentation"] = {
            "manifest_content_sha256": correction_manifest["content_sha256"],
            "collection_digest": correction_manifest["collection_digest"],
            "rows": rows - source_rows,
            "episode_lengths": episode_lengths,
            "normalization": "unchanged_nominal_statistics",
        }
    report["content_sha256"] = content_sha256(report)
    checkpoint_payload = {
        "schema_version": 1,
        **identity,
        "run_digest": run_digest,
        "input_dim": int(features_np.shape[1]),
        "hidden_width": config.hidden_width,
        "chunk_horizon": config.chunk_horizon,
        "extras_mean": torch.from_numpy(extras_mean.copy()),
        "extras_std": torch.from_numpy(extras_std.copy()),
        "teacher_horizon": int(manifest["teacher_horizon"]),
        "state_dict": cpu_state_dict(model) if numerics is not None else model.state_dict(),
        "report_content_sha256": report["content_sha256"],
    }
    if config.saturation_mode != "none":
        checkpoint_payload["saturation_mode"] = config.saturation_mode
        checkpoint_payload["decoder_eta"] = config.decoder_eta
        checkpoint_payload["margin_act"] = config.margin_act
    if correction_manifest is not None:
        checkpoint_payload["correction_augmentation"] = report["correction_augmentation"]
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint_payload, checkpoint_buffer)
    _write_immutable_bytes(directory / "model.pt", checkpoint_buffer.getvalue())
    write_immutable_json(directory / "report.json", report)
    return ChunkedCloneResult(
        chunk_horizon=config.chunk_horizon,
        steps=config.max_steps,
        normalized_mse=normalized_mse,
        max_act_error=max_act_error,
        directory=directory,
        checkpoint=directory / "model.pt",
        report_json=directory / "report.json",
    )


__all__ = [
    "ChunkedCloneConfig",
    "ChunkedCloneResult",
    "SATURATION_MODES",
    "build_chunked_clone_model",
    "build_chunked_targets",
    "decode_feasible_chunk",
    "feasible_decode_telemetry",
    "train_chunked_clone",
]
