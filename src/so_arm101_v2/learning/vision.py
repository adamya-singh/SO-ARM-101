"""Vision-conditioned chunked clones (exploratory vision rung).

Replaces the privileged ``cube_position`` feature with the wrist camera
frame: inputs are pixels + ``normalize_act(current_act)`` + the open-loop
progress clock. Trained by minibatched Adam over a frames sidecar capture
(``images.npy`` memmap; see ``capture_oracle_demonstrations(store_frames=True)``).

This lane is exploratory (`notes/vision-rung-notebook.md`): artifacts stay
content-addressed and numerics-fingerprinted, but there are no legacy
digests to preserve and no pre-registration per run.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

from .chunked import (
    build_chunked_targets,
    chunked_learning_rate,
    CHUNKED_LR_SCHEDULES,
    _write_immutable_bytes,
)
from .numerics import (
    AUTO,
    NumericsSpec,
    apply_numerics,
    cpu_state_dict,
    noise_generator,
    numerics_identity,
    resolve_default_numerics,
)
from .tiny_model import normalize_act

VISION_INPUT_SCHEMA = ("wrist_frame[256x256x3]", "current_act[6]", "progress[1]")
VISION_STATE_DIM = 7


@dataclass(frozen=True)
class VisionChunkedConfig:
    chunk_horizon: int = 90
    seed: int = 101
    hidden_width: int = 256
    learning_rate: float = 1e-3
    max_steps: int = 20_000
    batch_size: int = 64
    lr_schedule: str = "cosine_floor_v1"

    def __post_init__(self) -> None:
        if not 1 <= self.chunk_horizon <= 480:
            raise ValueError("chunk_horizon must lie in [1, 480]")
        if self.seed < 0 or self.learning_rate <= 0 or self.max_steps <= 0:
            raise ValueError("invalid vision clone configuration")
        if self.hidden_width not in (128, 256, 512):
            raise ValueError("vision clone hidden_width must be 128, 256, or 512")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr_schedule not in CHUNKED_LR_SCHEDULES:
            raise ValueError("unsupported lr_schedule")


@dataclass(frozen=True)
class VisionChunkedResult:
    chunk_horizon: int
    steps: int
    normalized_mse: float
    directory: Path
    checkpoint: Path
    report_json: Path


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vision distillation requires the 'learn' extra") from exc
    return torch


def build_vision_chunked_model(hidden_width: int, chunk_horizon: int) -> Any:
    """Conv trunk (the proven image_state encoder) + state branch + zero-init head."""
    torch = _torch()
    if hidden_width not in (128, 256, 512):
        raise ValueError("vision clone hidden_width must be 128, 256, or 512")
    if not 1 <= chunk_horizon <= 480:
        raise ValueError("chunk_horizon must lie in [1, 480]")

    class VisionChunkedNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=8, stride=8), torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, kernel_size=4, stride=4), torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            self.state = torch.nn.Sequential(
                torch.nn.Linear(VISION_STATE_DIM, 32), torch.nn.ReLU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(512 + 32, hidden_width), torch.nn.ReLU(),
                torch.nn.Linear(hidden_width, hidden_width), torch.nn.ReLU(),
                torch.nn.Linear(hidden_width, chunk_horizon * 6),
            )
            with torch.no_grad():
                self.head[-1].weight.zero_()
                self.head[-1].bias.zero_()

        def forward(self, images: Any, state: Any) -> Any:
            return self.head(torch.cat([self.encoder(images), self.state(state)], dim=1))

    return VisionChunkedNetwork()


def _read_frame_rows(frames: Any, index_array: "np.ndarray") -> "np.ndarray":
    """Read ``frames[index_array]`` as raw uint8, via a sorted gather.

    Sorting improves disk locality on a memmap; the inverse permutation
    restores the exact requested order, so the result is bitwise identical
    to direct fancy indexing.
    """
    order = np.argsort(index_array, kind="stable")
    block = np.asarray(frames[index_array[order]])
    result = np.empty_like(block)
    result[order] = block
    return result


def load_vision_frames(manifest_path: str | Path) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    """Manifest + memmapped frames + demonstration arrays for a frames capture."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stated = manifest.pop("content_sha256")
    if content_sha256(manifest) != stated:
        raise ValueError("oracle manifest content hash mismatch")
    manifest["content_sha256"] = stated
    frames_block = manifest.get("frames")
    if frames_block is None:
        raise ValueError("vision training requires a frames-sidecar capture (store_frames)")
    frames_path = manifest_path.parent / frames_block["path"]
    frames = np.load(frames_path, mmap_mode="r")
    if frames.shape != (int(frames_block["rows"]), 256, 256, 3) or frames.dtype != np.uint8:
        raise ValueError("frames sidecar shape/dtype mismatch")
    with np.load(manifest_path.parent / manifest["arrays"]["path"]) as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    if arrays["action_index"].shape[0] != frames.shape[0]:
        raise ValueError("frames sidecar row count disagrees with the arrays")
    return manifest, frames, arrays


def train_vision_chunked(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: VisionChunkedConfig,
    numerics: NumericsSpec | None | str = AUTO,
    on_loss: "Callable[[int, float], None] | None" = None,
) -> VisionChunkedResult:
    """Train one vision-conditioned chunked clone (minibatched).

    ``on_loss`` is a pure observer called as ``on_loss(step, batch_mse)`` at
    the loss-trace cadence (external experiment tracking); it never enters
    the identity payload and must not affect training.
    """
    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()

    manifest, frames, arrays = load_vision_frames(manifest_path)
    episode_lengths = [int(item["rows"]) for item in manifest["episodes"]]
    rows = int(arrays["action_index"].shape[0])
    if sum(episode_lengths) != rows:
        raise ValueError("oracle manifest episode lengths disagree with its row count")
    targets_np = build_chunked_targets(
        arrays, config.chunk_horizon, episode_lengths=episode_lengths,
    ).reshape(rows, -1)
    current_norm_np = normalize_act(np.asarray(arrays["current_act"], dtype=np.float32))
    state_np = np.concatenate(
        [current_norm_np, np.asarray(arrays["progress"], dtype=np.float32)[:, None]],
        axis=1,
    ).astype(np.float32)

    model_kind = f"vision_h{config.chunk_horizon}"
    identity: dict[str, Any] = {
        "manifest_content_sha256": manifest["content_sha256"],
        "collection_digest": manifest["collection_digest"],
        "frames_sha256": manifest["frames"]["sha256"],
        "model_kind": model_kind,
        "optimizer": "adam_minibatch",
        "source_rows": rows,
        "config": asdict(config),
        "input_schema": list(VISION_INPUT_SCHEMA),
        "image_convention": "preprocess_wrist_image_div255_chw",
        "target": "normalized_absolute_act_residual_on_current_pose",
        "chunk_padding": "repeat_final_command",
        "offline_role": "telemetry_only_promotion_by_closed_loop",
    }
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    run_digest = content_sha256(identity)
    directory = Path(output_dir) / "models" / model_kind / run_digest[:16]
    existing = directory / "report.json"
    if existing.is_file():
        report = json.loads(existing.read_text(encoding="utf-8"))
        if report.get("run_digest") != run_digest:
            raise FileExistsError(f"immutable vision run identity differs: {directory}")
        return VisionChunkedResult(
            chunk_horizon=config.chunk_horizon,
            steps=int(report["steps"]),
            normalized_mse=float(report["normalized_mse"]),
            directory=directory,
            checkpoint=directory / "model.pt",
            report_json=existing,
        )

    torch = _torch()
    device = apply_numerics(torch, numerics, seed=config.seed)
    model = build_vision_chunked_model(config.hidden_width, config.chunk_horizon).to(device)
    targets = torch.from_numpy(targets_np).to(device)
    state = torch.from_numpy(state_np).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = noise_generator(torch, config.seed)

    def batch_images(indices: Any) -> Any:
        block = np.asarray(frames[indices.numpy()], dtype=np.float32) / np.float32(255.0)
        return torch.from_numpy(np.transpose(block, (0, 3, 1, 2))).to(device)

    def to_device_images(uint8_block: "np.ndarray") -> Any:
        # Identical float path to batch_images, applied to a pre-read block.
        block = np.asarray(uint8_block, dtype=np.float32) / np.float32(255.0)
        return torch.from_numpy(np.transpose(block, (0, 3, 1, 2))).to(device)

    # Deterministic prefetching (see notes/vision-rung-notebook.md): the
    # per-step minibatch read is the only disk I/O in the loop and dominates
    # wall time once the frames sidecar outgrows the page cache.  Reader
    # threads perform ONLY the raw uint8 memmap reads, many steps ahead and
    # concurrently (higher effective disk queue depth); index draws stay on
    # this thread in step order, so the seeded RNG stream, every float op,
    # and the op order are unchanged — results are bitwise identical to the
    # synchronous path (pinned by test).  Identity/digests unaffected.
    prefetch_enabled = os.environ.get("SO_ARM101_V2_PREFETCH", "1") != "0"
    prefetch_depth = max(1, int(os.environ.get("SO_ARM101_V2_PREFETCH_DEPTH", "8")))
    prefetch_workers = max(1, int(os.environ.get("SO_ARM101_V2_PREFETCH_WORKERS", "6")))

    loss_trace: list[dict[str, float | int]] = []
    batch = min(config.batch_size, rows)
    executor = None
    pending: "deque[tuple[Any, Any]]" = deque()
    drawn = 0

    def draw_indices() -> Any:
        return torch.randperm(rows, generator=generator)[:batch]

    if prefetch_enabled:
        executor = ThreadPoolExecutor(max_workers=prefetch_workers)

    def next_batch() -> "tuple[Any, Any]":
        nonlocal drawn
        if executor is None:
            indices = draw_indices()
            return indices, None
        while drawn < config.max_steps and len(pending) < prefetch_depth:
            indices = draw_indices()
            drawn += 1
            future = executor.submit(_read_frame_rows, frames, indices.numpy())
            pending.append((indices, future))
        indices, future = pending.popleft()
        return indices, future.result()

    try:
        for step in range(1, config.max_steps + 1):
            if config.lr_schedule != "fixed":
                lr_value = chunked_learning_rate(
                    config.lr_schedule, step, config.max_steps, config.learning_rate
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr_value
            indices, uint8_block = next_batch()
            optimizer.zero_grad(set_to_none=True)
            images = (
                batch_images(indices) if uint8_block is None
                else to_device_images(uint8_block)
            )
            prediction = model(images, state[indices.to(device)])
            loss = (prediction - targets[indices.to(device)]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("vision distillation loss became nonfinite")
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == config.max_steps:
                loss_value = float(loss.item())
                loss_trace.append({"step": step, "batch_normalized_mse": loss_value})
                if on_loss is not None:
                    on_loss(step, loss_value)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    # Final full-data evaluation in minibatches.
    model.eval()
    errors = []
    with torch.inference_mode():
        for start in range(0, rows, 256):
            indices = torch.arange(start, min(start + 256, rows))
            prediction = model(batch_images(indices), state[indices.to(device)])
            errors.append(
                (prediction - targets[indices.to(device)]).square().mean(dim=1).cpu().numpy()
            )
    normalized_mse = float(np.mean(np.concatenate(errors)))

    report: dict[str, Any] = {
        "schema_version": 1,
        **identity,
        "run_digest": run_digest,
        "rows": rows,
        "hidden_width": config.hidden_width,
        "chunk_horizon": config.chunk_horizon,
        "teacher_horizon": int(manifest["teacher_horizon"]),
        "steps": config.max_steps,
        "normalized_mse": normalized_mse,
        "loss_trace": loss_trace,
        "claim": "offline_telemetry_only_promotion_requires_closed_loop_evaluation",
    }
    report["content_sha256"] = content_sha256(report)
    checkpoint_payload = {
        "schema_version": 1,
        **identity,
        "run_digest": run_digest,
        "hidden_width": config.hidden_width,
        "chunk_horizon": config.chunk_horizon,
        "teacher_horizon": int(manifest["teacher_horizon"]),
        "state_dict": cpu_state_dict(model),
        "report_content_sha256": report["content_sha256"],
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint_payload, checkpoint_buffer)
    _write_immutable_bytes(directory / "model.pt", checkpoint_buffer.getvalue())
    write_immutable_json(directory / "report.json", report)
    return VisionChunkedResult(
        chunk_horizon=config.chunk_horizon,
        steps=config.max_steps,
        normalized_mse=normalized_mse,
        directory=directory,
        checkpoint=directory / "model.pt",
        report_json=directory / "report.json",
    )


__all__ = [
    "VISION_INPUT_SCHEMA",
    "VisionChunkedConfig",
    "VisionChunkedResult",
    "build_vision_chunked_model",
    "load_vision_frames",
    "train_vision_chunked",
]
