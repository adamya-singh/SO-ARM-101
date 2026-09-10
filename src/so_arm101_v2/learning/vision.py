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
import random
import tempfile
import time
import zlib
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
    # Train on every ``frame_stride``-th row of each episode (chunk starts 1/30 s apart are near
    # duplicates). 1 = every row (the historical recipe; identities unchanged). Introduced 2026-09-10
    # so the incompressible appearance-randomized frames can live on the GPU (see SO_ARM101_V2_FRAME_CACHE=gpu).
    frame_stride: int = 1
    # Image encoder: "v1" = the historical 8k-parameter trunk (stride-8 first layer, 8/16/32 channels, 4x4x32 map);
    # "v2" (2026-09-10) = stride-4 first layer, 16/32/64/64 channels, 8x8x64 map (4096 features) for sub-patch
    # localisation of a 12-25 px cube anywhere in the frame. Identity-bearing only when not "v1".
    encoder: str = "v1"

    def __post_init__(self) -> None:
        if not 1 <= self.chunk_horizon <= 480:
            raise ValueError("chunk_horizon must lie in [1, 480]")
        if int(self.frame_stride) < 1 or int(self.frame_stride) != self.frame_stride:
            raise ValueError("frame_stride must be a positive integer")
        if self.encoder not in ("v1", "v2"):
            raise ValueError("encoder must be 'v1' or 'v2'")
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


def build_vision_chunked_model(hidden_width: int, chunk_horizon: int, encoder: str = "v1") -> Any:
    """Conv trunk (the proven image_state encoder, or the finer v2 trunk) + state branch + zero-init head."""
    torch = _torch()
    if hidden_width not in (128, 256, 512):
        raise ValueError("vision clone hidden_width must be 128, 256, or 512")
    if not 1 <= chunk_horizon <= 480:
        raise ValueError("chunk_horizon must lie in [1, 480]")
    if encoder not in ("v1", "v2"):
        raise ValueError("encoder must be 'v1' or 'v2'")
    feature_width = 512 if encoder == "v1" else 4096

    class VisionChunkedNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if encoder == "v1":
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 8, kernel_size=8, stride=8), torch.nn.ReLU(),
                    torch.nn.Conv2d(8, 16, kernel_size=4, stride=4), torch.nn.ReLU(),
                    torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),
                    torch.nn.Flatten(),
                )
            else:
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 16, kernel_size=4, stride=4), torch.nn.ReLU(),                 # 256 -> 64
                    torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),     # 64 -> 32
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),     # 32 -> 16
                    torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),     # 16 -> 8
                    torch.nn.Flatten(),
                )
            self.state = torch.nn.Sequential(
                torch.nn.Linear(VISION_STATE_DIM, 32), torch.nn.ReLU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(feature_width + 32, hidden_width), torch.nn.ReLU(),
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


class CompressedFrames:
    """Lossless zlib copy of a frames sidecar held in RAM; indexing decodes rows to uint8 (n, 256, 256, 3).

    Rendered wrist frames are mostly flat (black ground, few surfaces) and compress ~40x, so a
    37.7 GB sidecar that cannot fit the page cache becomes < 1 GB of RAM. Decoding a row costs
    ~0.1 ms, versus a random disk read of 192 KiB, which was the whole training bottleneck
    (measured 2026-09-09: 10 steps/s disk-bound vs a 5 ms GPU step). Pixels are bit-identical to
    the memmap, so run digests and checkpoints are unaffected (pinned by test).
    """

    LEVEL = 3

    def __init__(self, blob: np.ndarray, offsets: np.ndarray, shape: tuple[int, ...]) -> None:
        self.blob = np.ascontiguousarray(blob, dtype=np.uint8)
        self.offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(np.uint8)
        if self.offsets.shape != (self.shape[0] + 1,):
            raise ValueError("frame cache offsets do not match the row count")

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        return int(self.blob.nbytes + self.offsets.nbytes)

    def _row(self, index: int) -> np.ndarray:
        start, stop = self.offsets[index], self.offsets[index + 1]
        raw = zlib.decompress(self.blob[start:stop].tobytes())
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.shape[1:])

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, (int, np.integer)):
            return self._row(int(index))
        if isinstance(index, slice):
            index = np.arange(*index.indices(self.shape[0]))
        index = np.asarray(index)
        out = np.empty((index.shape[0],) + self.shape[1:], dtype=np.uint8)
        for position, row in enumerate(index.tolist()):
            out[position] = self._row(row)
        return out

    @classmethod
    def build(cls, frames: np.ndarray, *, threads: int = 12, chunk: int = 256) -> "CompressedFrames":
        rows = int(frames.shape[0])
        bounds = [(start, min(start + chunk, rows)) for start in range(0, rows, chunk)]

        def compress(bound: tuple[int, int]) -> list[bytes]:
            start, stop = bound
            block = np.ascontiguousarray(frames[start:stop])   # sequential read
            return [zlib.compress(block[i].tobytes(), cls.LEVEL) for i in range(stop - start)]

        with ThreadPoolExecutor(max_workers=threads) as pool:
            pieces = [piece for chunk_pieces in pool.map(compress, bounds) for piece in chunk_pieces]
        sizes = np.fromiter((len(piece) for piece in pieces), dtype=np.int64, count=len(pieces))
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        blob = np.frombuffer(b"".join(pieces), dtype=np.uint8)
        return cls(blob, offsets, tuple(int(v) for v in frames.shape))

    def save(self, path: Path) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, blob=self.blob, offsets=self.offsets, shape=np.asarray(self.shape, dtype=np.int64))
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "CompressedFrames":
        with np.load(path) as handle:
            return cls(np.asarray(handle["blob"]), np.asarray(handle["offsets"]), tuple(int(v) for v in handle["shape"]))


def _projected_cache_bytes(frames: np.ndarray, sample_rows: int = 256) -> int:
    rows = int(frames.shape[0])
    picks = np.linspace(0, rows - 1, min(sample_rows, rows)).astype(np.int64)
    sizes = [len(zlib.compress(np.ascontiguousarray(frames[int(i)]).tobytes(), CompressedFrames.LEVEL)) for i in picks]
    return int(np.mean(sizes) * rows)


def cache_frames(frames: np.ndarray, manifest: dict[str, Any], directory: Path, *, log=print) -> Any:
    """Return the frames source to train from: a RAM zlib cache (built once, persisted under ``directory``) or the memmap.

    ``SO_ARM101_V2_FRAME_CACHE=off`` keeps the memmap. The cache is skipped (with a message) when its
    projected size exceeds half of physical memory.
    """
    mode = os.environ.get("SO_ARM101_V2_FRAME_CACHE", "zlib")
    if mode not in ("zlib", "off", "gpu"):
        raise ValueError("SO_ARM101_V2_FRAME_CACHE must be 'zlib', 'gpu' or 'off'")
    if mode in ("off", "gpu"):
        return frames   # 'gpu': the training loop moves the (strided) rows onto the device once it exists
    key = str(manifest["frames"]["sha256"])[:16]
    path = Path(directory) / f"frames_cache_{key}.npz"
    if path.exists():
        cache = CompressedFrames.load(path)
        if cache.shape == tuple(frames.shape):
            log(f"frame cache: loaded {path.name} ({cache.nbytes / 1e9:.2f} GB in RAM)")
            return cache
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    projected = _projected_cache_bytes(frames)
    if projected > physical // 2:
        log(f"frame cache: skipped, projected {projected / 1e9:.1f} GB exceeds half of physical memory ({physical / 1e9:.1f} GB)")
        return frames
    started = time.time()
    cache = CompressedFrames.build(frames)
    Path(directory).mkdir(parents=True, exist_ok=True)
    cache.save(path)
    log(f"frame cache: built {path.name} ({cache.nbytes / 1e9:.2f} GB in RAM, {frames.nbytes / max(cache.nbytes, 1):.0f}x) in {time.time() - started:.0f}s")
    return cache


def load_device_frames(frames: np.ndarray, rows: np.ndarray, device: Any, *, log=print, block: int = 512) -> Any:
    """Copy the selected memmap rows onto ``device`` as one uint8 tensor (N, 256, 256, 3), read sequentially in blocks.

    Why (2026-09-10): appearance-randomized frames are incompressible (zlib ~2x), so the RAM cache no
    longer fits and training fell back to disk-bound random reads (10.6 steps/s). A 24 GB GPU holds
    every third frame of a 400-episode capture (12.6 GB) with room to spare; a gather on the device
    replaces the whole host read path. Pixels are the memmap's bytes, so the images handed to the
    network are identical to the pinned host path (same permute / float / div on the device).
    """
    torch = _torch()
    rows = np.asarray(rows, dtype=np.int64)
    started = time.time()
    store = torch.empty((int(rows.shape[0]),) + tuple(int(v) for v in frames.shape[1:]), dtype=torch.uint8, device=device)
    for start in range(0, rows.shape[0], block):
        picks = rows[start:start + block]
        chunk = np.ascontiguousarray(frames[picks])          # increasing rows: near-sequential disk reads
        store[start:start + picks.shape[0]].copy_(torch.from_numpy(chunk), non_blocking=False)
    if hasattr(torch, "cuda") and device.type == "cuda":
        torch.cuda.synchronize(device)
    log(f"frame store: {rows.shape[0]} rows on {device} ({store.numel() / 1e9:.2f} GB) in {time.time() - started:.0f}s")
    return store


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
    scratch_checkpoint: str | Path | None = None,
    checkpoint_interval: int = 5000,
    stop_after_steps: int | None = None,
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
    # The cache is keyed by the frames digest and shared by every run under this output root.
    frames = cache_frames(frames, manifest, Path(output_dir) / "frame_cache")
    frame_store = os.environ.get("SO_ARM101_V2_FRAME_CACHE", "zlib")
    episode_lengths = [int(item["rows"]) for item in manifest["episodes"]]
    rows = int(arrays["action_index"].shape[0])
    # Training rows: every row (stride 1, historical) or every frame_stride-th row of each episode.
    stride = int(config.frame_stride)
    kept_rows = np.arange(rows, dtype=np.int64) if stride == 1 else np.flatnonzero(np.asarray(arrays["action_index"], dtype=np.int64) % stride == 0).astype(np.int64)
    rows_train = int(kept_rows.shape[0])
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
        # frame_stride enters the identity only when it changes the sample set, so every historical digest is unchanged.
        "config": {k: v for k, v in asdict(config).items() if not ((k == "frame_stride" and v == 1) or (k == "encoder" and v == "v1"))},
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
    model = build_vision_chunked_model(config.hidden_width, config.chunk_horizon, encoder=config.encoder).to(device)
    targets = torch.from_numpy(targets_np).to(device)
    state = torch.from_numpy(state_np).to(device)
    if stride > 1:
        identity_note = f"frame_stride {stride}: {rows_train} of {rows} rows"
        print(f"training rows: {identity_note}", flush=True)
    row_map = torch.from_numpy(kept_rows)                       # sample index -> source row (identity when stride == 1)
    row_map_device = row_map.to(device)
    device_frames = load_device_frames(frames, kept_rows, device) if frame_store == "gpu" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = noise_generator(torch, config.seed)
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    scratch = Path(scratch_checkpoint) if scratch_checkpoint else None
    resumed = None
    if scratch is not None and scratch.exists():
        resumed = torch.load(scratch, map_location="cpu", weights_only=False)
        if resumed["run_digest"] != run_digest:
            raise ValueError("scratch checkpoint identity differs from requested training")
        # cpu_state_dict saves the unwrapped module; load into the same
        # object so a torch.compile wrapper does not reject the keys.
        getattr(model, "_orig_mod", model).load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        generator.set_state(resumed["generator"])
        torch.set_rng_state(resumed["torch_rng"])
        if torch.cuda.is_available() and resumed["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(resumed["cuda_rng"])
        np.random.set_state(resumed["numpy_rng"])
        random.setstate(resumed["python_rng"])

    # Image upload path. "device" (default) ships the raw uint8 block through a
    # pinned staging buffer and does the /255 and HWC->CHW on the device: a
    # quarter of the host-to-device bytes and no host float conversion. It is
    # bitwise identical to the historical "cpu" path (single float32 division
    # either way, same strides handed to the convolution; pinned by test), so
    # identities and digests are unaffected. SO_ARM101_V2_IMAGE_UPLOAD=cpu
    # restores the old path.
    image_upload = os.environ.get("SO_ARM101_V2_IMAGE_UPLOAD", "device")
    if image_upload not in ("device", "cpu"):
        raise ValueError("SO_ARM101_V2_IMAGE_UPLOAD must be 'device' or 'cpu'")
    staging = None

    def to_device_images(uint8_block: "np.ndarray") -> Any:
        nonlocal staging
        block = np.ascontiguousarray(uint8_block)
        if image_upload == "cpu":
            floats = np.asarray(block, dtype=np.float32) / np.float32(255.0)
            return torch.from_numpy(np.transpose(floats, (0, 3, 1, 2))).to(device)
        source = torch.from_numpy(block)
        if device.type == "cuda":
            if staging is None or tuple(staging.shape) != tuple(source.shape):
                staging = torch.empty(source.shape, dtype=torch.uint8).pin_memory()
            staging.copy_(source)
            source = staging
        on_device = source.to(device)
        return on_device.permute(0, 3, 1, 2).float().div_(255.0)

    def batch_images(indices: Any) -> Any:
        if device_frames is not None:
            gathered = device_frames[indices.to(device)]
            return gathered.permute(0, 3, 1, 2).float().div_(255.0)
        return to_device_images(np.asarray(frames[kept_rows[indices.numpy()]]))

    # Deterministic prefetching (see notes/vision-rung-notebook.md): the
    # per-step minibatch read is the only disk I/O in the loop and dominates
    # wall time once the frames sidecar outgrows the page cache.  Reader
    # threads perform ONLY the raw uint8 memmap reads, many steps ahead and
    # concurrently (higher effective disk queue depth); index draws stay on
    # this thread in step order, so the seeded RNG stream, every float op,
    # and the op order are unchanged — results are bitwise identical to the
    # synchronous path (pinned by test).  Identity/digests unaffected.
    prefetch_enabled = os.environ.get("SO_ARM101_V2_PREFETCH", "1") != "0" and device_frames is None
    # Defaults raised 8/6 -> 16/12 on 2026-09-09: the 37.7 GB bench sidecar is
    # 2.5x this machine's RAM, so the disk queue depth is the throughput lever.
    prefetch_depth = max(1, int(os.environ.get("SO_ARM101_V2_PREFETCH_DEPTH", "16")))
    prefetch_workers = max(1, int(os.environ.get("SO_ARM101_V2_PREFETCH_WORKERS", "12")))

    loss_trace: list[dict[str, float | int]] = []
    batch = min(config.batch_size, rows_train)
    executor = None
    pending: "deque[tuple[Any, Any]]" = deque()
    drawn = 0
    start_step = 0

    def draw_indices() -> Any:
        return torch.randperm(rows_train, generator=generator)[:batch]

    if prefetch_enabled:
        executor = ThreadPoolExecutor(max_workers=prefetch_workers)
    if resumed is not None:
        if resumed["prefetch_enabled"] != prefetch_enabled:
            raise ValueError("resume requires the same prefetch mode")
        start_step = resumed["step"]
        drawn = resumed["drawn"]
        loss_trace = resumed["loss_trace"]
        for indices in resumed["pending_indices"]:
            pending.append((indices, executor.submit(_read_frame_rows, frames, kept_rows[indices.numpy()])))

    def save_scratch(step):
        if scratch is None:
            return
        scratch.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(run_digest=run_digest, step=step, drawn=drawn,
            model=cpu_state_dict(model), optimizer=optimizer.state_dict(),
            generator=generator.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            pending_indices=[indices for indices, _ in pending],
            prefetch_enabled=prefetch_enabled, loss_trace=loss_trace,
            saved_at=time.time())
        fd, temp = tempfile.mkstemp(dir=scratch.parent, prefix=".checkpoint-", suffix=".pt")
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, scratch)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def next_batch() -> "tuple[Any, Any]":
        nonlocal drawn
        if executor is None:
            indices = draw_indices()
            return indices, None
        while drawn < config.max_steps and len(pending) < prefetch_depth:
            indices = draw_indices()
            drawn += 1
            future = executor.submit(_read_frame_rows, frames, kept_rows[indices.numpy()])
            pending.append((indices, future))
        indices, future = pending.popleft()
        return indices, future.result()

    try:
        for step in range(start_step + 1, config.max_steps + 1):
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
            source_rows = row_map_device[indices.to(device)]
            prediction = model(images, state[source_rows])
            loss = (prediction - targets[source_rows]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("vision distillation loss became nonfinite")
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == config.max_steps:
                loss_value = float(loss.item())
                loss_trace.append({"step": step, "batch_normalized_mse": loss_value})
                if on_loss is not None:
                    on_loss(step, loss_value)
            if scratch is not None and (step % checkpoint_interval == 0 or step == config.max_steps
                                        or step == stop_after_steps):
                save_scratch(step)
            if step == stop_after_steps:
                raise InterruptedError(f"requested training interruption at step {step}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    # Final full-data evaluation in minibatches.
    model.eval()
    errors = []
    with torch.inference_mode():
        for start in range(0, rows_train, 256):
            indices = torch.arange(start, min(start + 256, rows_train))
            source_rows = row_map_device[indices.to(device)]
            prediction = model(batch_images(indices), state[source_rows])
            errors.append(
                (prediction - targets[source_rows]).square().mean(dim=1).cpu().numpy()
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
    if config.encoder != "v1":
        checkpoint_payload["encoder"] = config.encoder   # conditionally present: v1 checkpoints are byte-identical to before
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
