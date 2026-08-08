"""Deterministic lead-3 full-split training and image-dependence evidence."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from so_arm101_v2.contracts import ACT_DATASET_HIGH, ACT_DATASET_LOW, JOINT_NAMES
from so_arm101_v2.data._serialization import (
    canonical_json_bytes,
    content_sha256,
    write_immutable_bytes,
    write_immutable_json,
)
from so_arm101_v2.data.samples import _episode_video_offsets

from .baselines import compute_pose_metrics
from .tiny_model import denormalize_act, normalize_act

if TYPE_CHECKING:
    from so_arm101_v2.data import DatasetInventory, SplitManifest


LEAD_STEPS = 3
CACHE_SCHEMA_VERSION = 1


class SmallModelKind(str, Enum):
    STATE_ONLY = "state_only"
    IMAGE_STATE = "image_state"


class ImageSignalStatus(str, Enum):
    USEFUL = "useful"
    USED_NOT_HELPFUL = "used_not_helpful"
    NOT_USED = "not_used"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class SmallModelConfig:
    seeds: tuple[int, ...] = (101, 202, 303)
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience_epochs: int = 7
    minimum_improvement: float = 1e-6
    num_workers: int = 4
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must contain integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.learning_rate <= 0 or self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("learning rate, batch size, and max epochs must be positive")
        if self.patience_epochs <= 0 or self.minimum_improvement < 0:
            raise ValueError("patience must be positive and minimum improvement nonnegative")
        if self.num_workers < 0 or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("invalid worker count or device")


@dataclass(frozen=True)
class Lead3TrainingCache:
    directory: Path
    metadata_path: Path
    images_path: Path
    arrays_path: Path
    dataset_digest: str
    train_count: int
    validation_count: int
    images: NDArray[np.uint8]
    episode_ids: NDArray[np.int64]
    frame_indices: NDArray[np.int64]
    split_codes: NDArray[np.uint8]
    current_states: NDArray[np.float32]
    targets: NDArray[np.float32]

    def indices(self, split: str) -> NDArray[np.int64]:
        code = {"train": 0, "validation": 1}.get(split)
        if code is None:
            raise ValueError("cache only contains train and validation")
        return np.flatnonzero(self.split_codes == code).astype(np.int64)


@dataclass(frozen=True)
class SmallModelRun:
    kind: SmallModelKind
    seed: int
    directory: Path
    checkpoint: Path
    report_json: Path
    best_epoch: int
    validation_normalized_mse: float


@dataclass(frozen=True)
class SmallModelComparison:
    directory: Path
    report_json: Path
    report_html: Path
    runs: tuple[SmallModelRun, ...]
    dataset_digest: str


def _optional_data() -> tuple[Any, Any]:
    try:
        import av
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("full-dataset caching requires the 'data' extra") from exc
    return av, pq


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("small-model training requires the 'learn' extra") from exc
    return torch


def _cache_identity(inventory: "DatasetInventory", manifests: Mapping[str, "SplitManifest"]) -> str:
    body = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_digest": inventory.dataset_digest,
        "lead_steps": LEAD_STEPS,
        "splits": {
            name: manifests[name].content_sha256 for name in ("train", "validation")
        },
        "image": {"format": "rgb24", "shape": [256, 256, 3], "dtype": "uint8"},
    }
    return content_sha256(body)[:16]


def _cache_rows(
    inventory: "DatasetInventory", manifests: Mapping[str, "SplitManifest"], offsets: Mapping[int, tuple[str, int]]
) -> dict[str, Any]:
    records = {record.episode_id: record for record in inventory.episodes}
    episode_ids: list[int] = []
    frame_indices: list[int] = []
    split_codes: list[int] = []
    global_rows: list[int] = []
    video_files: list[str] = []
    video_frames: list[int] = []
    for split, code in (("train", 0), ("validation", 1)):
        for episode_id in manifests[split].episode_ids:
            record = records[episode_id]
            video_file, video_start = offsets[episode_id]
            for frame in range(record.frame_count - LEAD_STEPS):
                episode_ids.append(episode_id)
                frame_indices.append(frame)
                split_codes.append(code)
                global_rows.append(record.global_from_index + frame)
                video_files.append(video_file)
                video_frames.append(video_start + frame)
    rows = np.asarray(global_rows, dtype=np.int64)
    return {
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "split_codes": np.asarray(split_codes, dtype=np.uint8),
        "global_rows": rows,
        "video_files": np.asarray(video_files),
        "video_frames": np.asarray(video_frames, dtype=np.int64),
        "current_states": np.asarray(inventory.states[rows], dtype=np.float32),
        "targets": np.asarray(inventory.states[rows + LEAD_STEPS], dtype=np.float32),
    }


def _sentinel_rows(count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    return tuple(sorted({0, count // 2, count - 1}))


def _decode_sentinels(
    root: Path, rows: Mapping[str, Any], row_indices: Iterable[int]
) -> dict[int, NDArray[np.uint8]]:
    av, _ = _optional_data()
    requests: dict[str, dict[int, list[int]]] = {}
    for row in row_indices:
        relative = str(rows["video_files"][row])
        frame = int(rows["video_frames"][row])
        requests.setdefault(relative, {}).setdefault(frame, []).append(int(row))
    result: dict[int, NDArray[np.uint8]] = {}
    for relative, wanted in requests.items():
        remaining = set(wanted)
        with av.open(str(root / relative)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in remaining:
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                        raise ValueError(f"unexpected decoded image from {relative}")
                    for row in wanted[frame_index]:
                        result[row] = rgb
                    remaining.remove(frame_index)
                    if not remaining:
                        break
        if remaining:
            raise ValueError(f"video {relative} is missing cache frames {sorted(remaining)}")
    return result


def _validate_cache(
    directory: Path,
    root: Path,
    inventory: "DatasetInventory",
    manifests: Mapping[str, "SplitManifest"],
) -> Lead3TrainingCache:
    metadata_path = directory / "metadata.json"
    images_path = directory / "images.npy"
    arrays_path = directory / "anchors.npz"
    if not all(path.is_file() for path in (metadata_path, images_path, arrays_path)):
        raise FileNotFoundError("lead-3 cache is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_digest") != inventory.dataset_digest:
        raise ValueError("lead-3 cache dataset digest mismatch")
    if metadata.get("identity") != _cache_identity(inventory, manifests):
        raise ValueError("lead-3 cache split/configuration mismatch")
    source_hashes = {item.path: item.sha256 for item in inventory.files if item.role == "wrist_video"}
    if metadata.get("source_video_sha256") != source_hashes:
        raise ValueError("lead-3 cache source video hashes mismatch")
    images = np.load(images_path, mmap_mode="r", allow_pickle=False)
    arrays = np.load(arrays_path, allow_pickle=False)
    count = int(metadata["sample_count"])
    if images.shape != (count, 256, 256, 3) or images.dtype != np.uint8:
        raise ValueError("lead-3 cache image shape or dtype mismatch")
    required = {
        "episode_ids": ((count,), np.dtype(np.int64)),
        "frame_indices": ((count,), np.dtype(np.int64)),
        "split_codes": ((count,), np.dtype(np.uint8)),
        "current_states": ((count, 6), np.dtype(np.float32)),
        "targets": ((count, 6), np.dtype(np.float32)),
        "video_files": ((count,), None),
        "video_frames": ((count,), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in required.items():
        if name not in arrays or arrays[name].shape != shape or (dtype is not None and arrays[name].dtype != dtype):
            raise ValueError(f"lead-3 cache anchor array {name!r} is invalid")
    if not np.all(np.isfinite(arrays["current_states"])) or not np.all(np.isfinite(arrays["targets"])):
        raise ValueError("lead-3 cache contains nonfinite state data")
    rows = {name: arrays[name] for name in arrays.files}
    decoded = _decode_sentinels(root, rows, metadata["sentinel_rows"])
    for row, source in decoded.items():
        if not np.array_equal(images[row], source):
            raise ValueError(f"lead-3 cache RGB sentinel mismatch at row {row}")
    split_codes = arrays["split_codes"]
    train_count = int(np.sum(split_codes == 0))
    validation_count = int(np.sum(split_codes == 1))
    if train_count != metadata["train_count"] or validation_count != metadata["validation_count"]:
        raise ValueError("lead-3 cache split counts mismatch")
    return Lead3TrainingCache(
        directory=directory,
        metadata_path=metadata_path,
        images_path=images_path,
        arrays_path=arrays_path,
        dataset_digest=inventory.dataset_digest,
        train_count=train_count,
        validation_count=validation_count,
        images=images,
        episode_ids=arrays["episode_ids"],
        frame_indices=arrays["frame_indices"],
        split_codes=split_codes,
        current_states=arrays["current_states"],
        targets=arrays["targets"],
    )


def build_lead3_training_cache(
    dataset_root: str | Path,
    inventory: "DatasetInventory",
    manifests: Mapping[str, "SplitManifest"],
    cache_dir: str | Path,
) -> Lead3TrainingCache:
    """Build or validate the train/validation-only RGB lead-3 cache."""
    if not {"train", "validation"}.issubset(manifests):
        raise ValueError("train and validation manifests are required")
    for name in ("train", "validation"):
        if manifests[name].dataset_digest != inventory.dataset_digest:
            raise ValueError("split manifest dataset digest mismatch")
    root = Path(dataset_root).resolve()
    _, pq = _optional_data()
    offsets = _episode_video_offsets(root, pq, inventory.fps)
    identity = _cache_identity(inventory, manifests)
    directory = Path(cache_dir) / f"lead3_rgb_v1.{inventory.dataset_digest[:12]}.{identity}"
    if directory.exists():
        return _validate_cache(directory, root, inventory, manifests)

    rows = _cache_rows(inventory, manifests, offsets)
    count = len(rows["episode_ids"])
    train_count = int(np.sum(rows["split_codes"] == 0))
    validation_count = int(np.sum(rows["split_codes"] == 1))
    if train_count != 33125 or validation_count != 3852:
        raise ValueError(
            f"unexpected lead-3 cache counts: train={train_count}, validation={validation_count}"
        )
    building = directory.with_name(directory.name + f".building.{os.getpid()}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        images = np.lib.format.open_memmap(
            building / "images.npy", mode="w+", dtype=np.uint8, shape=(count, 256, 256, 3)
        )
        requests: dict[str, dict[int, int]] = {}
        for row, (relative, frame) in enumerate(zip(rows["video_files"], rows["video_frames"], strict=True)):
            mapping = requests.setdefault(str(relative), {})
            if int(frame) in mapping:
                raise ValueError("duplicate source video frame in lead-3 cache")
            mapping[int(frame)] = row
        av, _ = _optional_data()
        filled = 0
        for relative, wanted in sorted(requests.items()):
            remaining = set(wanted)
            with av.open(str(root / relative)) as container:
                for frame_index, frame in enumerate(container.decode(video=0)):
                    row = wanted.get(frame_index)
                    if row is not None:
                        rgb = frame.to_ndarray(format="rgb24")
                        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                            raise ValueError(f"unexpected RGB frame from {relative}")
                        images[row] = rgb
                        filled += 1
                        remaining.remove(frame_index)
                        if not remaining:
                            break
            if remaining:
                raise ValueError(f"video {relative} is missing requested cache frames")
        if filled != count:
            raise ValueError(f"filled {filled} cache rows, expected {count}")
        images.flush()
        del images
        np.savez(
            building / "anchors.npz",
            episode_ids=rows["episode_ids"],
            frame_indices=rows["frame_indices"],
            split_codes=rows["split_codes"],
            current_states=rows["current_states"],
            targets=rows["targets"],
            video_files=rows["video_files"],
            video_frames=rows["video_frames"],
        )
        source_hashes = {item.path: item.sha256 for item in inventory.files if item.role == "wrist_video"}
        body: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity": identity,
            "dataset_digest": inventory.dataset_digest,
            "lead_steps": LEAD_STEPS,
            "sample_count": count,
            "train_count": train_count,
            "validation_count": validation_count,
            "excluded_split": "test",
            "source_video_sha256": source_hashes,
            "image": {"format": "rgb24", "shape": [256, 256, 3], "dtype": "uint8"},
            "sentinel_rows": list(_sentinel_rows(count)),
            "split_content_sha256": {
                name: manifests[name].content_sha256 for name in ("train", "validation")
            },
        }
        body["content_sha256"] = content_sha256(body)
        (building / "metadata.json").write_bytes(canonical_json_bytes(body))
        building.rename(directory)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise
    return _validate_cache(directory, root, inventory, manifests)


def build_small_model(kind: SmallModelKind | str, output_bias: NDArray[np.float32]) -> Any:
    torch = _torch()
    selected = SmallModelKind(kind)

    class StateOnlyNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(6, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 6),
            )
            with torch.no_grad():
                self.network[-1].bias.copy_(torch.as_tensor(output_bias, dtype=torch.float32))

        def forward(self, image: Any, state: Any) -> Any:
            del image
            return self.network(state)

    class ImageStateNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=8, stride=8), torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, kernel_size=4, stride=4), torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),
            )
            self.state_encoder = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU())
            self.head = torch.nn.Sequential(
                torch.nn.Linear(544, 128), torch.nn.ReLU(), torch.nn.Linear(128, 6)
            )
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.as_tensor(output_bias, dtype=torch.float32))

        def forward(self, image: Any, state: Any) -> Any:
            features = self.image_encoder(image).flatten(1)
            return self.head(torch.cat((features, self.state_encoder(state)), dim=1))

    return StateOnlyNetwork() if selected is SmallModelKind.STATE_ONLY else ImageStateNetwork()


def _device(torch: Any, requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")


def _git_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout)
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _diagnostic_indices(cache: Lead3TrainingCache) -> NDArray[np.int64]:
    validation = cache.indices("validation")
    chosen: list[int] = []
    for episode in sorted(set(cache.episode_ids[validation].tolist())):
        local = validation[cache.episode_ids[validation] == episode]
        episode_chosen: list[int] = []
        for fraction in (0.25, 0.5, 0.75):
            episode_chosen.append(int(local[round((len(local) - 1) * fraction)]))
        errors = np.linalg.norm(
            cache.targets[local].astype(np.float64) - cache.current_states[local].astype(np.float64), axis=1
        )
        for ranked in np.argsort(errors)[::-1]:
            candidate = int(local[int(ranked)])
            if candidate not in episode_chosen:
                episode_chosen.append(candidate)
                break
        chosen.extend(episode_chosen)
    result = np.asarray(chosen, dtype=np.int64)
    if result.shape != (40,) or len(set(result.tolist())) != 40:
        raise ValueError("fixed validation diagnostic batch is not exactly 40 unique anchors")
    return result


def _batch_predictions(
    torch: Any,
    model: Any,
    cache: Lead3TrainingCache,
    indices: NDArray[np.int64],
    device: Any,
    batch_size: int,
    *,
    image_indices: NDArray[np.int64] | None = None,
    black_images: bool = False,
) -> NDArray[np.float32]:
    outputs: list[NDArray[np.float32]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            states = torch.from_numpy(normalize_act(cache.current_states[rows])).to(device)
            if not hasattr(model, "image_encoder"):
                images = torch.empty((len(rows), 0), dtype=torch.float32, device=device)
            elif black_images:
                images = torch.zeros((len(rows), 3, 256, 256), dtype=torch.float32, device=device)
            else:
                source = rows if image_indices is None else image_indices[start:start + len(rows)]
                array = np.transpose(np.asarray(cache.images[source], dtype=np.float32) / np.float32(255.0), (0, 3, 1, 2))
                images = torch.from_numpy(array).to(device)
            predicted = model(images, states)
            outputs.append(predicted.detach().cpu().numpy().astype(np.float32))
    return denormalize_act(np.concatenate(outputs, axis=0))


def _per_episode_metrics(
    cache: Lead3TrainingCache, indices: NDArray[np.int64], predictions: NDArray[np.float32]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    episodes = cache.episode_ids[indices]
    for episode in sorted(set(episodes.tolist())):
        mask = episodes == episode
        difference = predictions[mask].astype(np.float64) - cache.targets[indices[mask]].astype(np.float64)
        result[str(episode)] = {
            "samples": int(np.sum(mask)),
            "normalized_mse": float(np.mean(np.square(
                difference / ((ACT_DATASET_HIGH - ACT_DATASET_LOW) / 2.0)
            ))),
            "act_l2_mean": float(np.mean(np.linalg.norm(difference, axis=1))),
        }
    return result


def _run_directory(base: Path, cache: Lead3TrainingCache, config: SmallModelConfig, kind: SmallModelKind, seed: int) -> Path:
    identity = content_sha256({"config": asdict(config), "kind": kind.value, "seed": seed})[:12]
    return base / f"{kind.value}.lead3.seed{seed}.v1.{cache.dataset_digest[:12]}.{identity}"


def _load_run(directory: Path) -> SmallModelRun:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    return SmallModelRun(
        kind=SmallModelKind(report["model_kind"]), seed=int(report["seed"]), directory=directory,
        checkpoint=directory / "model.pt", report_json=directory / "report.json",
        best_epoch=int(report["best_epoch"]),
        validation_normalized_mse=float(report["metrics"]["validation"]["normalized_mse"]),
    )


def _train_one(
    cache: Lead3TrainingCache,
    output_dir: Path,
    kind: SmallModelKind,
    seed: int,
    config: SmallModelConfig,
    train_mean: NDArray[np.float32],
) -> SmallModelRun:
    directory = _run_directory(output_dir, cache, config, kind, seed)
    if (directory / "report.json").is_file() and (directory / "model.pt").is_file():
        return _load_run(directory)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(torch, config.device)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    model = build_small_model(kind, normalize_act(train_mean)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    train_rows = cache.indices("train")
    validation_rows = cache.indices("validation")
    diagnostic_rows = _diagnostic_indices(cache)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    first_gradients: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(train_rows)
        model.train()
        total_squared = 0.0
        total_values = 0
        for start in range(0, len(order), config.batch_size):
            rows = order[start:start + config.batch_size]
            states = torch.from_numpy(normalize_act(cache.current_states[rows])).to(device)
            targets = torch.from_numpy(normalize_act(cache.targets[rows])).to(device)
            if kind is SmallModelKind.IMAGE_STATE:
                image_np = np.transpose(
                    np.asarray(cache.images[rows], dtype=np.float32) / np.float32(255.0), (0, 3, 1, 2)
                )
                images = torch.from_numpy(image_np).to(device)
            else:
                images = torch.empty((len(rows), 0), dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(images, states)
            loss = criterion(predicted, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("small-model training loss became nonfinite")
            loss.backward()
            if epoch == 1 and start == 0:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        first_gradients[name] = float(parameter.grad.norm().item())
                if not first_gradients or not all(np.isfinite(list(first_gradients.values()))):
                    raise RuntimeError("small-model gradients are missing or nonfinite")
            optimizer.step()
            total_squared += float(loss.detach().item()) * int(targets.numel())
            total_values += int(targets.numel())
        validation_predictions = _batch_predictions(
            torch, model, cache, validation_rows, device, config.batch_size,
            black_images=kind is SmallModelKind.STATE_ONLY,
        )
        val_normalized = normalize_act(validation_predictions)
        val_targets = normalize_act(cache.targets[validation_rows])
        validation_loss = float(np.mean(np.square(val_normalized - val_targets)))
        diagnostic_predictions = _batch_predictions(
            torch, model, cache, diagnostic_rows, device, config.batch_size,
            black_images=kind is SmallModelKind.STATE_ONLY,
        )
        history.append({
            "epoch": epoch,
            "train_normalized_mse": total_squared / total_values,
            "validation_normalized_mse": validation_loss,
            "diagnostic_predictions_act": diagnostic_predictions.tolist(),
        })
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience_epochs:
            break
    if best_state is None:
        raise RuntimeError("small-model training retained no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    split_metrics: dict[str, Any] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        predictions = _batch_predictions(
            torch, model, cache, rows, device, config.batch_size,
            black_images=kind is SmallModelKind.STATE_ONLY,
        )
        metrics = compute_pose_metrics(cache.current_states[rows], cache.targets[rows], predictions)
        if split == "validation":
            metrics["per_episode"] = _per_episode_metrics(cache, rows, predictions)
        split_metrics[split] = metrics
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "model_kind": kind.value,
        "seed": seed,
        "config": asdict(config),
        "dataset_digest": cache.dataset_digest,
        "train_mean_target_act": train_mean.tolist(),
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    write_immutable_bytes(
        directory / "model.pt",
        buffer.getvalue(),
        conflict_message=f"immutable small-model checkpoint differs: {directory / 'model.pt'}",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "claim": "lead3_train_and_validation_only_no_test_evaluation",
        "dataset_digest": cache.dataset_digest,
        "cache_metadata_sha256": hashlib.sha256(cache.metadata_path.read_bytes()).hexdigest(),
        "model_kind": kind.value,
        "seed": seed,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_mean_target_act": train_mean.tolist(),
        "metrics": split_metrics,
        "first_batch_gradient_norms": first_gradients,
        "diagnostic_references": [
            {"episode_id": int(cache.episode_ids[row]), "frame_index": int(cache.frame_indices[row]), "lead_steps": 3}
            for row in diagnostic_rows
        ],
        "history": history,
        "joint_order": list(JOINT_NAMES),
        "provenance": {
            "git": _git_provenance(), "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "test_evaluated": False,
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(directory / "report.json", report)
    return _load_run(directory)


def _development_baselines(cache: Lead3TrainingCache) -> tuple[NDArray[np.float32], dict[str, Any]]:
    train = cache.indices("train")
    validation = cache.indices("validation")
    mean = np.mean(cache.targets[train], axis=0, dtype=np.float64).astype(np.float32)
    report: dict[str, Any] = {
        "mean_fit_split": "train",
        "lead_steps": 3,
        "train_mean_target": mean.tolist(),
        "splits": {},
        "test_evaluated": False,
    }
    for name, rows in (("train", train), ("validation", validation)):
        report["splits"][name] = {
            "samples": len(rows),
            "current_pose": compute_pose_metrics(cache.current_states[rows], cache.targets[rows], cache.current_states[rows]),
            "train_mean_target": compute_pose_metrics(
                cache.current_states[rows], cache.targets[rows], np.broadcast_to(mean, cache.targets[rows].shape)
            ),
        }
    return mean, report


def _comparison_html(report: Mapping[str, Any]) -> str:
    rows = []
    for run in report["runs"]:
        rows.append(
            f"<tr><td>{run['model_kind']}</td><td>{run['seed']}</td><td>{run['best_epoch']}</td>"
            f"<td>{run['validation_normalized_mse']:.8g}</td></tr>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'><title>Small-model comparison</title>
<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #999;padding:6px}</style>
</head><body><h1>Lead-3 small-model comparison</h1><p>Training and validation only; test was not evaluated.</p>
<table><tr><th>model</th><th>seed</th><th>best epoch</th><th>validation normalized MSE</th></tr>""" + "".join(rows) + "</table></body></html>"


def train_small_models(
    cache: Lead3TrainingCache,
    output_dir: str | Path,
    *,
    config: SmallModelConfig | None = None,
) -> SmallModelComparison:
    """Train state-only and image-plus-state models for all configured seeds."""
    config = config or SmallModelConfig()
    base = Path(output_dir)
    train_mean, baselines = _development_baselines(cache)
    runs = tuple(
        _train_one(cache, base, kind, seed, config, train_mean)
        for kind in (SmallModelKind.STATE_ONLY, SmallModelKind.IMAGE_STATE)
        for seed in config.seeds
    )
    identity = content_sha256({"config": asdict(config), "dataset_digest": cache.dataset_digest})[:12]
    directory = base / f"comparison.lead3.v1.{cache.dataset_digest[:12]}.{identity}"
    run_rows = []
    for run in runs:
        payload = json.loads(run.report_json.read_text(encoding="utf-8"))
        run_rows.append({
            "model_kind": run.kind.value,
            "seed": run.seed,
            "best_epoch": run.best_epoch,
            "validation_normalized_mse": run.validation_normalized_mse,
            "report_content_sha256": payload["content_sha256"],
            "checkpoint": str(run.checkpoint.resolve()),
        })
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_digest": cache.dataset_digest,
        "lead_steps": 3,
        "config": asdict(config),
        "development_baselines": baselines,
        "runs": run_rows,
        "test_evaluated": False,
    }
    report["content_sha256"] = content_sha256(report)
    directory.mkdir(parents=True, exist_ok=True)
    write_immutable_json(directory / "comparison.json", report)
    html = _comparison_html(report).encode("utf-8")
    html_path = directory / "comparison.html"
    write_immutable_bytes(
        html_path, html, conflict_message=f"immutable comparison HTML differs: {html_path}"
    )
    return SmallModelComparison(directory, directory / "comparison.json", html_path, runs, cache.dataset_digest)


def load_small_model_comparison(path: str | Path) -> SmallModelComparison:
    """Load a comparison and its immutable run reports/checkpoints."""
    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    stated = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("small-model comparison content hash mismatch")
    runs: list[SmallModelRun] = []
    for item in payload["runs"]:
        checkpoint = Path(item["checkpoint"])
        directory = checkpoint.parent
        run = _load_run(directory)
        if not checkpoint.is_file() or run.seed != int(item["seed"]) or run.kind.value != item["model_kind"]:
            raise ValueError("small-model comparison references an incompatible run")
        runs.append(run)
    return SmallModelComparison(
        report_path.parent, report_path, report_path.with_name("comparison.html"),
        tuple(runs), str(payload["dataset_digest"]),
    )


def cross_episode_image_indices(cache: Lead3TrainingCache, indices: NDArray[np.int64]) -> NDArray[np.int64]:
    """Map every anchor to a deterministic image from the next validation episode."""
    episodes = sorted(set(cache.episode_ids[indices].tolist()))
    by_episode = {episode: indices[cache.episode_ids[indices] == episode] for episode in episodes}
    donor = {episode: episodes[(position + 1) % len(episodes)] for position, episode in enumerate(episodes)}
    result = np.empty_like(indices)
    positions = {int(row): position for position, row in enumerate(indices.tolist())}
    for episode in episodes:
        anchors = by_episode[episode]
        donors = by_episode[donor[episode]]
        for local_index, row in enumerate(anchors):
            fraction = local_index / max(len(anchors) - 1, 1)
            source = donors[round(fraction * (len(donors) - 1))]
            result[positions[int(row)]] = source
    if np.any(cache.episode_ids[result] == cache.episode_ids[indices]):
        raise RuntimeError("cross-episode image derangement retained an original episode")
    return result


def _bootstrap_interval(differences: NDArray[np.float64], *, draws: int = 10_000) -> tuple[float, float]:
    if differences.ndim != 2:
        raise ValueError("bootstrap differences must have shape (seeds, episodes)")
    rng = np.random.default_rng(101)
    values = np.empty(draws, dtype=np.float64)
    episode_count = differences.shape[1]
    for draw in range(draws):
        sample = rng.integers(0, episode_count, size=episode_count)
        values[draw] = float(np.mean(differences[:, sample]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def classify_image_signal(
    clean: NDArray[np.float64],
    state_only: NDArray[np.float64],
    blank: NDArray[np.float64],
    shuffled: NDArray[np.float64],
) -> tuple[ImageSignalStatus, dict[str, Any]]:
    """Classify paired seed-by-episode normalized-MSE evidence."""
    shapes = {array.shape for array in (clean, state_only, blank, shuffled)}
    if len(shapes) != 1 or clean.ndim != 2 or not all(np.all(np.isfinite(x)) for x in (clean, state_only, blank, shuffled)):
        raise ValueError("image-signal arrays must be matching finite (seeds, episodes) matrices")
    clean_seed = np.mean(clean, axis=1)
    state_seed = np.mean(state_only, axis=1)
    blank_seed = np.mean(blank, axis=1)
    shuffled_seed = np.mean(shuffled, axis=1)
    state_gain = (state_seed - clean_seed) / state_seed
    blank_damage = (blank_seed - clean_seed) / clean_seed
    shuffled_damage = (shuffled_seed - clean_seed) / clean_seed
    intervals = {
        "state_only_minus_clean": _bootstrap_interval(state_only - clean),
        "blank_minus_clean": _bootstrap_interval(blank - clean),
        "shuffled_minus_clean": _bootstrap_interval(shuffled - clean),
    }
    useful = bool(
        np.mean(state_gain) >= 0.05 and np.all(state_gain > 0)
        and np.all(blank_damage >= 0.05) and np.all(shuffled_damage >= 0.05)
        and all(interval[0] > 0 for interval in intervals.values())
    )
    ablation_change = np.maximum(np.abs(blank_damage), np.abs(shuffled_damage))
    if useful:
        status = ImageSignalStatus.USEFUL
    elif np.all(ablation_change < 0.05):
        status = ImageSignalStatus.NOT_USED
    elif np.mean(state_gain) <= 0 and np.any(ablation_change >= 0.05):
        status = ImageSignalStatus.USED_NOT_HELPFUL
    else:
        status = ImageSignalStatus.INCONCLUSIVE
    return status, {
        "state_only_relative_gain_by_seed": state_gain.tolist(),
        "blank_relative_damage_by_seed": blank_damage.tolist(),
        "shuffled_relative_damage_by_seed": shuffled_damage.tolist(),
        "episode_bootstrap_95ci": {name: list(value) for name, value in intervals.items()},
    }


def _episode_mse_matrix(
    cache: Lead3TrainingCache, indices: NDArray[np.int64], predictions: list[NDArray[np.float32]]
) -> tuple[list[int], NDArray[np.float64]]:
    episodes = sorted(set(cache.episode_ids[indices].tolist()))
    matrix = np.empty((len(predictions), len(episodes)), dtype=np.float64)
    for seed_index, predicted in enumerate(predictions):
        for episode_index, episode in enumerate(episodes):
            mask = cache.episode_ids[indices] == episode
            delta = predicted[mask].astype(np.float64) - cache.targets[indices[mask]].astype(np.float64)
            matrix[seed_index, episode_index] = float(np.mean(np.square(
                delta / ((ACT_DATASET_HIGH - ACT_DATASET_LOW) / 2.0)
            )))
    return episodes, matrix


def _load_model_from_run(run: SmallModelRun, device: Any) -> Any:
    torch = _torch()
    payload = torch.load(run.checkpoint, map_location="cpu", weights_only=True)
    mean = np.asarray(payload["train_mean_target_act"], dtype=np.float32)
    model = build_small_model(run.kind, normalize_act(mean))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def evaluate_image_ablation(
    cache: Lead3TrainingCache,
    comparison: SmallModelComparison,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate clean, black, and cross-episode images on validation only."""
    torch = _torch()
    device = _device(torch, "auto")
    validation = cache.indices("validation")
    shuffled_indices = cross_episode_image_indices(cache, validation)
    state_predictions: list[NDArray[np.float32]] = []
    clean_predictions: list[NDArray[np.float32]] = []
    blank_predictions: list[NDArray[np.float32]] = []
    shuffled_predictions: list[NDArray[np.float32]] = []
    seeds = sorted({run.seed for run in comparison.runs})
    for seed in seeds:
        state_run = next(run for run in comparison.runs if run.seed == seed and run.kind is SmallModelKind.STATE_ONLY)
        image_run = next(run for run in comparison.runs if run.seed == seed and run.kind is SmallModelKind.IMAGE_STATE)
        state_model = _load_model_from_run(state_run, device)
        image_model = _load_model_from_run(image_run, device)
        state_predictions.append(_batch_predictions(torch, state_model, cache, validation, device, 256, black_images=True))
        clean_predictions.append(_batch_predictions(torch, image_model, cache, validation, device, 256))
        blank_predictions.append(_batch_predictions(torch, image_model, cache, validation, device, 256, black_images=True))
        shuffled_predictions.append(_batch_predictions(
            torch, image_model, cache, validation, device, 256, image_indices=shuffled_indices
        ))
    episodes, state_matrix = _episode_mse_matrix(cache, validation, state_predictions)
    _, clean_matrix = _episode_mse_matrix(cache, validation, clean_predictions)
    _, blank_matrix = _episode_mse_matrix(cache, validation, blank_predictions)
    _, shuffled_matrix = _episode_mse_matrix(cache, validation, shuffled_predictions)
    status, evidence = classify_image_signal(clean_matrix, state_matrix, blank_matrix, shuffled_matrix)
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_digest": cache.dataset_digest,
        "lead_steps": 3,
        "seeds": seeds,
        "validation_episodes": episodes,
        "status": status.value,
        "evidence": evidence,
        "normalized_mse_by_seed_episode": {
            "state_only": state_matrix.tolist(), "clean": clean_matrix.tolist(),
            "blank": blank_matrix.tolist(), "shuffled": shuffled_matrix.tolist(),
        },
        "test_evaluated": False,
    }
    report["content_sha256"] = content_sha256(report)
    destination = Path(output_dir) / f"image_ablation.lead3.v1.{cache.dataset_digest[:12]}.json"
    write_immutable_json(destination, report)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Image ablation</title></head>"
        f"<body><h1>Image-dependence result: {status.value}</h1>"
        "<p>Paired validation evidence only; test was not evaluated.</p>"
        f"<pre>{json.dumps(evidence, indent=2, sort_keys=True)}</pre></body></html>"
    ).encode("utf-8")
    html_path = destination.with_suffix(".html")
    write_immutable_bytes(
        html_path, html, conflict_message=f"immutable image-ablation HTML differs: {html_path}"
    )
    return report


__all__ = [
    "ImageSignalStatus",
    "Lead3TrainingCache",
    "SmallModelComparison",
    "SmallModelConfig",
    "SmallModelKind",
    "SmallModelRun",
    "build_lead3_training_cache",
    "build_small_model",
    "classify_image_signal",
    "cross_episode_image_indices",
    "evaluate_image_ablation",
    "load_small_model_comparison",
    "train_small_models",
]
