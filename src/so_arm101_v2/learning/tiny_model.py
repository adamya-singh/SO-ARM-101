"""A deliberately tiny image-plus-state memorization gate."""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from so_arm101_v2.contracts import ACT_DATASET_HIGH, ACT_DATASET_LOW, JOINT_NAMES
from so_arm101_v2.data import SampleReference, load_future_state_samples
from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

if TYPE_CHECKING:
    from so_arm101_v2.data import DatasetInventory, FutureStateSample


MEMORIZATION_REFERENCES = (
    SampleReference(0, 68, 3), SampleReference(0, 137, 3),
    SampleReference(0, 205, 3), SampleReference(0, 274, 3),
    SampleReference(1, 68, 3), SampleReference(1, 136, 3),
    SampleReference(1, 203, 3), SampleReference(1, 271, 3),
    SampleReference(2, 87, 3), SampleReference(2, 174, 3),
    SampleReference(2, 261, 3), SampleReference(2, 348, 3),
    SampleReference(3, 82, 3), SampleReference(3, 163, 3),
    SampleReference(3, 245, 3), SampleReference(3, 326, 3),
)


@dataclass(frozen=True)
class TinyModelConfig:
    lead_steps: int = 3
    sample_count: int = 16
    seed: int = 101
    learning_rate: float = 1e-3
    max_steps: int = 5000
    normalized_mse_threshold: float = 1e-6
    max_act_error_threshold: float = 0.01
    baseline_improvement_factor: float = 100.0

    def __post_init__(self) -> None:
        if self.lead_steps != 3:
            raise ValueError("the v1 tiny gate is fixed to lead 3")
        if not 1 <= self.sample_count <= len(MEMORIZATION_REFERENCES):
            raise ValueError("sample_count must be between 1 and 16")
        if self.max_steps <= 0 or self.learning_rate <= 0:
            raise ValueError("max_steps and learning_rate must be positive")


@dataclass(frozen=True)
class TinyOverfitResult:
    passed: bool
    steps: int
    normalized_mse: float
    max_act_error: float
    directory: Path
    checkpoint: Path
    report_json: Path
    report_html: Path


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tiny memorization requires the 'learn' extra") from exc
    return torch


def normalize_act(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        2.0 * (values - ACT_DATASET_LOW) / (ACT_DATASET_HIGH - ACT_DATASET_LOW) - 1.0,
        dtype=np.float32,
    )


def denormalize_act(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        ACT_DATASET_LOW
        + (values + 1.0) * 0.5 * (ACT_DATASET_HIGH - ACT_DATASET_LOW),
        dtype=np.float32,
    )


def build_memorization_subset(
    dataset_root: str | Path,
    inventory: "DatasetInventory",
    *,
    sample_count: int = 16,
) -> tuple["FutureStateSample", ...]:
    if not 1 <= sample_count <= 16:
        raise ValueError("sample_count must be between 1 and 16")
    return load_future_state_samples(
        dataset_root, MEMORIZATION_REFERENCES[:sample_count], inventory=inventory
    )


def _load_baseline(path: Path, dataset_digest: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"baseline report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_digest") != dataset_digest:
        raise ValueError("baseline report dataset digest mismatch")
    stated = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("baseline report content hash mismatch")
    if "3" not in payload.get("leads", {}):
        raise ValueError("baseline report has no lead-3 result")
    return payload


def _build_model(torch: Any, output_bias: np.ndarray) -> Any:
    class TinyImageStateNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=8, stride=8),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, kernel_size=4, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            )
            self.state_encoder = torch.nn.Sequential(
                torch.nn.Linear(6, 32), torch.nn.ReLU()
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(544, 128), torch.nn.ReLU(), torch.nn.Linear(128, 6)
            )
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.as_tensor(output_bias, dtype=torch.float32))

        def forward(self, image: Any, state: Any) -> Any:
            image_features = self.image_encoder(image).flatten(1)
            state_features = self.state_encoder(state)
            return self.head(torch.cat((image_features, state_features), dim=1))

    return TinyImageStateNetwork()


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable tiny-model artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _image_uri(sample: "FutureStateSample") -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tiny HTML reporting requires the 'visualize' extra") from exc
    stream = io.BytesIO()
    Image.fromarray(sample.raw_image, mode="RGB").save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _html(samples: tuple["FutureStateSample", ...], report: dict[str, Any]) -> str:
    rows = []
    for item in report["predictions"]:
        ref = item["reference"]
        rows.append(
            f"<tr><td>{ref['episode_id']}:{ref['frame_index']}</td>"
            f"<td>{item['act_l2_error']:.8f}</td>"
            f"<td>{item['max_abs_act_error']:.8f}</td></tr>"
        )
    first = samples[0]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Tiny memorization gate</title>
<style>body{{font-family:system-ui;background:#10151c;color:#e7edf5;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #445;padding:6px}}img{{width:320px}}.pass{{color:#75d18a}}.fail{{color:#ff6b6b}}</style></head>
<body><h1>Tiny image + state memorization gate</h1><h2 class="{'pass' if report['passed'] else 'fail'}">{'PASS' if report['passed'] else 'FAIL'}</h2>
<p>Dataset {report['dataset_digest']} · {report['sample_count']} samples · {report['steps']} steps<br>
Normalized MSE {report['normalized_mse']:.10g} · max ACT error {report['max_act_error']:.10g}</p>
<img src="{_image_uri(first)}"><p>Fixed example: episode {first.reference.episode_id}, frame {first.reference.frame_index}</p>
<table><tr><th>sample</th><th>ACT L2 error</th><th>max absolute ACT error</th></tr>{''.join(rows)}</table>
<p>This gate proves plumbing and memorization only. It is not held-out or robot-task evidence.</p></body></html>"""


def _result_from_report(directory: Path, report: dict[str, Any]) -> TinyOverfitResult:
    return TinyOverfitResult(
        passed=bool(report["passed"]),
        steps=int(report["steps"]),
        normalized_mse=float(report["normalized_mse"]),
        max_act_error=float(report["max_act_error"]),
        directory=directory,
        checkpoint=directory / "tiny_model.pt",
        report_json=directory / "report.json",
        report_html=directory / "report.html",
    )


def run_tiny_overfit(
    dataset_root: str | Path,
    inventory: "DatasetInventory",
    baseline_report: str | Path,
    output_dir: str | Path,
    *,
    config: TinyModelConfig | None = None,
) -> TinyOverfitResult:
    """Train or replay the deterministic tiny memorization acceptance gate."""
    config = config or TinyModelConfig()
    baseline = _load_baseline(Path(baseline_report), inventory.dataset_digest)
    directory = Path(output_dir) / (
        f"memorize{config.sample_count}.lead3.seed{config.seed}.v1."
        f"{inventory.dataset_digest[:12]}"
    )
    existing_report = directory / "report.json"
    if existing_report.is_file():
        report = json.loads(existing_report.read_text(encoding="utf-8"))
        if report.get("dataset_digest") != inventory.dataset_digest or report.get("config") != asdict(config):
            raise FileExistsError("existing tiny-model report has incompatible identity")
        return _result_from_report(directory, report)

    torch = _torch()
    torch.manual_seed(config.seed)
    # This acceptance run is CPU-only.  Its operators are deterministic with a
    # fixed seed and one thread; enabling PyTorch's global switch also imports
    # the optional CUDA/Triton compiler in recent releases, even on this CPU
    # path, and can make an otherwise dependency-free CPU run fail at import.
    torch.set_num_threads(1)
    np.random.seed(config.seed)
    samples = build_memorization_subset(
        dataset_root, inventory, sample_count=config.sample_count
    )
    images_np = np.stack([sample.model_image for sample in samples])
    states_np = np.stack([sample.current_state for sample in samples])
    targets_np = np.stack([sample.future_target for sample in samples])
    states_normalized = normalize_act(states_np)
    targets_normalized = normalize_act(targets_np)
    mean_target = np.asarray(
        baseline["leads"]["3"]["train_mean_target"], dtype=np.float32
    )
    mean_target_normalized = normalize_act(mean_target)

    images = torch.from_numpy(images_np)
    states = torch.from_numpy(states_normalized)
    targets = torch.from_numpy(targets_normalized)
    model = _build_model(torch, mean_target_normalized)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    current_mse = float(np.mean(np.square(states_normalized - targets_normalized)))
    mean_mse = float(np.mean(np.square(
        np.broadcast_to(mean_target_normalized, targets_normalized.shape) - targets_normalized
    )))
    loss_trace: list[dict[str, float | int]] = []
    image_gradient_norm = 0.0
    state_gradient_norm = 0.0
    passed = False
    predictions_normalized = np.empty_like(targets_normalized)
    normalized_mse = math.inf
    max_act_error = math.inf
    steps = 0
    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(images, states)
        loss = criterion(predicted, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("tiny-model loss became nonfinite")
        loss.backward()
        if step == 1:
            image_gradient_norm = float(model.image_encoder[0].weight.grad.norm().item())
            state_gradient_norm = float(model.state_encoder[0].weight.grad.norm().item())
        optimizer.step()
        with torch.inference_mode():
            predicted_after = model(images, states)
        predictions_normalized = predicted_after.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(predictions_normalized)):
            raise RuntimeError("tiny-model predictions became nonfinite")
        normalized_mse = float(np.mean(np.square(predictions_normalized - targets_normalized)))
        predictions_act = denormalize_act(predictions_normalized)
        max_act_error = float(np.max(np.abs(predictions_act - targets_np)))
        steps = step
        if step == 1 or step % 100 == 0:
            loss_trace.append({"step": step, "normalized_mse": normalized_mse})
        passed = bool(
            normalized_mse <= config.normalized_mse_threshold
            and max_act_error <= config.max_act_error_threshold
            and normalized_mse <= current_mse / config.baseline_improvement_factor
            and normalized_mse <= mean_mse / config.baseline_improvement_factor
            and image_gradient_norm > 0
            and state_gradient_norm > 0
            and np.isfinite(image_gradient_norm)
            and np.isfinite(state_gradient_norm)
        )
        if passed:
            break
    if not loss_trace or loss_trace[-1]["step"] != steps:
        loss_trace.append({"step": steps, "normalized_mse": normalized_mse})

    predictions_act = denormalize_act(predictions_normalized)
    prediction_rows = []
    for sample, target, prediction in zip(samples, targets_np, predictions_act):
        error = prediction.astype(np.float64) - target.astype(np.float64)
        prediction_rows.append({
            "reference": asdict(sample.reference),
            "target_act": target.tolist(),
            "prediction_act": prediction.tolist(),
            "act_l2_error": float(np.linalg.norm(error)),
            "max_abs_act_error": float(np.max(np.abs(error))),
        })
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_digest": inventory.dataset_digest,
        "baseline_report_content_sha256": baseline["content_sha256"],
        "config": asdict(config),
        "architecture": {
            "image": ["conv_3_8_k8_s8", "relu", "conv_8_16_k4_s4", "relu", "conv_16_32_k3_s2_p1", "relu"],
            "state": ["linear_6_32", "relu"],
            "head": ["linear_544_128", "relu", "linear_128_6"],
        },
        "sample_count": len(samples),
        "references": [asdict(sample.reference) for sample in samples],
        "steps": steps,
        "passed": passed,
        "normalized_mse": normalized_mse,
        "max_act_error": max_act_error,
        "subset_baseline_normalized_mse": {
            "current_pose": current_mse,
            "train_mean_target": mean_mse,
        },
        "gradient_norms_at_step_1": {
            "image_branch": image_gradient_norm,
            "state_branch": state_gradient_norm,
        },
        "loss_trace": loss_trace,
        "predictions": prediction_rows,
        "joint_order": list(JOINT_NAMES),
        "claim": "pipeline_memorization_only_not_held_out_or_task_success",
    }
    report["content_sha256"] = content_sha256(report)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_buffer = io.BytesIO()
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "dataset_digest": inventory.dataset_digest,
    }, checkpoint_buffer)
    _write_immutable(directory / "tiny_model.pt", checkpoint_buffer.getvalue())
    write_immutable_json(directory / "report.json", report)
    _write_immutable(directory / "report.html", _html(samples, report).encode("utf-8"))
    return _result_from_report(directory, report)


__all__ = [
    "MEMORIZATION_REFERENCES",
    "TinyModelConfig",
    "TinyOverfitResult",
    "build_memorization_subset",
    "denormalize_act",
    "normalize_act",
    "run_tiny_overfit",
]
