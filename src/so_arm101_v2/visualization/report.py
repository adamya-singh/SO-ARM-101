"""Self-contained HTML sample diagnostics."""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    JOINT_NAMES,
    MUJOCO_JOINT_HIGH,
    MUJOCO_JOINT_LOW,
    PHYSICAL_NORMALIZED_HIGH,
    PHYSICAL_NORMALIZED_LOW,
    PhysicalCalibration,
    evaluate_physical_command,
    load_physical_calibration,
)
from so_arm101_v2.data import FutureStateSample
from so_arm101_v2.data._serialization import canonical_json_bytes, write_immutable_json


@dataclass(frozen=True)
class SampleReportPaths:
    directory: Path
    html: Path
    json: Path


def _png_data_uri(image: np.ndarray) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sample reports require the 'visualize' extra") from exc
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable report already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _trajectory_svg(values: np.ndarray, width: int = 230, height: int = 54) -> str:
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = 4 + index * (width - 8) / max(1, len(values) - 1)
        y = 4 + (high - float(value)) / span * (height - 8)
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#47b0ff" '
        'stroke-width="2"/><circle cx="4" cy="' + points[0].split(",")[1]
        + '" r="3" fill="#75d18a"/><circle cx="' + points[-1].split(",")[0]
        + '" cy="' + points[-1].split(",")[1] + '" r="3" fill="#ffb55a"/></svg>'
    )


def _evaluation_dict(sample: FutureStateSample, calibration: PhysicalCalibration) -> dict[str, Any]:
    command = evaluate_physical_command(
        sample.current_state, sample.future_target, calibration=calibration
    )
    arrays = {
        "requested_act": command.requested_act,
        "hard_clipped_act": command.hard_clipped_act,
        "act_clip_mask": command.act_clip_mask,
        "requested_mujoco": command.requested_mujoco,
        "clipped_mujoco": command.clipped_mujoco,
        "mujoco_clip_mask": command.mujoco_clip_mask,
        "requested_physical": command.requested_physical,
        "hard_clipped_physical": command.hard_clipped_physical,
        "physical_clip_mask": command.physical_clip_mask,
        "current_physical": command.current_physical,
        "relative_limited_physical": command.relative_limited_physical,
        "relative_limit_mask": command.relative_limit_mask,
        "raw_goal_ticks": command.raw_goal_ticks,
    }
    return {
        "schema_version": 1,
        "dataset_digest": sample.dataset_digest,
        "reference": {
            "episode_id": sample.reference.episode_id,
            "frame_index": sample.reference.frame_index,
            "lead_steps": sample.reference.lead_steps,
        },
        "source": {
            "data_file": sample.source_data_file,
            "video_file": sample.source_video_file,
            "video_frame": sample.source_video_frame,
        },
        "image": {
            "raw": {"shape": [256, 256, 3], "dtype": "uint8", "layout": "HWC", "range": [0, 255]},
            "model": {"shape": [3, 256, 256], "dtype": "float32", "layout": "CHW", "range": [0.0, 1.0]},
            "preprocessing": "uint8_to_float32_divide_255_then_hwc_to_chw",
        },
        "joint_order": list(JOINT_NAMES),
        "current_state": sample.current_state.tolist(),
        "future_target": sample.future_target.tolist(),
        "observed_states": sample.observed_states.tolist(),
        "observed_timestamps": sample.observed_timestamps.tolist(),
        "limits": {
            "act_low": ACT_DATASET_LOW.tolist(),
            "act_high": ACT_DATASET_HIGH.tolist(),
            "mujoco_low": MUJOCO_JOINT_LOW.tolist(),
            "mujoco_high": MUJOCO_JOINT_HIGH.tolist(),
            "physical_low": PHYSICAL_NORMALIZED_LOW.tolist(),
            "physical_high": PHYSICAL_NORMALIZED_HIGH.tolist(),
            "max_relative_target": command.max_relative_target,
        },
        "conversion": {
            key: value.tolist() for key, value in arrays.items()
        },
        "calibration": {
            "resource": calibration.resource_name,
            "raw_ranges": [
                [joint.range_min, joint.range_max] for joint in calibration.joints
            ],
            "capture_time_calibration_claimed": False,
        },
        "hardware_io_performed": False,
    }


def _html_document(sample: FutureStateSample, payload: dict[str, Any]) -> str:
    conversion = payload["conversion"]
    limits = payload["limits"]
    rows = []
    for index, joint in enumerate(JOINT_NAMES):
        masks = [
            bool(conversion[name][index])
            for name in ("act_clip_mask", "mujoco_clip_mask", "physical_clip_mask", "relative_limit_mask")
        ]
        rows.append(
            "<tr>"
            f"<th>{html.escape(joint)}</th>"
            f"<td>{sample.current_state[index]:.6f}</td>"
            f"<td>{sample.future_target[index]:.6f}</td>"
            f"<td>{sample.future_target[index] - sample.current_state[index]:+.6f}</td>"
            f"<td>{conversion['requested_mujoco'][index]:.6f}</td>"
            f"<td>{conversion['clipped_mujoco'][index]:.6f}</td>"
            f"<td>{conversion['requested_physical'][index]:.3f}</td>"
            f"<td>{conversion['relative_limited_physical'][index]:.3f}</td>"
            f"<td>{conversion['raw_goal_ticks'][index]}</td>"
            f"<td>{' ⚠ '.join(name for name, flag in zip(('ACT','MJ','hard','relative'), masks) if flag) or 'none'}</td>"
            f"<td>{_trajectory_svg(sample.observed_states[:, index])}</td>"
            "</tr>"
        )
    final_rgb = np.transpose(sample.model_image, (1, 2, 0))
    final_rgb = np.rint(np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)
    ref = sample.reference
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SO-ARM-101 sample ep{ref.episode_id} frame{ref.frame_index} lead{ref.lead_steps}</title>
<style>
body{{font-family:ui-monospace,SFMono-Regular,monospace;background:#10151c;color:#e7edf5;margin:24px}}
h1,h2{{font-family:system-ui,sans-serif}} .images{{display:flex;gap:24px;flex-wrap:wrap}}
.card{{background:#18212b;border:1px solid #334253;border-radius:9px;padding:14px}} img{{width:384px;max-width:100%;image-rendering:auto}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #334253;padding:6px;text-align:right}}
th{{text-align:left;background:#1e2a36}} svg{{width:180px;height:44px}} code{{color:#75d18a}} .warning{{color:#ffb55a}}
</style></head><body>
<h1>One sample, every boundary</h1>
<p>Dataset <code>{sample.dataset_digest}</code><br>Episode {ref.episode_id}, frame {ref.frame_index} → {ref.frame_index + ref.lead_steps}, lead {ref.lead_steps} ({ref.lead_steps / 30:.3f}s)</p>
<div class="images"><section class="card"><h2>Raw image</h2><img src="{_png_data_uri(sample.raw_image)}"><p>RGB uint8 · HWC · 256×256</p></section>
<section class="card"><h2>Final model input</h2><img src="{_png_data_uri(final_rgb)}"><p>float32 · CHW · 3×256×256 · divide by 255 only</p></section></div>
<h2>State, target, conversions, and observed motion</h2>
<table><thead><tr><th>joint</th><th>current ACT</th><th>target ACT</th><th>observed Δ</th><th>requested MJ</th><th>clipped MJ</th><th>physical target</th><th>relative-limited</th><th>raw tick</th><th>clipping</th><th>t…t+lead</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="warning">Diagnostic only. The relative limiter uses the recorded current state as a proxy for live motor position. No hardware I/O occurred.</p>
<p>ACT limits: {limits['act_low']} … {limits['act_high']}<br>MuJoCo limits: {limits['mujoco_low']} … {limits['mujoco_high']}<br>Physical normalized limits: {limits['physical_low']} … {limits['physical_high']}; max relative target: {limits['max_relative_target']}</p>
</body></html>"""


def build_sample_report(
    sample: FutureStateSample,
    output_dir: str | Path,
    *,
    calibration: PhysicalCalibration | None = None,
) -> SampleReportPaths:
    """Create immutable JSON and self-contained HTML for one sample."""
    calibration = calibration or load_physical_calibration()
    ref = sample.reference
    directory = Path(output_dir) / (
        f"ep{ref.episode_id:03d}_f{ref.frame_index:04d}_l{ref.lead_steps:02d}."
        f"{sample.dataset_digest[:12]}"
    )
    json_path = directory / "sample.json"
    html_path = directory / "sample.html"
    payload = _evaluation_dict(sample, calibration)
    write_immutable_json(json_path, payload)
    _write_immutable(html_path, _html_document(sample, payload).encode("utf-8"))
    return SampleReportPaths(directory, html_path, json_path)


__all__ = ["SampleReportPaths", "build_sample_report"]
