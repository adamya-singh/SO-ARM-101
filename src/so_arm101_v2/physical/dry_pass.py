"""Offline policy dry pass and the real-frame gate: what the network would do on one frame, nothing sent.

The dry pass runs the chunked vision policy once on an observation and walks
its first chunk through the shared bench gate against a simulated "current"
pose (each executed command becomes the next current, as the runner would
do), reporting the chunk as physical units. The real-frame gate applies it to
a frame taken at the reset pose and demands a hold-like chunk: on the
simulated reset frame the lens policies move the shoulder and elbow by well
under one unit; on the 2026-09-09 real reset frame the same network moved
them by 26 and 23 units with 34 holds. Thresholds sit far from both.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.physical import act_to_physical_normalized, bench_hold_decision

REAL_FRAME_MAX_DELTA_UNITS = 3.0
REAL_FRAME_JOINTS = ("shoulder_lift", "elbow_flex")
REAL_FRAME_GATE_VERSION = "real_frame_gate_v1"


def policy_dry_pass(policy: Any, image: np.ndarray, current_act: np.ndarray, bench: Any) -> dict[str, Any]:
    """Run the network once on ``image`` and summarise the first chunk without sending anything."""
    policy.reset()
    chunk = []
    holds = 0
    simulated_current = np.asarray(current_act, dtype=np.float32)
    for k in range(policy.chunk_horizon):
        act = policy.predict(image if k == 0 else None, simulated_current)
        decision = bench_hold_decision(simulated_current, act, shoulder_floor=bench.shoulder_floor, joint_map=bench.joint_map_object)
        holds += int(decision.held)
        chunk.append(act_to_physical_normalized(act))
        simulated_current = decision.executed_act
    chunk = np.asarray(chunk, dtype=np.float64)
    policy.reset()
    start = act_to_physical_normalized(np.asarray(current_act, dtype=np.float32))
    return dict(
        chunk_len=int(chunk.shape[0]),
        max_abs_delta_from_start_units={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, np.max(np.abs(chunk - start), axis=0))},
        min_units={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, np.min(chunk, axis=0))},
        max_units={n: round(float(v), 2) for n, v in zip(JOINT_NAMES, np.max(chunk, axis=0))},
        holds_in_dry_chunk=int(holds),
        first_command=[round(float(v), 3) for v in chunk[0]],
        start_physical=[round(float(v), 3) for v in start],
        chunk_physical=[[round(float(v), 3) for v in row] for row in chunk],
    )


def chunk_difference_units(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Per-joint max |chunk_a - chunk_b| over the common prefix of two dry passes."""
    ca, cb = np.asarray(a["chunk_physical"], dtype=np.float64), np.asarray(b["chunk_physical"], dtype=np.float64)
    n = min(ca.shape[0], cb.shape[0])
    return {name: round(float(v), 2) for name, v in zip(JOINT_NAMES, np.max(np.abs(ca[:n] - cb[:n]), axis=0))}


def check_reset_frame(policy: Any, image: np.ndarray, anchor_act: np.ndarray, bench: Any, *, label: str = "frame",
                      reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """The real-frame gate: a chunk on a reset-pose frame must hold (no gate holds, tiny shoulder/elbow motion, above the floor)."""
    dry = policy_dry_pass(policy, image, anchor_act, bench)
    reasons = []
    for joint in REAL_FRAME_JOINTS:
        delta = dry["max_abs_delta_from_start_units"][joint]
        if delta > REAL_FRAME_MAX_DELTA_UNITS:
            reasons.append(f"{joint} moves {delta:.2f} units in the dry chunk (limit {REAL_FRAME_MAX_DELTA_UNITS})")
    if dry["holds_in_dry_chunk"] > 0:
        reasons.append(f"{dry['holds_in_dry_chunk']} gate holds in the dry chunk")
    if dry["min_units"]["shoulder_lift"] < bench.shoulder_floor:
        reasons.append(f"shoulder would reach {dry['min_units']['shoulder_lift']:.2f} units, below the floor {bench.shoulder_floor}")
    result: dict[str, Any] = dict(
        gate=REAL_FRAME_GATE_VERSION, label=label, passed=not reasons, reasons=reasons,
        thresholds=dict(max_delta_units=REAL_FRAME_MAX_DELTA_UNITS, joints=list(REAL_FRAME_JOINTS), holds=0, shoulder_floor=float(bench.shoulder_floor)),
        dry_pass=dry,
    )
    if reference is not None:
        result["delta_to_reference_units"] = chunk_difference_units(dry, reference)
    return result


def sim_reference_dry_pass(policy: Any, model_path: str | Path, anchor_act: np.ndarray, bench: Any) -> dict[str, Any]:
    """The same dry pass on the simulator's nominal reset observation (what the policy was scored on)."""
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    adapter = MujocoTaskAdapter(model_path)
    try:
        adapter.reset(bench_suite(bench, [(0, 0)], label="reference", repeats=1).scenarios[0])
        image = adapter.render_wrist_observation()
    finally:
        adapter.close()
    result = policy_dry_pass(policy, image, anchor_act, bench)
    result["observation_array_sha256"] = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    return result


def load_boundary_frame(episode_dir: str | Path, step: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """A recorded physical boundary observation with its measured anchor pose, verified against the run's hashes."""
    from PIL import Image
    episode_dir = Path(episode_dir)
    entries = json.loads((episode_dir / "boundaries" / "boundaries.json").read_text())
    entry = next((e for e in entries if int(e["step"]) == int(step)), None)
    if entry is None:
        raise ValueError(f"{episode_dir} has no boundary record for step {step}")
    png = episode_dir / "boundaries" / entry["observation_png"]
    if hashlib.sha256(png.read_bytes()).hexdigest() != entry["observation_png_sha256"]:
        raise ValueError(f"{png} does not match its recorded hash")
    image = np.asarray(Image.open(png).convert("RGB"))
    if image.shape != (256, 256, 3) or hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != entry["observation_array_sha256"]:
        raise ValueError(f"{png} decodes to an array that does not match its recorded hash")
    with (episode_dir / "steps.csv").open() as handle:
        row = next((r for r in csv.DictReader(handle) if int(r["step"]) == int(step)), None)
    if row is None:
        raise ValueError(f"{episode_dir}/steps.csv has no row for step {step}")
    anchor = np.asarray([float(row[f"measured_act_{name}"]) for name in JOINT_NAMES], dtype=np.float32)
    evidence = dict(episode_dir=str(episode_dir), step=int(step), observation_png=str(png), observation_png_sha256=entry["observation_png_sha256"],
                    observation_array_sha256=entry["observation_array_sha256"], anchor_act=[float(v) for v in anchor],
                    anchor_physical=[round(float(v), 3) for v in act_to_physical_normalized(anchor)])
    return image, anchor, evidence


__all__ = ["REAL_FRAME_GATE_VERSION", "REAL_FRAME_JOINTS", "REAL_FRAME_MAX_DELTA_UNITS", "check_reset_frame", "chunk_difference_units",
           "load_boundary_frame", "policy_dry_pass", "sim_reference_dry_pass"]
