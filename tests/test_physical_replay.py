"""Hardware-free tests for the physical trajectory replay tool."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from replay_physical_trajectory import (  # noqa: E402
    gate_command,
    load_csv_trajectory,
    load_manifest_trajectory,
)

from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds


def test_gate_command_accepts_small_steps_and_refuses_violations() -> None:
    current = np.zeros(6, dtype=np.float32)
    current[5] = 0.5
    target = current + np.float32(0.01)
    ok, reason = gate_command(current, target)
    assert ok and reason == ""
    # Sub-floor gripper: refused with the joint named.
    low, _ = effective_safe_act_bounds()
    bad = current.copy()
    bad[5] = low[5] - 0.01
    ok, reason = gate_command(current, bad)
    assert not ok and "gripper" in reason
    # Excessive step: relative limiter names the joint.
    fast = current.copy()
    fast[0] = current[0] + 1.5
    ok, reason = gate_command(current, fast)
    assert not ok and "relative_limit" in reason
    # Nonfinite refused.
    nan = current.copy()
    nan[2] = np.nan
    ok, reason = gate_command(current, nan)
    assert not ok


def test_load_manifest_trajectory(tmp_path: Path) -> None:
    rows = 6
    executed = np.linspace(0, 0.05, rows * 6, dtype=np.float32).reshape(rows, 6)
    np.savez(tmp_path / "demonstrations.npz",
             executed_act=executed, action_index=np.arange(rows))
    manifest = {
        "arrays": {"path": "demonstrations.npz"},
        "episodes": [
            {"scenario_id": "other", "rows": 2},
            {"scenario_id": "nominal", "rows": 4},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    trajectory = load_manifest_trajectory(tmp_path / "manifest.json", "nominal")
    assert trajectory.shape == (4, 6)
    assert np.array_equal(trajectory, executed[2:6])
    with pytest.raises(ValueError, match="not in manifest"):
        load_manifest_trajectory(tmp_path / "manifest.json", "missing")


def test_load_csv_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "actions.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"action_rad_{i}" for i in range(6)])
        writer.writerow([0.1] * 6)
        writer.writerow([0.2] * 6)
    trajectory = load_csv_trajectory(path)
    assert trajectory.shape == (2, 6)
    with pytest.raises(ValueError, match="lacks"):
        load_csv_trajectory(path, prefix="wrong_")
