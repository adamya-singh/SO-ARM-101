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


def _fake_lerobot(monkeypatch, robot):
    import types
    module = types.ModuleType("lerobot.robots.so_follower")
    module.SO101Follower = lambda config: robot
    module.SO101FollowerConfig = lambda **kwargs: kwargs
    cameras = types.ModuleType("lerobot.cameras.opencv")
    cameras.OpenCVCameraConfig = object
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.robots", types.ModuleType("lerobot.robots"))
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", module)
    monkeypatch.setitem(sys.modules, "lerobot.cameras", types.ModuleType("lerobot.cameras"))
    monkeypatch.setitem(sys.modules, "lerobot.cameras.opencv", cameras)


def _write_trajectory(tmp_path: Path, pose: np.ndarray, rows: int = 3) -> Path:
    path = tmp_path / "trajectory.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"action_rad_{i}" for i in range(6)])
        for _ in range(rows):
            writer.writerow([float(v) for v in pose])
    return path


def test_motion_claims_log_and_checks_calibration_before_hardware(tmp_path: Path, monkeypatch) -> None:
    """The exclusive log must exist before torque is enabled or any goal is sent."""
    from unittest.mock import Mock
    import replay_physical_trajectory as tool
    from so_arm101_v2.contracts.bench import BenchConfig
    from so_arm101_v2.contracts.physical import physical_normalized_to_act

    pose = physical_normalized_to_act([0, -90, 90, 43, -1, 13])
    trajectory = _write_trajectory(tmp_path, pose)
    bench_path = tmp_path / "bench_config.json"
    from dataclasses import asdict
    bench_path.write_text(json.dumps(asdict(BenchConfig(reset_physical=(0, -90, 90, 43, -1, 13)))))
    log_path = tmp_path / "logs" / "motion.csv"

    events: list[str] = []
    robot = Mock()
    robot.bus.enable_torque.side_effect = lambda *a, **k: events.append(
        "torque" if log_path.exists() else "torque_before_log")
    robot.bus.sync_write.side_effect = lambda *a, **k: events.append("goal")
    robot.send_action.side_effect = lambda action: (events.append("send"), action)[1]
    _fake_lerobot(monkeypatch, robot)
    monkeypatch.setattr(tool, "assert_pinned_calibration", lambda r: events.append("calibration"))
    monkeypatch.setattr(tool, "connect_read_only", lambda r: events.append("connect"))
    monkeypatch.setattr(tool, "disconnect_read_only", lambda r: events.append("disconnect"))
    monkeypatch.setattr(tool, "read_measured_act", lambda r: pose.copy())
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(tool.time, "sleep", lambda s: None)

    code = tool.main(["--csv", str(trajectory), "--bench-config", str(bench_path),
                      "--enable-motion", "--log", str(log_path), "--robot-port", "/dev/null"])
    assert code == 0
    assert events[:3] == ["calibration", "connect", "goal"], events
    assert "torque_before_log" not in events and events.count("torque") == 1
    assert events.count("send") == 3 and events[-1] == "disconnect"
    robot.bus.disable_torque.assert_not_called()
    with open(log_path, newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 4 and rows[0][0] == "step" and all(row[-1] == "" for row in rows[1:])


def test_failed_connection_removes_unused_log(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import Mock
    import replay_physical_trajectory as tool
    from so_arm101_v2.contracts.bench import BenchConfig
    from so_arm101_v2.contracts.physical import physical_normalized_to_act
    from dataclasses import asdict

    pose = physical_normalized_to_act([0, -90, 90, 43, -1, 13])
    trajectory = _write_trajectory(tmp_path, pose)
    bench_path = tmp_path / "bench_config.json"
    bench_path.write_text(json.dumps(asdict(BenchConfig(reset_physical=(0, -90, 90, 43, -1, 13)))))
    log_path = tmp_path / "motion.csv"
    robot = Mock()
    _fake_lerobot(monkeypatch, robot)
    monkeypatch.setattr(tool, "assert_pinned_calibration", lambda r: None)

    def refuse(r):
        raise RuntimeError("motor calibration does not match file")
    monkeypatch.setattr(tool, "connect_read_only", refuse)
    with pytest.raises(RuntimeError, match="calibration"):
        tool.main(["--csv", str(trajectory), "--bench-config", str(bench_path),
                   "--enable-motion", "--log", str(log_path), "--robot-port", "/dev/null"])
    assert not log_path.exists()
    robot.bus.enable_torque.assert_not_called()
    robot.send_action.assert_not_called()
