"""The physical episode tool: hardware ordering invariants with a fake LeRobot, refusals, and the sim rehearsal path."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.bench import scene_bench_config

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
CHECKPOINT = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909/models/vision_h90/5e96018881d4140f/model.pt"
REST = np.array([0.0, -92.08, 100.0, 40.0, -2.0, 13.5], dtype=np.float64)   # gravity rest, shoulder below the floor

needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.exists(), reason="lens-run checkpoint not present")


def _tool():
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    import run_physical_episode as tool
    return tool


class TrackingRobot:
    """Fake SO101Follower: Present_Position follows every sent goal exactly; records the event order."""

    def __init__(self, events, start_physical):
        self.events = events
        self.pose = np.asarray(start_physical, dtype=np.float64).copy()
        self.voltage_raw = 74   # Present_Voltage in 0.1 V units: a healthy 7.4 V supply
        self.bus = Mock()

        def sync_read(name="Present_Position", *a, **k):
            if name == "Present_Voltage":
                return {n: float(self.voltage_raw) for n in JOINT_NAMES}
            return {n: float(v) for n, v in zip(JOINT_NAMES, self.pose)}

        self.bus.sync_read.side_effect = sync_read
        self.bus.sync_write.side_effect = lambda *a, **k: events.append("goal")
        self.bus.enable_torque.side_effect = lambda *a, **k: events.append("torque")
        self.bus.disable_torque.side_effect = lambda *a, **k: events.append("DISABLE")
        self.calibration = {}

    def send_action(self, action):
        self.events.append("send")
        self.pose = np.asarray([action[f"{n}.pos"] for n in JOINT_NAMES], dtype=np.float64)
        return dict(action)


class FakeGrabber:
    def __init__(self, device=0, width=1920, height=1080, **kwargs):
        self.properties = dict(fourcc="MJPG", width=width, height=height, fps=30.0)
        self.size = (height, width)
        self.seq = 0
        self.closed = False
        FakeGrabber.events.append("camera")

    def latest(self):
        import time
        self.seq += 1
        frame = np.zeros((*self.size, 3), np.uint8)
        frame[:, :, 2] = 40
        return self.seq, time.time(), frame

    def wait_for_new(self, after_seq, timeout=1.0):
        return self.latest()

    def measure_rate(self, seconds=1.0):
        return 30.0

    def close(self):
        self.closed = True
        FakeGrabber.events.append("camera_closed")


class FakeRecorder:
    def __init__(self, grabber, path, fps=30):
        self.path = Path(path)

    def start(self):
        pass

    def stop(self):
        return dict(path=self.path.name, frames=0, dropped=0, sha256=None)


def _install(monkeypatch, events, robot):
    fake_root = types.ModuleType("lerobot")
    fake_robots = types.ModuleType("lerobot.robots")
    fake_follower = types.ModuleType("lerobot.robots.so_follower")
    fake_follower.SO101Follower = lambda config: robot
    fake_follower.SO101FollowerConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "lerobot", fake_root)
    monkeypatch.setitem(sys.modules, "lerobot.robots", fake_robots)
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", fake_follower)
    tool = _tool()
    FakeGrabber.events = events
    monkeypatch.setattr(tool, "FrameGrabber", FakeGrabber)
    monkeypatch.setattr(tool, "VideoRecorder", FakeRecorder)
    monkeypatch.setattr(tool, "assert_pinned_calibration", lambda r: events.append("calibration"))
    monkeypatch.setattr(tool, "connect_read_only", lambda r: events.append("connect"))
    monkeypatch.setattr(tool, "disconnect_read_only", lambda r: events.append("disconnect"))
    monkeypatch.setattr(tool, "run_pan_sign_check", lambda *a, **k: dict(passed=True, measured_sign=-1, expected_sign=-1))
    # The fake camera's flat frame is not a bench frame; the real-frame gate is exercised by its own tests below.
    monkeypatch.setattr(tool, "check_reset_frame", lambda policy, image, anchor, bench, **kw: dict(
        gate="stub", label=kw.get("label", "frame"), passed=True, reasons=[], thresholds={},
        dry_pass=dict(chunk_len=90, max_abs_delta_from_start_units={n: 0.0 for n in JOINT_NAMES}, holds_in_dry_chunk=0, chunk_physical=[])))
    monkeypatch.setattr(tool.time, "sleep", lambda s: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    return tool


@needs_checkpoint
def test_motion_ordering_claims_run_dir_before_torque_and_never_disables_torque(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    robot = TrackingRobot(events, REST)
    tool = _install(monkeypatch, events, robot)
    original_mkdir = Path.mkdir

    def tracking_mkdir(self, *a, **k):
        if self.name == "run":
            events.append("rundir")
        return original_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", tracking_mkdir)
    run_dir = tmp_path / "run"
    code = tool.main(["--enable-motion", "--run-dir", str(run_dir), "--no-preview", "--max-actions", "5"])
    assert code == 0, events[:12]
    first = [e for e in events if e in ("calibration", "rundir", "camera", "connect", "goal", "torque", "disconnect", "camera_closed")]
    assert first[:6] == ["calibration", "rundir", "camera", "connect", "goal", "torque"]
    assert events.count("torque") == 1 and "DISABLE" not in events
    assert events.index("torque") < events.index("send") and events[-2:] == ["disconnect", "camera_closed"]
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "completed" and record["actions"] == 5 and record["hold_frames"] == 0
    assert record["approach"]["steps"] > 0 and max(abs(v) for v in record["approach"]["residual_physical"]) <= 1.0
    assert record["start_pose_delta_act"] <= tool.START_POSE_TOLERANCE_ACT
    assert record["pan_sign"]["skipped"] and record["resampler"]["bit_identical"] and record["dry_pass"]["chunk_len"] == 90
    assert record["servo_voltage"]["ok"] and record["servo_voltage"]["lowest_v"] == 7.4
    assert record["real_frame_check"]["passed"] and record["real_frame_check"]["label"] == "reset_observation"
    assert (run_dir / "preflight" / "real_frame_gate_reset.png").exists()
    assert len(record["confirmations"]) == 2
    assert (run_dir / "steps.csv").exists() and (run_dir / "approach.csv").exists() and (run_dir / "boundaries" / "step_000.obs.png").exists()
    assert (run_dir / "preflight" / "observation.png").exists()
    # The approach delivered the arm to the recorded reset (the 5-step episode then moved it a little).
    assert max(abs(v) for v in record["approach"]["residual_physical"]) <= 1.0


@needs_checkpoint
def test_preflight_touches_no_torque_and_refuses_a_failed_pan_check(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    robot = TrackingRobot(events, REST)
    tool = _install(monkeypatch, events, robot)
    code = tool.main(["--preflight-only", "--run-dir", str(tmp_path / "pre"), "--no-preview"])
    assert code == 0 and "torque" not in events and "send" not in events and "goal" not in events
    record = json.loads((tmp_path / "pre" / "run.json").read_text())
    assert record["status"] == "preflight_ok" and record["approach_plan"]["shoulder_below_floor"]
    events.clear()
    monkeypatch.setattr(tool, "run_pan_sign_check", lambda *a, **k: dict(passed=False, measured_sign=1, expected_sign=-1))
    code = tool.main(["--enable-motion", "--run-dir", str(tmp_path / "refused"), "--no-preview", "--pan-check"])
    assert code == 2 and "torque" not in events and "send" not in events
    record = json.loads((tmp_path / "refused" / "run.json").read_text())
    assert record["status"] == "refused" and "pan sign" in record["reason"]


@needs_checkpoint
def test_connect_failure_sends_nothing_and_leaves_no_torque(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    robot = TrackingRobot(events, REST)
    tool = _install(monkeypatch, events, robot)

    def failing_connect(r):
        raise RuntimeError("bus not calibrated")

    monkeypatch.setattr(tool, "connect_read_only", failing_connect)
    code = tool.main(["--enable-motion", "--run-dir", str(tmp_path / "run"), "--no-preview"])
    assert code == 2 and "torque" not in events and "send" not in events and "disconnect" not in events
    assert "camera_closed" in events
    assert json.loads((tmp_path / "run" / "run.json").read_text())["status"] == "aborted"


@needs_checkpoint
def test_sim_rehearsal_reproduces_the_nominal_success_with_evidence(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    try:
        import mujoco
        mujoco.Renderer(mujoco.MjModel.from_xml_path(str(SCENE)), height=64, width=64).close()
    except Exception:
        pytest.skip("offscreen renderer unavailable")
    tool = _tool()
    run_dir = tmp_path / "rehearsal"
    assert tool.main(["--sim-rehearsal", "--run-dir", str(run_dir)]) == 0
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "completed" and record["success"] and record["hold_frames"] == 0 and record["actions"] == 427
    assert (run_dir / "steps.csv").read_text().count("\n") == 428
    assert len(record["boundaries"]) == 5 and (run_dir / "boundaries" / "step_360.obs.png").exists()


@needs_checkpoint
def test_real_frame_gate_failure_refuses_the_episode_after_the_approach(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    robot = TrackingRobot(events, REST)
    tool = _install(monkeypatch, events, robot)
    monkeypatch.setattr(tool, "check_reset_frame", lambda policy, image, anchor, bench, **kw: dict(
        gate="stub", label=kw.get("label", "frame"), passed=False, reasons=["shoulder_lift moves 26.0 units in the dry chunk (limit 3.0)"], thresholds={},
        dry_pass=dict(chunk_len=90, max_abs_delta_from_start_units={n: 0.0 for n in JOINT_NAMES}, holds_in_dry_chunk=34, chunk_physical=[])))
    run_dir = tmp_path / "gated"
    code = tool.main(["--enable-motion", "--run-dir", str(run_dir), "--no-preview", "--max-actions", "5"])
    assert code == 2
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "refused" and "real-frame gate" in record["reason"] and record["real_frame_check"]["passed"] is False
    assert record["approach"]["steps"] > 0                       # the approach ran (torque on, arm holding at the reset)
    assert events.count("torque") == 1 and "DISABLE" not in events
    assert not (run_dir / "steps.csv").exists() and "episode_started_at" not in record


@needs_checkpoint
def test_low_servo_voltage_refuses_before_any_torque(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    robot = TrackingRobot(events, REST)
    robot.voltage_raw = 54   # the 2026-09-09 bench supply
    tool = _install(monkeypatch, events, robot)
    run_dir = tmp_path / "lowv"
    code = tool.main(["--enable-motion", "--run-dir", str(run_dir), "--no-preview"])
    assert code == 2 and "torque" not in events and "goal" not in events and "send" not in events
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "refused" and "5.4 V" in record["reason"] and record["servo_voltage"]["ok"] is False
    assert record["servo_voltage"]["volts"]["gripper"] == 5.4 and record["servo_voltage"]["minimum_v"] == 6.0
