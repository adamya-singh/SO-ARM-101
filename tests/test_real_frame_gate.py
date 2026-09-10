"""The offline real-frame gate: calibrated on the 2026-09-09 real reset frame (the lens policy must FAIL it), hold policies pass,
the simulated reset chunk passes, tampered evidence is refused, and the servo-voltage preflight reads the right register."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.physical.dry_pass import (
    REAL_FRAME_MAX_DELTA_UNITS,
    check_reset_frame,
    chunk_difference_units,
    load_boundary_frame,
    policy_dry_pass,
    sim_reference_dry_pass,
)
from so_arm101_v2.physical.lerobot_backend import check_servo_voltage, read_present_voltages

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
CHECKPOINT = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909/models/vision_h90/5e96018881d4140f/model.pt"
EPISODE_02 = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/physical/episode_02_20260909"


def _legacy_bench():
    """The pre-placement configuration: fixed viewing pose (= reset) and no placement regime (gate v2 semantics)."""
    from dataclasses import replace
    bench = scene_bench_config(SCENE)
    return replace(bench, placement=None, viewing_qpos=tuple(float(v) for v in bench.reset_qpos))


def _renderer_available() -> bool:
    try:
        import mujoco
        mujoco.Renderer(mujoco.MjModel.from_xml_path(str(SCENE)), height=256, width=256).close()
        return True
    except Exception:
        return False


needs_evidence = pytest.mark.skipif(not (CHECKPOINT.exists() and EPISODE_02.exists() and _renderer_available()),
                                    reason="lens checkpoint, episode_02 evidence or EGL renderer unavailable")


class HoldPolicy:
    chunk_horizon = 90

    def reset(self):
        pass

    def predict(self, image, current):
        return np.asarray(current, dtype=np.float32)


@needs_evidence
def test_lens_policy_fails_the_gate_on_the_real_reset_frame_and_passes_in_sim():
    """Calibration anchor: the physical failure of 2026-09-09 must be caught offline; the sim chunk is a hold."""
    torch = pytest.importorskip("torch")
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    torch.set_num_threads(1)
    bench = _legacy_bench()
    image, anchor, evidence = load_boundary_frame(EPISODE_02, step=0)
    assert evidence["observation_array_sha256"] == "1dae371e9de6e572be85c6ca383d588e3f2bbd11bbc8479a9530325497700c38"
    assert abs(evidence["anchor_physical"][1] - (-90.6)) < 0.5     # the measured reset pose, shoulder just above the floor
    policy = VisionChunkedPolicy(CHECKPOINT, black_image=False, clamp_channels=(5,))
    reference = sim_reference_dry_pass(policy, SCENE, anchor, bench)
    real = check_reset_frame(policy, image, anchor, bench, label="episode_02", reference=reference)
    assert real["passed"] is False and real["reasons"]
    assert real["dry_pass"]["max_abs_delta_from_start_units"]["shoulder_lift"] > 10
    assert real["dry_pass"]["max_abs_delta_from_start_units"]["elbow_flex"] > 10
    assert real["dry_pass"]["holds_in_dry_chunk"] > 0
    assert real["delta_to_reference_units"]["shoulder_lift"] > 10
    # The same network on the simulated reset observation at the same anchor is a hold: well inside the thresholds.
    assert reference["max_abs_delta_from_start_units"]["shoulder_lift"] < 1.0 and reference["max_abs_delta_from_start_units"]["elbow_flex"] < 1.0
    assert reference["holds_in_dry_chunk"] == 0
    assert REAL_FRAME_MAX_DELTA_UNITS == 3.0
    json.dumps(real)   # evidence must serialise


def test_hold_policy_passes_and_thresholds_bite():
    bench = _legacy_bench()
    anchor = np.asarray([0.0017, -2.8565, 2.8529, 1.3711, -0.0345, 0.2324], dtype=np.float32)
    image = np.zeros((256, 256, 3), np.uint8)
    result = check_reset_frame(HoldPolicy(), image, anchor, bench)
    assert result["passed"] and result["reasons"] == [] and result["dry_pass"]["holds_in_dry_chunk"] == 0
    assert all(v == 0.0 for v in result["dry_pass"]["max_abs_delta_from_start_units"].values())

    class Drift(HoldPolicy):
        def __init__(self, units):
            self.units = units
            self.calls = 0

        def predict(self, image, current):
            from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act
            self.calls += 1
            physical = act_to_physical_normalized(np.asarray(current, dtype=np.float32))
            physical[2] += self.units   # elbow creeps by `units` per step
            return physical_normalized_to_act(physical)

    small = check_reset_frame(Drift(0.03), image, anchor, bench)     # 90 x 0.03 = 2.7 units: inside the limit
    assert small["passed"], small["reasons"]
    big = check_reset_frame(Drift(0.05), image, anchor, bench)       # 4.5 units: refused
    assert not big["passed"] and "elbow_flex" in big["reasons"][0]
    assert chunk_difference_units(small["dry_pass"], big["dry_pass"])["elbow_flex"] == pytest.approx(1.8, abs=0.05)
    assert policy_dry_pass(HoldPolicy(), image, anchor, bench)["chunk_len"] == 90

    class Graze(HoldPolicy):
        """Dips the shoulder `depth` units below the start on one step (the live 2026-09-10 case: 0.12 below the floor, one hold)."""
        def __init__(self, depth):
            self.depth = depth; self.calls = 0

        def predict(self, image, current):
            from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act
            self.calls += 1
            physical = act_to_physical_normalized(np.asarray(current, dtype=np.float32))
            if self.calls == 5:
                physical[1] = bench.shoulder_floor - self.depth
            return physical_normalized_to_act(physical)

    graze = check_reset_frame(Graze(0.12), image, anchor, bench)
    assert graze["passed"] and graze["dry_pass"]["holds_in_dry_chunk"] == 1, graze["reasons"]
    deep = check_reset_frame(Graze(0.8), image, anchor, bench)
    assert not deep["passed"] and any("below the floor" in r for r in deep["reasons"])


@pytest.mark.skipif(not EPISODE_02.exists(), reason="episode_02 evidence unavailable")
def test_tampered_evidence_is_refused(tmp_path):
    copy = tmp_path / "episode"
    shutil.copytree(EPISODE_02, copy)
    load_boundary_frame(copy, step=0)
    from PIL import Image
    png = copy / "boundaries" / "step_000.obs.png"
    image = np.asarray(Image.open(png).convert("RGB")).copy()
    image[0, 0, 0] ^= 1
    Image.fromarray(image).save(png)
    with pytest.raises(ValueError, match="hash"):
        load_boundary_frame(copy, step=0)
    with pytest.raises(ValueError, match="no boundary record"):
        load_boundary_frame(EPISODE_02, step=17)


def test_servo_voltage_preflight_reads_the_raw_register_and_accepts_the_stock_supply():
    robot = Mock()
    seen = {}

    def sync_read(name, *a, **k):
        seen["name"], seen["kwargs"] = name, k
        return {n: 45.0 for n in JOINT_NAMES}

    robot.bus.sync_read.side_effect = sync_read
    assert read_present_voltages(robot) == {n: 4.5 for n in JOINT_NAMES}
    assert seen["name"] == "Present_Voltage" and seen["kwargs"] == {"normalize": False}
    low = check_servo_voltage(robot)
    assert low["ok"] is False and low["lowest_v"] == 4.5 and low["minimum_v"] == 4.8
    # The stock 5 V adapter reads 5.3-5.4 V under load on this arm and must pass.
    robot.bus.sync_read.side_effect = lambda name, *a, **k: {n: (54.0 if n != "gripper" else 53.0) for n in JOINT_NAMES}
    good = check_servo_voltage(robot)
    assert good["ok"] and good["lowest_v"] == 5.3 and good["volts"]["gripper"] == 5.3
    robot.bus.enable_torque.assert_not_called()
    robot.send_action.assert_not_called()


def test_survey_pose_gate_tracks_the_simulated_reference_chunk():
    """v3: with a survey pose the first chunk is a move; it must match the simulated reference and end at the survey pose."""
    from dataclasses import replace
    from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act
    from so_arm101_v2.physical.dry_pass import REAL_FRAME_GATE_VERSION_SURVEY, survey_pose_active
    base = _legacy_bench()
    survey_qpos = tuple(float(a + d) for a, d in zip(base.reset_qpos, (0.0, 0.40, -0.60, 0.40, 0.0, 0.0)))
    bench = replace(base, viewing_qpos=survey_qpos)
    assert survey_pose_active(bench) and not survey_pose_active(replace(base, viewing_qpos=tuple(float(v) for v in base.reset_qpos)))
    anchor = physical_normalized_to_act(np.asarray(bench.reset_physical, np.float32))
    target = act_to_physical_normalized(bench.joint_map_object.mujoco_to_act(np.asarray(survey_qpos, np.float32)))
    start = act_to_physical_normalized(anchor)

    class Mover(HoldPolicy):
        """Ramps from the reset pose to the survey pose over the chunk (what the trained policy does), with an optional error."""
        def __init__(self, error_units=0.0):
            self.error, self.calls = error_units, 0

        def predict(self, image, current):
            self.calls += 1
            frac = min(1.0, self.calls / 60.0)
            physical = start + frac * (target - start)
            physical[2] += self.error * frac
            return physical_normalized_to_act(physical)

    image = np.zeros((256, 256, 3), np.uint8)
    reference = policy_dry_pass(Mover(), image, anchor, bench)
    good = check_reset_frame(Mover(), image, anchor, bench, reference=reference)
    assert good["gate"] == REAL_FRAME_GATE_VERSION_SURVEY and good["passed"], good["reasons"]
    off = check_reset_frame(Mover(8.0), image, anchor, bench, reference=reference)
    assert not off["passed"] and any("differs from the simulated survey chunk" in r for r in off["reasons"])
    assert any("ends the chunk" in r for r in off["reasons"])
    missing = check_reset_frame(Mover(), image, anchor, bench)
    assert not missing["passed"] and "needs the simulated reference" in missing["reasons"][0]
    # A pure hold now fails: it never reaches the survey pose.
    hold = check_reset_frame(HoldPolicy(), image, anchor, bench, reference=reference)
    assert not hold["passed"]
