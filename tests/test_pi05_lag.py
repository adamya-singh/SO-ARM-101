"""Pi05ChunkPolicy under servo lag (re-read schedule, lag compensation) and the simulator's lag stress switch (no model, no GPU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from so_arm101_v2.simulation.pi05_policy import ANCHOR_LEAD_CAP_ACT, SIM_SERVO_RESPONSE, Pi05ChunkPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


class _Config:
    chunk_horizon = 90


class _Stats:
    def norm_state(self, state):
        return state


class _Model:
    def parameters(self):
        yield torch.zeros(1)


def _policy(monkeypatch, **kw):
    """The network is replaced by a fixed plan: joint 2 ramps -0.004 (normalised) per step, the others hold."""
    import so_arm101_v2.learning.pi05 as lane

    def fake_residual(policy, stats, images, tokens, masks, *, seed):
        residual = np.zeros((1, 90, 6), np.float32)
        residual[0, :, 2] = -0.004 * np.arange(1, 91)
        return torch.from_numpy(residual)

    monkeypatch.setattr(lane, "sample_residual", fake_residual)
    monkeypatch.setattr(lane, "prepare_images", lambda images, device: images)
    return Pi05ChunkPolicy(_Model(), _Config(), _Stats(), lambda state, device: (None, None), clamp_channels=(5,), **kw)


def _drive(policy, response: float, steps: int = 90, sag: float = 0.0) -> np.ndarray:
    """Run the policy against the simulated servo behind a first-order ``response`` (1.0 = the simulator); returns the commands.

    ``sag`` is a resting offset of joint 2 below its command (gravity), present at every step.
    """
    policy.reset()
    servo = np.array([0.0, -2.8, 2.8, 1.4, 0.0, 0.2], np.float64)
    filtered = servo.copy()
    image = np.zeros((256, 256, 3), np.uint8)
    offset = np.array([0.0, 0.0, sag, 0.0, 0.0, 0.0])
    commands = []
    for _ in range(steps):
        command = policy.predict(image if policy.needs_frame else None, (servo + offset).astype(np.float32))
        commands.append(command.copy())
        filtered = filtered + response * (command.astype(np.float64) - filtered)
        servo = servo + SIM_SERVO_RESPONSE * (filtered - servo)
    return np.array(commands)


SLOW = (1.0, 1.0, 0.16, 1.0, 1.0, 1.0)      # the 2026-10-04 elbow


def test_measured_anchor_loses_ground_on_a_slow_servo(monkeypatch):
    fast = _drive(_policy(monkeypatch, execute=30), 1.0)
    slow = _drive(_policy(monkeypatch, execute=30), 0.16)
    # Same plan, slower servo: every re-read restarts from the lagging joint, so the move ends short.
    assert slow[-1, 2] - fast[-1, 2] > 0.05
    # One read for the whole move does not care how slow the servo is.
    assert np.allclose(_drive(_policy(monkeypatch, execute=90), 1.0), _drive(_policy(monkeypatch, execute=90), 0.16), atol=1e-6)


def test_lag_compensation_restores_the_plan_and_is_a_no_op_at_response_one(monkeypatch):
    fast = _drive(_policy(monkeypatch, execute=30), 1.0)
    compensated = _drive(_policy(monkeypatch, execute=30, servo_response=SLOW), 0.16)
    assert np.allclose(fast, compensated, atol=1e-4)
    untouched = _drive(_policy(monkeypatch, execute=30, servo_response=(1.0,) * 6), 1.0)
    assert np.array_equal(fast, untouched)


def test_lag_compensation_keeps_a_resting_offset_in_the_anchor(monkeypatch):
    # Gravity sag is part of the measured pose in training; the anchor must not drop it (a command anchor does,
    # and walks away by the sag at every re-read).
    plain = _drive(_policy(monkeypatch, execute=30), 1.0, sag=0.03)
    compensated = _drive(_policy(monkeypatch, execute=30, servo_response=SLOW), 0.16, sag=0.03)
    assert np.allclose(plain, compensated, atol=1e-3)        # dropping the sag would drift by 0.03 per re-read


def test_correction_is_capped_and_the_gripper_is_never_corrected(monkeypatch):
    policy = _policy(monkeypatch, execute=30, servo_response=(0.05,) * 6)
    policy.reset()
    start = np.array([0.0, -2.8, 2.8, 1.4, 0.0, 0.2], np.float32)
    image = np.zeros((256, 256, 3), np.uint8)
    for _ in range(30):                                  # the arm is blocked: the measurement never moves
        policy.predict(image if policy.needs_frame else None, start)
    simulated, real, _ = policy._models
    assert abs(simulated[2] - real[2]) > 0.0             # the models have diverged on the moving joint
    anchor = policy._anchor(start.copy())
    assert np.all(np.abs(anchor - start) <= ANCHOR_LEAD_CAP_ACT + 1e-6) and anchor[5] == start[5]
    with pytest.raises(ValueError):
        _policy(monkeypatch, execute=30, servo_response=(1.0, 1.0, 0.0, 1.0, 1.0, 1.0))


def test_first_execute_runs_the_whole_first_chunk_from_one_read(monkeypatch):
    policy = _policy(monkeypatch, execute=30, first_execute=90)
    policy.reset()
    reads = []
    measured = np.array([0.0, -2.8, 2.8, 1.4, 0.0, 0.2], np.float32)
    for step in range(150):
        if policy.needs_frame:
            reads.append(step)
        policy.predict(np.zeros((256, 256, 3), np.uint8), measured)
    assert reads == [0, 90, 120]
    assert policy.policy_id == "pi05.exec30.first90.gripper_clamp_v1"
    assert _policy(monkeypatch, execute=30, servo_response=SLOW).policy_id == "pi05.exec30.lagcomp.gripper_clamp_v1"
    assert _policy(monkeypatch, execute=30).policy_id == "pi05.exec30.gripper_clamp_v1"


def test_lag_stress_switch_slows_the_simulated_joints_and_defaults_off():
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.lag_stress import LagStressPolicy

    class Ramp:
        policy_id = "ramp"
        requires_pixels = False

        def reset(self, adapter=None):
            self.k = 0

        def predict(self, image, current, adapter=None):
            self.k += 1
            command = np.asarray(self.start, np.float32).copy()
            command[2] -= 0.02 * self.k
            return command

    def run(policy):
        adapter = MujocoTaskAdapter(SCENE)
        try:
            adapter.reset(bench_suite(adapter.bench, [(0, 0)], label="lagtest", repeats=1).scenarios[0])
            inner = getattr(policy, "policy", policy)
            inner.start = adapter.current_act()
            policy.reset(adapter)
            for _ in range(20):
                adapter.apply_policy_command(policy.predict(None, adapter.current_act(), adapter))
                adapter.advance_control_period()
            return adapter.current_act(), adapter.command_response
        finally:
            adapter.close()

    plain, switch = run(Ramp())
    assert switch is None
    lagged, switch = run(LagStressPolicy(Ramp(), (1.0, 1.0, 0.16, 1.0, 1.0, 1.0)))
    assert switch is not None and lagged[2] - plain[2] > 0.05          # the elbow is further behind the same ramp
    assert LagStressPolicy(Ramp()).policy_id == "ramp.lagstress"
    with pytest.raises(ValueError):
        LagStressPolicy(Ramp(), (1.0, 1.0, 0.0, 1.0, 1.0, 1.0))
