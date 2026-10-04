"""Servo assist: make the real joints reach what a policy commands, as the simulator's stiff actuators do.

The real STS3215 servos run a proportional position loop with no integral term (lerobot's configuration) on a
5 V supply. Commanded exactly at a pose, a joint settles short of it by (gravity torque + friction) / stiffness:
1.5 to 6.4 units on the elbow, and the base and wrist ignore corrections of a couple of units (servo probe and
teacher replays, 2026-10-04). Policies are trained in a simulator whose actuators reach their commands, so on the
arm their plans stop short, look at the wrong place and close beside the cube (physical/episode_20).

``ServoAssist`` sits between the policy and the bench gate and turns the policy's target into the command that
gets the measured joint there. Per arm joint (the gripper is left alone: it is meant to stall on the cube):

- gravity: the gravity torque at the target pose, from the scene's mass model, over the measured stiffness;
- friction: the measured dry friction over the stiffness, in the direction the target last moved (turned over a
  few steps when it reverses, kept while it rests);
- a slow integral of the target-minus-measured error, taken only while the target is nearly at rest and capped:
  whatever the two model terms leave (a cable, stiction the model does not have) is pushed out at rest.

The command never leads the measured joint by more than ``max_lead_units`` (the gate holds at 20). The stiffness
and friction come from ``tools/fit_servo_model.py`` (resource ``servo_model_20261004.json``); the tuning is the
one found for the same law on the measured-servo scene, where it takes the scripted teacher from 1 of 10
certification placements to 10 of 10.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.data.resources import load_json_resource

DEFAULT_SERVO_RESOURCE = "servo_model_20261004.json"


@dataclass(frozen=True)
class AssistTuning:
    direction_slew: float = 0.1                 # per step: how fast the friction term turns when the target reverses
    move_threshold_rad_per_step: float = 5e-4   # a target moving slower than this keeps its last direction
    integral_gain: float = 0.03                 # per step, on the target-minus-measured error (rad)
    integral_cap_rad: float = 0.05
    integral_rest_rad_per_step: float = 0.002   # the error is integrated only while the target moves slower than this
    max_lead_units: float = 18.0
    friction_scale: float = 1.0                 # 0 drops the friction term (gravity and integral only)


class ServoAssist:
    def __init__(self, model_path: str | Path, bench: Any, servo: dict[str, Any] | None = None, tuning: AssistTuning = AssistTuning()) -> None:
        import mujoco

        self._mujoco = mujoco
        servo = load_json_resource(DEFAULT_SERVO_RESOURCE) if servo is None else servo
        self.kp = np.asarray(servo["kp"], dtype=np.float64)
        self.friction = np.asarray(servo["frictionloss"], dtype=np.float64)
        if self.kp.shape != (6,) or self.friction.shape != (6,) or np.any(self.kp <= 0) or np.any(self.friction < 0):
            raise ValueError("servo model needs six positive stiffnesses and six non-negative frictions")
        self.source = str(servo.get("name") or servo.get("source") or "inline")
        self.tuning = tuning
        self.joint_map = bench.joint_map_object
        self.low, self.high = np.asarray(bench.mujoco_low, dtype=np.float64), np.asarray(bench.mujoco_high, dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._qpos = [int(self.model.joint(name).qposadr[0]) for name in JOINT_NAMES]
        self._dofs = [int(self.model.joint(name).dofadr[0]) for name in JOINT_NAMES]
        self._rest_qpos = self.data.qpos.copy()
        self.reset()

    def reset(self) -> None:
        self._direction = np.zeros(6)
        self._previous: np.ndarray | None = None
        self._integral = np.zeros(6)
        self.max_offset_units = np.zeros(6)

    def _gravity(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self._rest_qpos
        self.data.qvel[:] = 0.0
        for address, value in zip(self._qpos, qpos):
            self.data.qpos[address] = float(value)
        self._mujoco.mj_forward(self.model, self.data)
        return np.asarray([self.data.qfrc_bias[dof] for dof in self._dofs], dtype=np.float64)

    def command(self, target_act: np.ndarray, measured_act: np.ndarray) -> np.ndarray:
        """The command (ACT) that brings the measured joints to ``target_act``."""
        tuning = self.tuning
        target_act = np.asarray(target_act, dtype=np.float32)
        if not np.all(np.isfinite(target_act)):
            return target_act                                   # the gate turns a nonfinite request into a hold
        plan = np.asarray(self.joint_map.act_to_mujoco(target_act), dtype=np.float64)
        measured = np.asarray(self.joint_map.act_to_mujoco(np.asarray(measured_act, dtype=np.float32)), dtype=np.float64)
        velocity = np.zeros(6) if self._previous is None else plan - self._previous
        wanted = np.where(np.abs(velocity) > tuning.move_threshold_rad_per_step, np.sign(velocity), self._direction)
        self._direction = self._direction + np.clip(wanted - self._direction, -tuning.direction_slew, tuning.direction_slew)
        self._previous = plan.copy()
        resting = np.abs(velocity) < tuning.integral_rest_rad_per_step
        self._integral = np.clip(self._integral + resting * tuning.integral_gain * (plan - measured), -tuning.integral_cap_rad, tuning.integral_cap_rad)
        offset = (self._gravity(plan) + tuning.friction_scale * self.friction * self._direction) / self.kp + self._integral
        offset[5] = 0.0
        margin = 1e-3
        command = np.clip(plan + offset, np.minimum(self.low + margin, plan), np.maximum(self.high - margin, plan))
        act = np.asarray(self.joint_map.mujoco_to_act(command.astype(np.float32)), dtype=np.float64)
        current = np.asarray(measured_act, dtype=np.float64)
        cap = tuning.max_lead_units * np.pi / 100.0
        act[:5] = np.clip(act[:5], current[:5] - cap, current[:5] + cap)
        act[5] = float(target_act[5])
        self.max_offset_units = np.maximum(self.max_offset_units, np.abs(act - np.asarray(target_act, dtype=np.float64)) * 100.0 / np.pi)
        return act.astype(np.float32)

    def record(self) -> dict[str, Any]:
        return dict(servo_model=self.source, kp=self.kp.tolist(), frictionloss=self.friction.tolist(), tuning=asdict(self.tuning),
                    max_offset_units=[round(float(v), 2) for v in self.max_offset_units])


class AssistedPolicy:
    """A rollout policy with the assist applied to its output: how the runner drives the arm, for simulator evaluations."""

    def __init__(self, policy: Any, assist: ServoAssist, *, tag: str = "assist") -> None:
        self.policy, self.assist, self.tag = policy, assist, tag
        self.requires_pixels = bool(getattr(policy, "requires_pixels", True))

    @property
    def policy_id(self) -> str:
        return f"{self.policy.policy_id}.{self.tag}"

    @property
    def needs_frame(self) -> bool:
        return bool(getattr(self.policy, "needs_frame", True))

    def reset(self, adapter: Any | None = None) -> None:
        self.assist.reset()
        try:
            self.policy.reset(adapter)
        except TypeError:
            self.policy.reset()

    def predict(self, image: Any, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        return self.assist.command(self.policy.predict(image, current_act, adapter), current_act)


__all__ = ["AssistTuning", "AssistedPolicy", "DEFAULT_SERVO_RESOURCE", "ServoAssist"]
