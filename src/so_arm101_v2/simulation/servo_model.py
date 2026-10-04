"""Per-joint servo parameters of the simulated arm, and how to put a measured set on a loaded model.

The certified scene drives every joint with the same position actuator (kp 50 N·m/rad, critical damping) and
0.1 N·m of joint friction. The real STS3215 servos (5 V supply, lerobot's P gain of 16, no integral term) settle
short of a command by an amount that depends on the load and on the direction they arrived from: the elbow
rests 1.5 to 6.4 units from its command at the viewing pose, the other joints 0 to 1.5 (servo probe,
2026-10-04, four runs within 0.2 units of each other). ``ServoModel`` holds a stiffness, a damping and a dry
friction per joint; ``tools/fit_servo_model.py`` fits them to the probe by replaying its commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from so_arm101_v2.contracts.coordinates import JOINT_NAMES


@dataclass(frozen=True)
class ServoModel:
    kp: tuple[float, ...]             # actuator stiffness, N·m/rad
    kv: tuple[float, ...]             # actuator damping, N·m·s/rad
    frictionloss: tuple[float, ...]   # joint dry friction, N·m

    def __post_init__(self) -> None:
        for name in ("kp", "kv", "frictionloss"):
            values = getattr(self, name)
            if len(values) != 6 or any(not np.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"{name} must be six finite non-negative values")
        if any(v <= 0 for v in self.kp):
            raise ValueError("kp must be positive")


def model_servo(model: Any) -> ServoModel:
    """The servo parameters a loaded MuJoCo model carries."""
    kp, kv, friction = [], [], []
    for name in JOINT_NAMES:
        actuator = model.actuator(name)
        kp.append(float(actuator.gainprm[0]))
        kv.append(float(-actuator.biasprm[2]))
        friction.append(float(model.dof_frictionloss[model.joint(name).dofadr[0]]))
    return ServoModel(tuple(kp), tuple(kv), tuple(friction))


def apply_servo_model(model: Any, servo: ServoModel) -> None:
    """Overwrite the position actuators' gains and the joints' dry friction on a loaded model (in place)."""
    for index, name in enumerate(JOINT_NAMES):
        actuator = model.actuator(name).id
        model.actuator_gainprm[actuator, 0] = servo.kp[index]
        model.actuator_biasprm[actuator, 1] = -servo.kp[index]
        model.actuator_biasprm[actuator, 2] = -servo.kv[index]
        model.dof_frictionloss[model.joint(name).dofadr[0]] = servo.frictionloss[index]


__all__ = ["ServoModel", "apply_servo_model", "model_servo"]
