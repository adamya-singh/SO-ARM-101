"""Servo-lag stress evaluation: run a policy in the simulator with real-like actuator lag.

The certified scene's position actuators follow a command ramp about 3.5 control steps behind; the real
elbow follows 6-13 steps behind (fit on physical/episode_10, _13 and _20, 2026-10-04). A policy that
replans from the measured pose mid-move is fine on the first and fails on the second: the first live
pi0.5 episode stopped its survey move 13 elbow units short and never saw the cube.

``LagStressPolicy`` wraps any rollout policy and, at reset, puts a per-joint first-order response in
front of the adapter's actuators (``MujocoTaskAdapter.command_response``): each control step the actuator
target moves ``response`` of the way to the command. 1.0 leaves a joint untouched. Nothing here changes
the scene, a capture or an identity: it is an evaluation condition, carried in the policy id.

``TimeDilationPolicy`` is the matching remedy on the runner's side, evaluated here before the arm: every
command is held for ``repeat`` control periods, so the plan runs ``repeat`` times slower and a slow servo is
proportionally closer to its command when the next one arrives. The policy still counts one step per
command; its frame and measured pose are taken after the last held period.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Fit by replaying episode_20's commands open-loop in the simulator (tools/fit_servo_lag.py, 2026-10-04):
# the value per joint that minimises the rms difference to the measured real joints over steps 0-240.
# episode_20_20261004_pi05: pan 0.75, lift 0.85, elbow 0.16, wrist flex 0.70, wrist roll 0.65, gripper 0.60. Only the elbow
# changes the fit much (rms 3.15 -> 2.31 units); episodes 10 and 13 (2026-09-10) give a milder elbow, 0.40-0.45.
REAL_SERVO_RESPONSE_20261004: tuple[float, ...] = (0.75, 0.85, 0.16, 0.70, 0.65, 0.60)
REAL_SERVO_RESPONSE_20260910: tuple[float, ...] = (0.40, 0.90, 0.45, 0.60, 0.65, 0.65)


class LagStressPolicy:
    def __init__(self, policy: Any, response: tuple[float, ...] = REAL_SERVO_RESPONSE_20261004, *, tag: str = "lagstress") -> None:
        response_array = np.asarray(response, dtype=np.float64)
        if response_array.shape != (6,) or np.any(response_array <= 0.0) or np.any(response_array > 1.0):
            raise ValueError("response must be six values in (0, 1]")
        self.policy, self.response, self.tag = policy, response_array, tag
        self.requires_pixels = bool(getattr(policy, "requires_pixels", True))

    @property
    def policy_id(self) -> str:
        return f"{self.policy.policy_id}.{self.tag}"

    @property
    def needs_frame(self) -> bool:
        return bool(getattr(self.policy, "needs_frame", True))

    def reset(self, adapter: Any | None = None) -> None:
        if adapter is None:
            raise ValueError("LagStressPolicy needs the rollout's adapter at reset")
        adapter.command_response = self.response.copy()
        self.policy.reset(adapter)

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        return self.policy.predict(image, current_act, adapter)


class TimeDilationPolicy:
    """Hold each command of ``policy`` for ``repeat`` control periods (the rollout advances one; this adds the rest)."""

    def __init__(self, policy: Any, repeat: int, assist: Any | None = None) -> None:
        if int(repeat) < 1:
            raise ValueError("repeat must be at least 1")
        self.policy, self.repeat, self.assist = policy, int(repeat), assist
        self.requires_pixels = bool(getattr(policy, "requires_pixels", True))
        self._last: np.ndarray | None = None

    @property
    def policy_id(self) -> str:
        return f"{self.policy.policy_id}.hold{self.repeat}" + ("" if self.assist is None else ".assist")

    @property
    def needs_frame(self) -> bool:
        return bool(getattr(self.policy, "needs_frame", True))

    def reset(self, adapter: Any | None = None) -> None:
        self._last = None
        if self.assist is not None:
            self.assist.reset()
        self.policy.reset(adapter)

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        if self.repeat > 1 and self._last is not None:
            if adapter is None:
                raise ValueError("TimeDilationPolicy needs the rollout's adapter")
            for _ in range(self.repeat - 1):
                # The same target again (keeps a lag stress response in step); the servo assist, when given, runs
                # every control period, as it would in the runner.
                held = self._last if self.assist is None else self.assist.command(self._last, adapter.current_act())
                adapter.apply_policy_command(held)
                adapter.advance_control_period()
            current_act = adapter.current_act()
            if self.needs_frame and self.requires_pixels:
                image = adapter.render_wrist_observation()
        command = self.policy.predict(image, current_act, adapter)
        self._last = np.asarray(command, dtype=np.float32).copy()
        return command if self.assist is None else self.assist.command(command, current_act)


__all__ = ["LagStressPolicy", "REAL_SERVO_RESPONSE_20260910", "REAL_SERVO_RESPONSE_20261004", "TimeDilationPolicy"]
