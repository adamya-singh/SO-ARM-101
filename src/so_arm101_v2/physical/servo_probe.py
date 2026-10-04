"""Scripted motion that measures how each real servo follows its command (no network, no camera).

Run through the physical runner like a policy (``tools/run_physical_episode.py --servo-probe``): the same
approach, bench gate, guards, 30 Hz loop and step log. The commands are offsets from the pose measured at the
first step, in three parts:

1. the survey move (reset pose to viewing pose over 60 steps, then 30 steps of hold), the move that the
   first live pi0.5 episode failed to finish;
2. one joint at a time around the viewing pose: out by its amplitude at 1 unit per step, hold, back through
   to the other side, hold, return, hold;
3. the gripper: open by 15 units and back.

Everything happens in the air above the bench. ``tools/fit_servo_lag.py`` fits the response from the run's
steps.csv by replaying the same commands in the simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act

SURVEY_RAMP_STEPS = 60
SURVEY_HOLD_STEPS = 30
WIGGLE_UNITS = ((0, 10.0), (1, 8.0), (2, 10.0), (3, 7.0), (4, 10.0))     # (joint index, amplitude in physical units)
WIGGLE_HOLD_STEPS = 20
GRIPPER_UNITS = 15.0


def probe_offsets(bench: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Offsets (physical units, shape (T, 6)) from the start pose, and the segments for the record."""
    reset = np.asarray(bench.reset_physical, dtype=np.float64)
    viewing = np.asarray(act_to_physical_normalized(bench.joint_map_object.mujoco_to_act(np.asarray(bench.viewing_qpos, np.float32))), dtype=np.float64)
    survey = viewing - reset
    survey[5] = 0.0
    rows: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []

    def add(name: str, block: list[np.ndarray]) -> None:
        segments.append(dict(name=name, start=len(rows), stop=len(rows) + len(block)))
        rows.extend(block)

    ramp = [survey * 0.5 * (1.0 - np.cos(np.pi * (k + 1) / SURVEY_RAMP_STEPS)) for k in range(SURVEY_RAMP_STEPS)]
    add("survey_move", ramp + [survey.copy() for _ in range(SURVEY_HOLD_STEPS)])

    def wiggle(joint: int, amplitude: float, two_sided: bool) -> list[np.ndarray]:
        profile = list(np.arange(1.0, amplitude + 0.5)) + [amplitude] * WIGGLE_HOLD_STEPS
        if two_sided:
            profile += list(np.arange(amplitude - 1.0, -amplitude - 0.5, -1.0)) + [-amplitude] * WIGGLE_HOLD_STEPS
            profile += list(np.arange(-amplitude + 1.0, 0.5))
        else:
            profile += list(np.arange(amplitude - 1.0, -0.5, -1.0))
        profile += [0.0] * WIGGLE_HOLD_STEPS
        block = []
        for value in profile:
            offset = survey.copy()
            offset[joint] += value
            block.append(offset)
        return block

    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    for joint, amplitude in WIGGLE_UNITS:
        add(f"wiggle_{names[joint]}", wiggle(joint, amplitude, True))
    add("wiggle_gripper", wiggle(5, GRIPPER_UNITS, False))
    return np.asarray(rows), segments


class ServoProbePolicy:
    requires_pixels = False
    needs_frame = False
    report_content_sha256 = None
    chunk_horizon = SURVEY_RAMP_STEPS + SURVEY_HOLD_STEPS

    def __init__(self, bench: Any) -> None:
        self.offsets, self.segments = probe_offsets(bench)
        self.teacher_horizon = len(self.offsets)
        self.policy_id = "servo_probe_v1"
        self.reset()

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._start: np.ndarray | None = None

    def predict(self, image: Any, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del image, adapter
        if self._start is None:
            self._start = np.asarray(act_to_physical_normalized(np.asarray(current_act, dtype=np.float32)), dtype=np.float64)
        index = min(self.action_index, len(self.offsets) - 1)
        self.action_index += 1
        return physical_normalized_to_act((self._start + self.offsets[index]).astype(np.float32))


__all__ = ["ServoProbePolicy", "probe_offsets"]
