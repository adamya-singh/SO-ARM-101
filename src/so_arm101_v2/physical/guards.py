"""Episode guards for the physical runner: stop a policy that has left the task, whatever it outputs.

The bench gate (``contracts/physical.py``) checks each command against joint ranges, the shoulder floor and
the per-step speed. Every command of physical/episode_20_20261004_pi05 passed it while the arm reached down to
the table beside the base and pressed on it: the survey frame held no towel, and nothing checked that. These
guards close that gap. A trip ends the episode: the runner sends the measured pose (so the servos stop pushing)
and returns ``guard:<reason>``. Torque stays on.

Checks, in the order they run each step:

- ``dark_frame``: a camera read whose observation is nearly black (episode_16 ran on such frames).
- ``survey_pose_not_reached`` / ``survey_target_not_visible``: at the survey read (``bench.observation_steps``)
  the arm must be at the viewing pose and the towel and cube must be in the frame. ``survey_gate=False`` records
  both results without stopping on them (a fixed-view policy from before 2026-09-10 never makes the survey move).
- ``workspace_height`` / ``workspace_xy``: forward kinematics of the *commanded* pose through the joint map.
  The lower jaw tip may not go below ``min_tip_height_m`` and the jaw midpoint must stay over the placement
  rectangle plus a margin. Calibration (2026-10-04): the teacher's commands over 2400 placements keep the tip at
  35 mm or more and the midpoint in x -0.195..0.206, y 0.111..0.335 m; successful pi0.5 rollouts go down to
  30 mm; episode_10 (the live success) 36 mm; episode_20 commanded 4 mm.
- ``stall:<joint>``: an arm joint has stayed more than ``stall_error_units`` from its command for
  ``stall_steps`` steps without moving. No recorded episode trips it (the worst resting error on record is
  5.1 units, shoulder under gravity); it is the backstop for a blocked joint.

``stop_after_survey`` ends the episode at the survey read once those checks pass (``survey_only``): the staged
first run for a new policy class.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import JOINT_NAMES
from so_arm101_v2.contracts.physical import act_to_physical_normalized, physical_normalized_to_act

SURVEY_ONLY = "survey_only"


@dataclass(frozen=True)
class GuardLimits:
    min_tip_height_m: float = 0.022
    workspace_side_margin_m: float = 0.06       # beyond the rectangle's x range
    workspace_near_margin_m: float = 0.035      # nearer the base than the rectangle's near edge
    workspace_far_margin_m: float = 0.0         # beyond the rectangle's far edge (the teacher stops 34 mm short of it)
    stall_error_units: float = 7.0
    stall_steps: int = 10
    stall_motion_units: float = 1.5
    survey_tolerance_rad: float = 0.10
    min_frame_p99: float = 40.0
    min_frame_mean: float = 8.0


def frame_brightness(observation: np.ndarray, limits: GuardLimits = GuardLimits()) -> dict[str, Any]:
    """Mean and 99th percentile of the 256x256 observation. Lights off reads mean 1.4 / p99 4; the bench lit, 35-70 / 180-250."""
    grey = np.asarray(observation, dtype=np.float64).mean(-1)
    mean, p99 = float(grey.mean()), float(np.percentile(grey, 99))
    return dict(mean=round(mean, 1), p99=round(p99, 1), min_mean=limits.min_frame_mean, min_p99=limits.min_frame_p99,
                ok=bool(mean >= limits.min_frame_mean and p99 >= limits.min_frame_p99))


def survey_target_visible(observation: np.ndarray, *, min_towel_px: int = 150, min_cube_px: int = 25) -> dict[str, Any]:
    """Is something towel-and-cube-like in the frame: a bright, mostly convex patch with a compact dark blob inside its outline?

    A necessary condition for the survey frame, not a localiser (``placement.locate_cube`` assumes the centred layout
    and misses placements at the sides). Checked 2026-10-04: found in 220/220 simulated survey frames (held-out,
    certification and 200 training placements, randomised appearance) and in the real reset frames of episodes 4, 10,
    13 and 20; not found in episode_20's survey frame (bare mat).
    """
    import cv2

    grey = np.asarray(observation, dtype=np.float64).mean(-1)
    p99 = float(np.percentile(grey, 99))
    bright = (grey > max(90.0, 0.7 * p99)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    best: dict[str, Any] | None = None
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_towel_px:
            continue
        hull = np.zeros_like(bright)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero((labels == index).astype(np.uint8))), 1)
        dark = (hull == 1) & (labels != index) & (grey < min(70.0, 0.45 * p99))
        blobs, _, blob_stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
        cube = max((int(blob_stats[k, cv2.CC_STAT_AREA]) for k in range(1, blobs)), default=0)
        fill = area / float(hull.sum())
        if cube >= min_cube_px and fill >= 0.5 and (best is None or cube > best["cube_px"]):
            best = dict(towel_px=area, cube_px=cube, fill=round(fill, 2), bbox=[int(v) for v in stats[index, :4]])
    return dict(found=best is not None, p99=round(p99, 1), **(best or {}))


class _Kinematics:
    """Jaw tips of the bench scene at simulator joint angles (kinematics only, one loaded model)."""

    def __init__(self, model_path: str | Path) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._qpos = [int(self.model.joint(name).qposadr[0]) for name in JOINT_NAMES]

    def _set(self, qpos: np.ndarray) -> None:
        for address, value in zip(self._qpos, qpos):
            self.data.qpos[address] = float(value)
        self._mujoco.mj_kinematics(self.model, self.data)

    def jaw(self, qpos: np.ndarray) -> tuple[np.ndarray, float]:
        """(jaw midpoint xyz, height of the lower jaw tip), metres."""
        self._set(qpos)
        fixed = self.data.site("fixed_jaw_tip").xpos
        moving = self.data.site("moving_jaw_tip").xpos
        return (fixed + moving) / 2.0, float(min(fixed[2], moving[2]))


class EpisodeGuards:
    def __init__(self, model_path: str | Path, bench: Any, limits: GuardLimits = GuardLimits(), *,
                 survey_gate: bool = True, stop_after_survey: bool = False) -> None:
        from so_arm101_v2.physical.dry_pass import survey_pose_active

        self.bench, self.limits = bench, limits
        self.survey_gate, self.stop_after_survey = bool(survey_gate), bool(stop_after_survey)
        self.joint_map = bench.joint_map_object
        self.kinematics = _Kinematics(model_path)
        self.survey_step = int(bench.observation_steps) if survey_pose_active(bench) else None
        if stop_after_survey and self.survey_step is None:
            raise ValueError("stop_after_survey needs a bench with a survey pose")
        if bench.placement is not None:
            x, y = bench.placement["x_range_m"], bench.placement["y_range_m"]
            self.box = (float(x[0]) - limits.workspace_side_margin_m, float(x[1]) + limits.workspace_side_margin_m,
                        float(y[0]) - limits.workspace_near_margin_m, float(y[1]) + limits.workspace_far_margin_m)
        else:
            self.box = None
        self.reset()

    def reset(self) -> None:
        self._last_sent: np.ndarray | None = None
        self._errors: deque[np.ndarray] = deque(maxlen=self.limits.stall_steps)
        self._poses: deque[np.ndarray] = deque(maxlen=self.limits.stall_steps + 1)
        self.survey: dict[str, Any] | None = None
        self.trip: dict[str, Any] | None = None
        self.min_tip_height_m: float | None = None

    # ------------------------------------------------------------------ individual checks
    def _check_survey(self, measured_act: np.ndarray, observation: np.ndarray | None) -> str | None:
        qpos = np.asarray(self.joint_map.act_to_mujoco(measured_act), dtype=np.float64)
        delta = qpos[:5] - np.asarray(self.bench.viewing_qpos, dtype=np.float64)[:5]
        worst = int(np.argmax(np.abs(delta)))
        self.survey = dict(max_abs_rad=round(float(np.abs(delta).max()), 4), joint=JOINT_NAMES[worst],
                           tolerance_rad=self.limits.survey_tolerance_rad,
                           pose_ok=bool(np.abs(delta).max() <= self.limits.survey_tolerance_rad))
        if observation is not None:
            found = survey_target_visible(observation)
            self.survey["target"] = found
            self.survey["target_ok"] = bool(found.get("found"))
        if self.survey_gate and not self.survey["pose_ok"]:
            return "survey_pose_not_reached"
        if self.survey_gate and self.survey.get("target_ok") is False:
            return "survey_target_not_visible"
        return SURVEY_ONLY if self.stop_after_survey else None

    def _check_workspace(self, sent_physical: np.ndarray) -> tuple[str | None, dict[str, Any]]:
        qpos = self.joint_map.act_to_mujoco(physical_normalized_to_act(np.asarray(sent_physical, dtype=np.float32)))
        middle, tip = self.kinematics.jaw(qpos)
        self.min_tip_height_m = tip if self.min_tip_height_m is None else min(self.min_tip_height_m, tip)
        detail = dict(jaw_midpoint_m=[round(float(v), 4) for v in middle], tip_height_m=round(tip, 4))
        if tip < self.limits.min_tip_height_m:
            return "workspace_height", detail
        if self.box is not None and not (self.box[0] <= middle[0] <= self.box[1] and self.box[2] <= middle[1] <= self.box[3]):
            return "workspace_xy", detail
        return None, detail

    def _check_stall(self, measured_physical: np.ndarray) -> str | None:
        self._poses.append(measured_physical[:5].copy())
        if self._last_sent is None:
            return None
        self._errors.append(np.abs(self._last_sent[:5] - measured_physical[:5]))
        if len(self._errors) < self.limits.stall_steps:
            return None
        stuck = np.min(np.asarray(self._errors), axis=0) > self.limits.stall_error_units
        still = np.abs(self._poses[-1] - self._poses[0]) < self.limits.stall_motion_units
        joints = np.nonzero(stuck & still)[0]
        return f"stall:{JOINT_NAMES[int(joints[0])]}" if len(joints) else None

    # ------------------------------------------------------------------ the runner's hook
    def before_send(self, step: int, measured_act: np.ndarray, decision: Any, observation: Any | None) -> str | None:
        """Reason to end the episode before this step's command is sent, or None. ``observation`` is the runner's Observation."""
        measured_act = np.asarray(measured_act, dtype=np.float32)
        image = None if observation is None else np.asarray(observation.image)
        reason, detail = None, {}
        if image is not None:
            brightness = frame_brightness(image, self.limits)
            if not brightness["ok"]:
                reason, detail = "dark_frame", brightness
        if reason is None and self.survey_step is not None and step == self.survey_step:
            reason = self._check_survey(measured_act, image)
            detail = dict(self.survey or {})
        stall = self._check_stall(np.asarray(act_to_physical_normalized(measured_act), dtype=np.float64))
        if reason is None and stall is not None:
            reason = stall
        if reason is None and not decision.held:
            reason, detail = self._check_workspace(decision.sent_physical)
        if reason is not None:
            self.trip = dict(step=int(step), reason=reason, **({"detail": detail} if detail else {}))
            return reason
        self._last_sent = np.asarray(decision.sent_physical, dtype=np.float64).copy()
        return None

    def record(self) -> dict[str, Any]:
        return dict(limits=asdict(self.limits), workspace_box_m=None if self.box is None else [round(v, 4) for v in self.box],
                    survey_step=self.survey_step, survey_gate=self.survey_gate, stop_after_survey=self.stop_after_survey,
                    survey=self.survey, trip=self.trip,
                    min_commanded_tip_height_m=None if self.min_tip_height_m is None else round(self.min_tip_height_m, 4))


__all__ = ["EpisodeGuards", "GuardLimits", "SURVEY_ONLY", "frame_brightness", "survey_target_visible"]
