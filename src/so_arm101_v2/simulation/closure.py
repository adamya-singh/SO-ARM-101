"""Closure error in millimetres from rollout telemetry.

Promotes the 2026-09-18 tranche-analysis measurement to a tool: the jaw midpoint (mean of the two
jaw-tip sites, forward kinematics on the logged joint angles) relative to the cube's start position
at step 250, just before closure, in the square's frame (along / perp / height). The target is the
working fixed-square policy's point; the grasp tolerance is about 8 mm lateral.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

from so_arm101_v2.contracts.coordinates import JOINT_NAMES

# Working fixed-square policy (ytn3eygr), 10/10 held-out poses, tranche_analysis_20260918.
REFERENCE_ALONG_MM = 14.874101267023972
REFERENCE_PERP_MM = 9.38633536739638
REFERENCE_HEIGHT_MM = 32.58821738316333
REFERENCE_YAW_DEG = 74.23836165372643
CLOSURE_STEP = 250
NOT_DESCENDED_MM = 6.0
GRASP_TOLERANCE_MM = 8.0


class ClosureKinematics:
    """Forward kinematics of the jaw-tip sites for logged joint angles."""

    def __init__(self, model_path: str | Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._qpos = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in JOINT_NAMES
        ]

    def jaw(self, robot_qpos: Iterable[float], gripper_qpos: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (jaw midpoint, jaw axis fixed->moving) in world coordinates, metres."""
        qpos = np.asarray(list(robot_qpos), dtype=np.float64)
        if gripper_qpos is not None:
            qpos[5] = gripper_qpos
        self.data.qpos[:] = 0.0
        self.data.qpos[self._qpos] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        fixed = self.data.site("fixed_jaw_tip").xpos.copy()
        moving = self.data.site("moving_jaw_tip").xpos.copy()
        return (fixed + moving) / 2.0, moving - fixed


def closure_row(kin: ClosureKinematics, telemetry: Mapping[str, Any], step: int = CLOSURE_STEP) -> dict[str, float]:
    """Jaw midpoint versus the cube's start position at telemetry row ``step``, in the jaw-axis frame.

    Matches tranche_analysis_20260918 exactly: the gripper angle is taken from row 200 (still open,
    so the midpoint is the grasp centre); along = component on the jaw axis, perp = across it.
    """
    rows = telemetry["rows"]
    row = rows[min(step, len(rows) - 1)]
    mid, axis3 = kin.jaw(row["robot_qpos"][:6], rows[min(200, len(rows) - 1)]["robot_qpos"][5])
    axis = axis3[:2] / (np.linalg.norm(axis3[:2]) + 1e-12)
    perp_axis = np.array([-axis[1], axis[0]])
    cube0 = np.asarray(rows[0]["cube_position"], dtype=np.float64)
    delta = (mid[:2] - cube0[:2]) * 1000.0
    along = float(delta @ axis)
    perp = float(delta @ perp_axis)
    height = float((mid[2] - cube0[2]) * 1000.0)
    yaw = float(telemetry["placement"]["square_yaw_rad"])
    rel_yaw = (math.degrees(math.atan2(axis[1], axis[0])) - math.degrees(yaw)) % 90.0
    err = (rel_yaw - REFERENCE_YAW_DEG) % 90.0
    return {
        "along": along,
        "perp": perp,
        "height": height,
        "rel_yaw": rel_yaw,
        "lat": math.hypot(along - REFERENCE_ALONG_MM, perp - REFERENCE_PERP_MM),
        "dz": height - REFERENCE_HEIGHT_MM,
        "yaw_err": min(err, 90.0 - err),
    }


def closure_summary(model_path: str | Path, telemetry_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Per-rollout closure rows plus the summary numbers the notebook reports."""
    kin = ClosureKinematics(model_path)
    rows = []
    for path in telemetry_paths:
        telemetry = json.loads(Path(path).read_text())
        if len(telemetry["rows"]) < CLOSURE_STEP:
            continue
        row = closure_row(kin, telemetry)
        row["scenario_id"] = telemetry["scenario_id"]
        row["repeat"] = int(telemetry.get("repeat", 0))
        rows.append(row)
    if not rows:
        return {"rows": [], "count": 0}
    lat = np.array([r["lat"] for r in rows])
    descended = np.array([r["dz"] <= NOT_DESCENDED_MM for r in rows])
    return {
        "rows": rows,
        "count": len(rows),
        "lateral_median_mm": float(np.median(lat)),
        "lateral_rms_mm": float(np.sqrt(np.mean(lat**2))),
        "lateral_max_mm": float(lat.max()),
        "within_tolerance": int(np.sum((lat <= GRASP_TOLERANCE_MM) & descended)),
        "not_descended": int(np.sum(~descended)),
        "yaw_err_median_deg": float(np.median([r["yaw_err"] for r in rows])),
        "dz_median_mm": float(np.median([abs(r["dz"]) for r in rows])),
    }
