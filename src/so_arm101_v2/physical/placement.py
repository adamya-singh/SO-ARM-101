"""Cube placement from a wrist observation: find the white towel and the dark cube, back-project the cube onto the bench.

The policies train on cube poses within ±10 mm of the task pose (the square's
centre 8.5 in forward of the base front edge). On 2026-09-10 the cube sat
~40 mm beyond it (the towel's near edge was at the mark) and the policy
executed the whole plan and closed on nothing. This module turns the
observation the runner already takes at the reset pose into a bench
coordinate for the cube, so the runner can refuse a trial the policy cannot
win and say by how much to move the cube. Geometry: the calibrated lens
(``LensModel``) and the simulated wrist-camera pose at the measured joints;
validated on the simulated reset frame (square near edge reproduced to
0.2 mm) and on the 2026-09-10 real frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np

CUBE_TOP_Z_M = 0.020
DEFAULT_TOLERANCE_MM = 15.0


def camera_pose_at(model_path, bench, current_act) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation of the wrist camera at the measured joints (forward kinematics only)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos = bench.joint_map_object.act_to_mujoco(np.asarray(current_act, dtype=np.float32))
    for name, value in zip(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"), qpos):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(model, data)
    cid = model.camera("wrist_camera").id
    return data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()


def camera_pose_at_qpos(model_path, qpos_mujoco) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation of the wrist camera at simulator joint angles (forward kinematics only)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    for name, value in zip(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"), qpos_mujoco):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(model, data)
    cid = model.camera("wrist_camera").id
    return data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()


def bench_to_observation(lens, cam_pos, cam_mat, point_xyz) -> tuple[float, float] | None:
    """Bench point -> camera frame -> distorted raw pixel -> observation pixel (256 grid); None if behind the camera or beyond the lens model."""
    d = cam_mat.T @ (np.asarray(point_xyz, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64))
    if d[2] >= -1e-9:
        return None
    x, y = d[0] / -d[2], -d[1] / -d[2]           # MuJoCo camera: x right, y up, looks along -z; image y grows downward
    try:
        xd, yd = lens.distort(np.array([x]), np.array([y]))
    except Exception:
        return None
    u_raw = lens.fx * float(xd[0]) + lens.cx
    v_raw = lens.fy * float(yd[0]) + lens.cy
    u_obs = (u_raw + 0.5) * lens.observation_size / lens.image_size[0] - 0.5
    v_obs = (v_raw + 0.5) * lens.observation_size / lens.image_size[1] - 0.5
    return float(u_obs), float(v_obs)


def cube_visible_at(model_path, bench, qpos_mujoco, cube_center_xyz, *, yaw_rad: float = 0.0, margin_px: float = 20.0) -> dict[str, Any]:
    """Are all four top-face corners of the cube inside the observation (with a margin) from the camera at ``qpos_mujoco``?

    ``margin_px`` is in raw-frame pixels along the shorter (vertical) axis; it is converted to the 256 grid.
    """
    cam_pos, cam_mat = camera_pose_at_qpos(model_path, qpos_mujoco)
    lens = bench.lens_model
    half = bench.cube_edge_m / 2
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    cx, cy, cz = (float(v) for v in cube_center_xyz)
    top = cz + half
    corners = [(cx + c * dx - s * dy, cy + s * dx + c * dy, top) for dx in (-half, half) for dy in (-half, half)]
    pixels = [bench_to_observation(lens, cam_pos, cam_mat, p) for p in corners]
    margin = float(margin_px) * lens.observation_size / lens.image_size[1]
    limit = lens.observation_size - 1
    inside = all(p is not None and margin <= p[0] <= limit - margin and margin <= p[1] <= limit - margin for p in pixels)
    return dict(visible=bool(inside), corners_obs_px=[None if p is None else [round(p[0], 1), round(p[1], 1)] for p in pixels],
                margin_obs_px=round(margin, 2))


def observation_to_bench(lens, cam_pos, cam_mat, u_obs: float, v_obs: float, z_plane: float) -> np.ndarray:
    """Observation pixel (256-grid, full-frame squash) -> raw pixel -> undistorted ray -> bench plane point."""
    u_raw = (u_obs + 0.5) * lens.image_size[0] / lens.observation_size - 0.5
    v_raw = (v_obs + 0.5) * lens.image_size[1] / lens.observation_size - 0.5
    xd, yd = lens.raw_to_normalised(np.array([u_raw]), np.array([v_raw]))
    x, y = lens.undistort(xd, yd)
    ray = cam_mat @ np.array([x[0], -y[0], -1.0])   # MuJoCo camera: x right, y up, looks along -z
    t = (z_plane - cam_pos[2]) / ray[2]
    return cam_pos + t * ray


def locate_cube(observation: np.ndarray, lens, cam_pos, cam_mat, *, cube_top_z: float = CUBE_TOP_Z_M) -> dict[str, Any]:
    """Towel = the brightest connected band; cube = dark pixels enclosed by towel on the same row."""
    g = np.asarray(observation).astype(np.float64).mean(-1)
    p99 = float(np.percentile(g, 99))
    towel_threshold = max(90.0, 0.7 * p99)
    dark_threshold = min(70.0, 0.45 * p99)
    towel = g > towel_threshold
    towel[:, :40] = False
    towel[:, 216:] = False
    ys, xs = np.nonzero(towel)
    if len(ys) < 200:
        return dict(found=False, reason="no towel-sized bright region", p99=round(p99, 1))
    ty0, ty1 = int(ys.min()), int(ys.max())
    dark = np.zeros_like(towel)
    for r in range(ty0, ty1 + 1):
        cols = np.nonzero(towel[r])[0]
        if len(cols) < 5:
            continue
        lo, hi = cols.min(), cols.max()
        dark[r, lo:hi + 1] = g[r, lo:hi + 1] < dark_threshold
    cy, cx = np.nonzero(dark)
    if len(cy) < 40:
        return dict(found=False, reason="no cube-sized dark region on the towel", towel_rows=[ty0, ty1], p99=round(p99, 1))
    u, v = (cx.min() + cx.max()) / 2, (cy.min() + cy.max()) / 2
    centre = observation_to_bench(lens, cam_pos, cam_mat, u, v, cube_top_z)
    clipped = bool(cy.min() == 0 or cy.max() == g.shape[0] - 1 or cx.min() == 0 or cx.max() == g.shape[1] - 1)
    return dict(found=True, cube_xy_mm=[round(float(centre[0]) * 1000, 1), round(float(centre[1]) * 1000, 1)],
                cube_pixels=[int(cx.min()), int(cy.min()), int(cx.max()), int(cy.max())], towel_rows=[ty0, ty1],
                clipped_at_frame_edge=clipped, thresholds=dict(towel=round(towel_threshold, 1), dark=round(dark_threshold, 1)))


def check_cube_placement(observation: np.ndarray, bench, model_path, current_act, *, tolerance_mm: float = DEFAULT_TOLERANCE_MM) -> dict[str, Any]:
    """Locate the cube from a reset-pose observation and compare with the task pose; ``ok`` only inside ``tolerance_mm``."""
    cam_pos, cam_mat = camera_pose_at(model_path, bench, current_act)
    result = locate_cube(observation, bench.lens_model, cam_pos, cam_mat)
    nominal = np.asarray(bench.cube_center[:2], dtype=np.float64) * 1000.0
    result.update(nominal_xy_mm=[round(float(v), 1) for v in nominal], tolerance_mm=float(tolerance_mm))
    regime = getattr(bench, "placement_regime", None)
    if regime is not None:
        # Placement randomization: the task pose is the whole rectangle; a cube outside the reset-pose view is expected.
        result["mode"] = "rectangle"
        result["rectangle_m"] = dict(x=list(regime.x_range_m), y=list(regime.y_range_m))
        if not result.get("found"):
            result.update(ok=True, inside_rectangle=None, advice="cube not visible from the reset pose (expected under placement randomization; the survey frame at step 90 reports it)")
            return result
        xy_m = np.asarray(result["cube_xy_mm"], dtype=np.float64) / 1000.0
        inside = bool(regime.contains(xy_m))
        result.update(inside_rectangle=inside, distance_to_edge_mm=round(regime.distance_to_edge_mm(xy_m), 1), ok=inside)
        result["advice"] = ("cube inside the placement rectangle" if inside else
                            f"cube {abs(result['distance_to_edge_mm']):.0f} mm outside the placement rectangle: move it inside the taped 14 x 10 in area")
        return result
    if not result.get("found"):
        result["ok"] = False
        result["advice"] = "the towel and cube must be visible at the reset pose"
        return result
    dx, dy = np.asarray(result["cube_xy_mm"]) - nominal
    distance = float(np.hypot(dx, dy))
    result.update(offset_mm=dict(dx=round(float(dx), 1), dy=round(float(dy), 1)), distance_mm=round(distance, 1),
                  ok=bool(distance <= tolerance_mm and not result["clipped_at_frame_edge"]))
    if not result["ok"]:
        toward = "toward the base" if dy > 0 else "away from the base"
        side = "to the left (−x)" if dx > 0 else "to the right (+x)"
        clip = " (the cube touches the frame edge, so it is at least this far)" if result["clipped_at_frame_edge"] else ""
        result["advice"] = f"move the cube {abs(dy):.0f} mm {toward} and {abs(dx):.0f} mm {side}{clip}"
    return result


__all__ = ["CUBE_TOP_Z_M", "DEFAULT_TOLERANCE_MM", "camera_pose_at", "check_cube_placement", "locate_cube", "observation_to_bench"]
