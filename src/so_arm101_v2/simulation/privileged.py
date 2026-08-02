"""Small deterministic staged controller used only to prove the simulator suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from so_arm101_v2.contracts import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    MUJOCO_JOINT_HIGH,
    MUJOCO_JOINT_LOW,
    mujoco_qpos_to_act,
)


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("privileged controller requires the 'sim' extra") from exc
    return mujoco


def _minimum_jerk(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


# Grasp pocket of the Menagerie gripper in the "gripper" body frame: the
# fixed_jaw_pad_4 / moving_jaw_pad_4 pair is exactly parallel with a 25.2 mm
# gap at gripper qpos ~0.0, centered at this point (fixed pad face at local
# x = +0.0133 minus half the gap). The 25 mm cube is gripped there.
POCKET_GRIPPER_FRAME = np.array([0.0017, -0.0572, 0.0], dtype=np.float64)
# Mouth center at the jaw-tip station. The pad staircase makes the mouth a
# funnel: ~36 mm wide here, narrowing to 25.2 mm at the pad-4 pocket, so the
# cube can enter between the tips and be funneled onto the pocket.
TIP_STATION_GRIPPER_FRAME = np.array([0.0017, -0.1014, 0.0], dtype=np.float64)
# Pad inner normal = -x of the gripper body (the mouth opens toward -x);
# jaw depth direction (palm -> pads -> tips) = -y of the gripper body.
PAD_NORMAL_LOCAL = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
DEPTH_LOCAL = np.array([0.0, -1.0, 0.0], dtype=np.float64)


@dataclass
class PrivilegedStagedController:
    """Pocket-frame IK followed by slow joint-space waypoints.

    The Menagerie jaws grip a 25 mm cube at the pad-4 station, which sits only
    ~57 mm from the palm while the jaw tips extend ~44 mm further, so a floor
    cube must be approached with the jaws pitched well forward: advance along
    the jaw depth axis at cube height with the moving jaw swung open, slide
    the pocket onto the cube, close, and lift. The gripper servo is commanded
    fully closed so squeeze force comes from the position controller.
    """

    damping: float = 1e-3
    maximum_update_rad: float = 0.05
    position_tolerance_m: float = 0.001
    normal_tolerance_deg: float = 10.0
    depth_tolerance_deg: float = 15.0
    axis_weight: float = 0.04
    depth_weight: float = 0.04
    approach_pitch_rad: float = np.deg2rad(78.0)
    approach_clearance_m: float = 0.005
    approach_back_m: float = 0.06
    approach_lift_m: float = 0.04
    open_gripper_rad: float = 0.8
    nearly_closed_gripper_rad: float = 0.0
    grasp_height_m: float = 0.021
    depth_lead_m: float = 0.006
    lift_command_m: float = 0.031

    def __post_init__(self) -> None:
        self.action_index = 0
        self.waypoints: list[np.ndarray] = []
        self.boundaries: tuple[int, ...] = ()
        self.solve_diagnostics: list[dict[str, float | bool]] = []

    def _targets(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame targets for the pad normal and jaw depth directions."""
        pitch = self.approach_pitch_rad
        depth_target = np.array([0.0, np.sin(pitch), -np.cos(pitch)])
        normal_target = np.array([1.0, 0.0, 0.0])
        return normal_target, depth_target

    @staticmethod
    def _pocket_frame(adapter: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        body = adapter.data.body("gripper")
        rotation = np.asarray(body.xmat).reshape(3, 3)
        pocket = body.xpos + rotation @ POCKET_GRIPPER_FRAME
        return pocket, rotation @ PAD_NORMAL_LOCAL, rotation @ DEPTH_LOCAL

    def _set_arm(self, adapter: Any, qpos: np.ndarray) -> None:
        mujoco = _mujoco()
        for address, value in zip(adapter._joint_qpos, qpos, strict=True):
            adapter.data.qpos[address] = value
        mujoco.mj_forward(adapter.model, adapter.data)

    def _solve(self, adapter: Any, seed: np.ndarray, pocket_target: np.ndarray, gripper: float, station: np.ndarray | None = None, loose: bool = False) -> np.ndarray:
        mujoco = _mujoco()
        original_qpos = adapter.data.qpos.copy()
        original_qvel = adapter.data.qvel.copy()
        qpos = seed.astype(np.float64).copy()
        qpos[5] = gripper
        # Carrying/placing solves do not need grasp-grade orientation: allow
        # the wrist to yaw and pitch with the reach, and soften the orientation
        # residual weights so they cannot stall position convergence.
        normal_tolerance = 60.0 if loose else self.normal_tolerance_deg
        depth_tolerance = 45.0 if loose else self.depth_tolerance_deg
        axis_weight = self.axis_weight * 0.2 if loose else self.axis_weight
        depth_weight = self.depth_weight * 0.2 if loose else self.depth_weight
        station_local = POCKET_GRIPPER_FRAME if station is None else station
        station_offset = station_local - POCKET_GRIPPER_FRAME
        normal_target, depth_target = self._targets()
        position_error = 1.0
        normal_error = 180.0
        depth_error = 180.0
        try:
            for _ in range(250):
                self._set_arm(adapter, qpos)
                pocket, normal, depth = self._pocket_frame(adapter)
                rotation = np.asarray(adapter.data.body("gripper").xmat).reshape(3, 3)
                pocket = pocket + rotation @ station_offset
                position_residual = pocket_target - pocket
                position_error = float(np.linalg.norm(position_residual))
                normal_error = float(np.degrees(np.arccos(np.clip(abs(np.dot(normal, normal_target)), -1, 1))))
                depth_error = float(np.degrees(np.arccos(np.clip(np.dot(depth, depth_target), -1, 1))))
                if (
                    position_error <= self.position_tolerance_m
                    and normal_error <= normal_tolerance
                    and depth_error <= depth_tolerance
                ):
                    break
                signed_normal_target = normal_target if np.dot(normal, normal_target) >= 0 else -normal_target
                residual = np.concatenate((
                    position_residual,
                    axis_weight * (signed_normal_target - normal),
                    depth_weight * (depth_target - depth),
                ))
                jacobian = np.empty((9, 5), dtype=np.float64)
                epsilon = 1e-4
                for joint in range(5):
                    adapter.data.qpos[adapter._joint_qpos[joint]] += epsilon
                    mujoco.mj_forward(adapter.model, adapter.data)
                    pocket2, normal2, depth2 = self._pocket_frame(adapter)
                    rotation2 = np.asarray(adapter.data.body("gripper").xmat).reshape(3, 3)
                    pocket2 = pocket2 + rotation2 @ station_offset
                    jacobian[:3, joint] = (pocket2 - pocket) / epsilon
                    jacobian[3:6, joint] = axis_weight * (normal2 - normal) / epsilon
                    jacobian[6:, joint] = depth_weight * (depth2 - depth) / epsilon
                    adapter.data.qpos[adapter._joint_qpos[joint]] = qpos[joint]
                lhs = jacobian @ jacobian.T + self.damping * np.eye(9)
                update = jacobian.T @ np.linalg.solve(lhs, residual)
                update = np.clip(update, -self.maximum_update_rad, self.maximum_update_rad)
                qpos[:5] = np.clip(qpos[:5] + update, MUJOCO_JOINT_LOW[:5], MUJOCO_JOINT_HIGH[:5])
            solved = bool(
                position_error <= self.position_tolerance_m
                and normal_error <= normal_tolerance
                and depth_error <= depth_tolerance
            )
            self.solve_diagnostics.append({
                "solved": solved, "position_error_m": position_error,
                "normal_error_deg": normal_error, "depth_error_deg": depth_error,
            })
            if not solved:
                raise RuntimeError(
                    f"privileged IK failed: position={position_error:.6f}m "
                    f"normal={normal_error:.3f}deg depth={depth_error:.3f}deg"
                )
            return qpos.astype(np.float32)
        finally:
            adapter.data.qpos[:] = original_qpos
            adapter.data.qvel[:] = original_qvel
            mujoco.mj_forward(adapter.model, adapter.data)

    def reset(self, adapter: Any) -> None:
        self.action_index = 0
        self.solve_diagnostics = []
        start = np.clip(adapter.mujoco_qpos(), MUJOCO_JOINT_LOW, MUJOCO_JOINT_HIGH)
        cube = adapter.data.body("red_block").xpos.copy()
        # Keep a margin inside the mechanical limit: the calibrated act range
        # overhangs the Menagerie ctrlrange at the closed end, and commanding
        # the exact bound flags a command_bound_violation every step.
        closed_gripper = float(MUJOCO_JOINT_LOW[5]) + 0.004
        open_gripper = self.open_gripper_rad
        pocket_grasp = np.array([cube[0], cube[1] - self.depth_lead_m, self.grasp_height_m], dtype=np.float64)
        seed = start.astype(np.float64)

        def with_gripper(qpos: np.ndarray, gripper: float) -> np.ndarray:
            result = qpos.copy()
            result[5] = np.float32(gripper)
            return result

        # Vertical entry: with the bill horizontal, the mouth's open bottom
        # descends straight down around the cube at the pad-4 station (the
        # mid-air wedge test proves the mouth hosts the full cube there).
        # The fixed pad face has only ~0.1 mm clearance from a centered cube,
        # so the descent runs shifted toward the (open) moving-jaw side and a
        # final horizontal slide recenters the pocket before closing.
        probe = self._solve(
            adapter, seed, pocket_grasp + np.array([0.0, 0.0, 0.06]), open_gripper
        )
        self._set_arm(adapter, probe)
        _, mouth_normal, _ = self._pocket_frame(adapter)
        # Shift the pocket TARGET against the pad normal: the cube (fixed in
        # the world) then sits displaced toward the open moving-jaw side of
        # the mouth, giving the fixed pad clearance during the descent.
        mouth_shift = -self.approach_clearance_m * mouth_normal
        self._set_arm(adapter, start.astype(np.float64))
        above = self._solve(
            adapter, seed, pocket_grasp + mouth_shift + np.array([0.0, 0.0, 0.06]), open_gripper
        )
        lowered = self._solve(
            adapter, above, pocket_grasp + mouth_shift + np.array([0.0, 0.0, 0.025]), open_gripper
        )
        descended = self._solve(
            adapter, lowered, pocket_grasp + mouth_shift + np.array([0.0, 0.0, 0.008]), open_gripper
        )
        engaged = self._solve(adapter, descended, pocket_grasp + mouth_shift, open_gripper)
        seated = self._solve(adapter, engaged, pocket_grasp, open_gripper)
        half_closed = with_gripper(seated, self.nearly_closed_gripper_rad)
        closed = with_gripper(seated, closed_gripper)
        lift = self._solve(
            adapter, closed, pocket_grasp + np.array([0.0, 0.0, self.lift_command_m]),
            closed_gripper,
        )
        self.waypoints = [
            start, above, lowered, descended, engaged, seated,
            half_closed, closed, closed, lift, lift,
        ]
        self.boundaries = (0, 70, 110, 145, 175, 205, 255, 285, 315, 355, 450)

        # Place phase, matching the physical dataset: carry the cube over the
        # napkin, set it down, release, and retreat. The task contract still
        # terminates on pickup success, so contract-evaluated episodes (and
        # the preflight) end before these stages; the live viewer plays them.
        mujoco = _mujoco()
        napkin_geom = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "napkin")
        if napkin_geom >= 0:
            napkin = adapter.data.geom_xpos[napkin_geom].copy()
            lift_z = self.grasp_height_m + self.lift_command_m
            # While held, the cube center rides ~8.5 mm below the pocket, so a
            # pocket 1.5 mm above grasp height sets the cube gently onto the
            # 1 mm napkin before release.
            pocket_over = np.array([napkin[0], napkin[1] - self.depth_lead_m, lift_z])
            pocket_down = np.array([napkin[0], napkin[1] - self.depth_lead_m, self.grasp_height_m + 0.0015])
            traverse = self._solve(adapter, lift, pocket_over, closed_gripper, loose=True)
            set_down = self._solve(adapter, traverse, pocket_down, closed_gripper, loose=True)
            released = with_gripper(set_down, open_gripper)
            retreat = self._solve(
                adapter, released, pocket_down + np.array([0.0, 0.0, 0.06]), open_gripper, loose=True
            )
            self.waypoints = [
                start, above, lowered, descended, engaged, seated,
                half_closed, closed, closed, lift, lift,
                traverse, set_down, released, released, retreat,
            ]
            # The retreat gets ~26 actions: a 5-action retreat produced a
            # ~670 mm/s snap that the phase-state oracle clone could not
            # imitate (worst-error row of the failed offline gate). The
            # gripper release keeps >=16 actions: opening faster trips the
            # delta limiter through servo lag (requested-vs-current gap).
            self.boundaries = (
                0, 70, 110, 145, 175, 205, 255, 285, 315, 350, 365,
                386, 403, 419, 424, 450,
            )

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del image, current_act, adapter
        boundaries = self.boundaries
        self.action_index += 1
        for stage in range(len(boundaries) - 1):
            if self.action_index <= boundaries[stage + 1]:
                start = self.waypoints[stage]
                stop = self.waypoints[stage + 1]
                fraction = (self.action_index - boundaries[stage]) / (boundaries[stage + 1] - boundaries[stage])
                result = mujoco_qpos_to_act(
                    (1.0 - _minimum_jerk(fraction)) * start + _minimum_jerk(fraction) * stop
                )
                return np.clip(
                    result, ACT_DATASET_LOW + np.float32(1e-6),
                    ACT_DATASET_HIGH - np.float32(1e-6),
                ).astype(np.float32)
        return np.clip(
            mujoco_qpos_to_act(self.waypoints[-1]),
            ACT_DATASET_LOW + np.float32(1e-6), ACT_DATASET_HIGH - np.float32(1e-6),
        ).astype(np.float32)


__all__ = ["PrivilegedStagedController"]
