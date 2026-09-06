"""Standalone reward-independent MuJoCo adapter for the v2 task contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import (
    JOINT_NAMES,
    PickPlaceMeasurement,
    TaskMeasurement,
    act_to_mujoco_qpos,
    evaluate_physical_command,
    mujoco_qpos_to_act,
    physical_normalized_to_act,
)
from so_arm101_v2.data import preprocess_wrist_image

from .contact import check_block_face_gripped
from .suites import SimulationScenario


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("simulation requires the 'sim' extra") from exc
    return mujoco


@dataclass(frozen=True)
class CommandApplication:
    requested_act: np.ndarray
    executed_act: np.ndarray
    command_bound_violation: bool
    delta_limiter_activated: bool
    nonfinite_command: bool
    act_clip_mask: np.ndarray
    mujoco_clip_mask: np.ndarray
    relative_limit_mask: np.ndarray


@dataclass(frozen=True)
class PrivilegedStateSnapshot:
    """Finite pre-action simulator state used only by diagnostic policies."""

    current_act: np.ndarray
    robot_qvel: np.ndarray
    cube_position: np.ndarray
    cube_quaternion_wxyz: np.ndarray
    cube_linear_velocity: np.ndarray
    cube_angular_velocity: np.ndarray


@dataclass(frozen=True)
class PrivilegedContactSnapshot:
    """Causal pre-action contact state exposed only to diagnostic policies."""

    any_contact: bool
    bilateral_interior_contact: bool
    strict_bilateral_grasp: bool

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.any_contact,
                self.bilateral_interior_contact,
                self.strict_bilateral_grasp,
            ),
            dtype=np.float32,
        )


class MujocoTaskAdapter:
    """Own one MuJoCo model/data pair without importing the legacy environment."""

    def __init__(self, model_path: str | Path, *, image_size: int = 256) -> None:
        if image_size != 256:
            raise ValueError("v2 wrist observations are fixed at 256x256")
        mujoco = _mujoco()
        self.model_path = Path(model_path).resolve()
        from so_arm101_v2.contracts.bench import scene_bench_config
        self.bench = scene_bench_config(self.model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._renderer: Any | None = None
        self._joint_qpos = []
        self._joint_dof = []
        self._actuator_ids = []
        for name in JOINT_NAMES:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint < 0 or actuator < 0:
                raise ValueError(f"MuJoCo model is missing named joint/actuator {name!r}")
            self._joint_qpos.append(int(self.model.jnt_qposadr[joint]))
            self._joint_dof.append(int(self.model.jnt_dofadr[joint]))
            self._actuator_ids.append(actuator)
        block_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        if block_body < 0:
            raise ValueError("MuJoCo model is missing red_block")
        block_joint = int(self.model.body_jntadr[block_body])
        self._block_qpos = int(self.model.jnt_qposadr[block_joint])
        self._block_dof = int(self.model.jnt_dofadr[block_joint])
        self._block_body = block_body
        self._fixed_body = int(self.model.body("gripper").id)
        self._moving_body = int(self.model.body("moving_jaw_so101_v1").id)
        self.initial_cube_height = 0.0
        self.control_origin_time = 0.0
        self.control_actions = 0
        self._previous_privileged_snapshot: PrivilegedStateSnapshot | None = None
        self._previous_contact_snapshot: PrivilegedContactSnapshot | None = None

    @property
    def renderer(self) -> Any:
        # Lazy so pixel-free rollouts never create an offscreen GL context.
        if self._renderer is None:
            self._renderer = _mujoco().Renderer(self.model, height=256, width=256)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self, scenario: SimulationScenario) -> None:
        if self.bench is not None:
            self.bench.validate_qpos(scenario.robot_qpos_mujoco)
        mujoco = _mujoco()
        mujoco.mj_resetData(self.model, self.data)
        for address, value, actuator in zip(
            self._joint_qpos, scenario.robot_qpos_mujoco, self._actuator_ids, strict=True
        ):
            self.data.qpos[address] = value
            self.data.ctrl[actuator] = value
        start = self._block_qpos
        self.data.qpos[start:start + 3] = scenario.cube_position_m
        self.data.qpos[start + 3:start + 7] = scenario.cube_quaternion_wxyz
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        robot_error = np.max(np.abs(self.mujoco_qpos() - np.asarray(scenario.robot_qpos_mujoco)))
        cube_error = np.linalg.norm(self.data.body("red_block").xpos[:2] - np.asarray(scenario.cube_position_m[:2]))
        if robot_error > 0.02 + 1e-9 or cube_error > 0.002 + 1e-9:
            raise RuntimeError(
                f"settled reset exceeds contract tolerance: robot={robot_error}, cube_xy={cube_error}"
            )
        self.initial_cube_height = float(self.data.body("red_block").xpos[2])
        self.control_origin_time = float(self.data.time)
        self.control_actions = 0
        # A two-frame policy repeats the reset observation at action zero.  Each
        # subsequent command application replaces these with the actual
        # preceding pre-action state, including during oracle-controlled prefixes.
        self._previous_privileged_snapshot = self.privileged_state()
        self._previous_contact_snapshot = self.privileged_contact_state()

    def mujoco_qpos(self) -> np.ndarray:
        return np.asarray([self.data.qpos[address] for address in self._joint_qpos], dtype=np.float32)

    def current_act(self) -> np.ndarray:
        return mujoco_qpos_to_act(self.mujoco_qpos())

    def privileged_state(self) -> PrivilegedStateSnapshot:
        quaternion = np.asarray(self.data.qpos[self._block_qpos + 3:self._block_qpos + 7], dtype=np.float32).copy()
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0 or not np.isfinite(norm):
            raise RuntimeError("cube quaternion is invalid")
        quaternion /= np.float32(norm)
        if quaternion[0] < 0:
            quaternion *= np.float32(-1.0)
        snapshot = PrivilegedStateSnapshot(
            current_act=self.current_act(),
            robot_qvel=np.asarray([self.data.qvel[address] for address in self._joint_dof], dtype=np.float32),
            cube_position=np.asarray(self.data.body("red_block").xpos, dtype=np.float32).copy(),
            cube_quaternion_wxyz=quaternion,
            cube_linear_velocity=np.asarray(self.data.qvel[self._block_dof:self._block_dof + 3], dtype=np.float32).copy(),
            cube_angular_velocity=np.asarray(self.data.qvel[self._block_dof + 3:self._block_dof + 6], dtype=np.float32).copy(),
        )
        arrays = tuple(getattr(snapshot, field) for field in snapshot.__dataclass_fields__)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError("privileged state contains nonfinite values")
        return snapshot

    def privileged_contact_state(self) -> PrivilegedContactSnapshot:
        """Measure contact at the current state without applying or advancing a command."""
        strict, _, diagnostics = check_block_face_gripped(self.model, self.data)
        return PrivilegedContactSnapshot(
            any_contact=self._any_cube_robot_contact(),
            bilateral_interior_contact=bool(diagnostics["bilateral_interior_face_contact"]),
            strict_bilateral_grasp=bool(strict),
        )

    def previous_privileged_state(self) -> PrivilegedStateSnapshot:
        if self._previous_privileged_snapshot is None:
            raise RuntimeError("adapter has not been reset")
        return self._previous_privileged_snapshot

    def previous_privileged_contact_state(self) -> PrivilegedContactSnapshot:
        if self._previous_contact_snapshot is None:
            raise RuntimeError("adapter has not been reset")
        return self._previous_contact_snapshot

    def rebase_observation_history(self) -> None:
        """Repeat the current state as history after diagnostic setup mutates derived state."""
        self._previous_privileged_snapshot = self.privileged_state()
        self._previous_contact_snapshot = self.privileged_contact_state()

    def render(self, camera: str = "wrist_camera") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        image = np.asarray(self.renderer.render(), dtype=np.uint8)
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"MuJoCo rendered unexpected image shape {image.shape}")
        return image

    def observation(
        self, *, render_pixels: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        if not render_pixels:
            return None, None, self.current_act()
        raw = self.render("wrist_camera")
        return raw, preprocess_wrist_image(raw), self.current_act()

    def apply_policy_command(self, requested_act: Any) -> CommandApplication:
        # Capture before changing actuator controls.  After the physics period
        # advances, this is exactly the previous pre-action observation.
        self._previous_privileged_snapshot = self.privileged_state()
        self._previous_contact_snapshot = self.privileged_contact_state()
        current = self.current_act()
        requested = np.asarray(requested_act, dtype=np.float32)
        nonfinite = requested.shape != (6,) or not np.all(np.isfinite(requested))
        if nonfinite:
            requested = current.copy()
        evaluation = evaluate_physical_command(current, requested,
            shoulder_floor=self.bench.shoulder_floor if self.bench else None)
        executed = physical_normalized_to_act(evaluation.relative_limited_physical)
        if self.bench and (np.any(evaluation.physical_clip_mask) or np.any(evaluation.mujoco_clip_mask)
                           or np.any(evaluation.act_clip_mask) or np.any(evaluation.relative_limit_mask)):
            executed = current.copy()
        qpos = act_to_mujoco_qpos(executed)
        for actuator, value in zip(self._actuator_ids, qpos, strict=True):
            self.data.ctrl[actuator] = float(value)
        return CommandApplication(
            requested_act=requested,
            executed_act=executed,
            command_bound_violation=bool(
                np.any(evaluation.act_clip_mask)
                or np.any(evaluation.mujoco_clip_mask)
                or np.any(evaluation.physical_clip_mask)
            ),
            delta_limiter_activated=bool(np.any(evaluation.relative_limit_mask)),
            nonfinite_command=bool(nonfinite),
            act_clip_mask=evaluation.act_clip_mask,
            mujoco_clip_mask=evaluation.mujoco_clip_mask,
            relative_limit_mask=evaluation.relative_limit_mask,
        )

    def apply_external_physical_gripper_opening(self, units: float) -> None:
        current = self.current_act()
        evaluation = evaluate_physical_command(current, current)
        physical = evaluation.current_physical.copy()
        physical[5] = np.clip(physical[5] + units, 0.0, 100.0)
        qpos = act_to_mujoco_qpos(physical_normalized_to_act(physical))
        for actuator, value in zip(self._actuator_ids, qpos, strict=True):
            self.data.ctrl[actuator] = float(value)

    def advance_control_period(self) -> int:
        mujoco = _mujoco()
        self.control_actions += 1
        target = self.control_origin_time + self.control_actions / 30.0
        steps = 0
        while self.data.time + 1e-12 < target:
            mujoco.mj_step(self.model, self.data)
            steps += 1
        return steps

    def _any_cube_robot_contact(self) -> bool:
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self._block_body in bodies and (self._fixed_body in bodies or self._moving_body in bodies):
                return True
        return False

    def _unsafe_contact(self, height_gain: float) -> bool:
        if height_gain <= 0.002:
            return False
        allowed = {self._block_body, self._fixed_body, self._moving_body}
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self._block_body in bodies and any(body not in allowed for body in bodies):
                return True
        return False

    def measurement(self, command: CommandApplication | None = None) -> tuple[TaskMeasurement, dict[str, Any]]:
        cube = self.data.body("red_block").xpos.copy()
        fixed = self.data.site("fixed_jaw_tip").xpos.copy()
        moving = self.data.site("moving_jaw_tip").xpos.copy()
        jaw_center = 0.5 * (fixed + moving)
        height_gain = float(cube[2] - self.initial_cube_height)
        strict, force, diagnostics = check_block_face_gripped(self.model, self.data)
        command = command or CommandApplication(
            requested_act=self.current_act(), executed_act=self.current_act(),
            command_bound_violation=False, delta_limiter_activated=False,
            nonfinite_command=False, act_clip_mask=np.zeros(6, bool),
            mujoco_clip_mask=np.zeros(6, bool), relative_limit_mask=np.zeros(6, bool),
        )
        measurement = TaskMeasurement(
            jaw_cube_distance_m=float(np.linalg.norm(jaw_center - cube)),
            any_contact=self._any_cube_robot_contact(),
            bilateral_interior_contact=bool(diagnostics["bilateral_interior_face_contact"]),
            strict_bilateral_grasp=bool(strict), cube_height_gain_m=height_gain,
            unsafe_contact=self._unsafe_contact(height_gain),
            command_bound_violation=command.command_bound_violation or bool(
                self.bench is not None and self.mujoco_qpos()[1] < self.bench.mujoco_low[1]),
            delta_limiter_activated=command.delta_limiter_activated,
            nonfinite_command=command.nonfinite_command,
        )
        return measurement, {"grip_force_n": float(force), **diagnostics}

    def pick_place_measurement(
        self, command: CommandApplication | None = None, *, footprint_edge_margin_m: float = 0.002
    ) -> tuple[PickPlaceMeasurement, dict[str, Any]]:
        """Measure the full task in napkin-local geometry."""
        mujoco = _mujoco()
        pickup, diagnostics = self.measurement(command)
        napkin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "napkin")
        if napkin_id < 0:
            raise ValueError("MuJoCo model is missing napkin geom")
        napkin_position = np.asarray(self.data.geom_xpos[napkin_id], dtype=np.float64)
        napkin_rotation = np.asarray(self.data.geom_xmat[napkin_id], dtype=np.float64).reshape(3, 3)
        napkin_size = np.asarray(self.model.geom_size[napkin_id], dtype=np.float64)
        cube_body = self.data.body("red_block")
        cube_position = np.asarray(cube_body.xpos, dtype=np.float64)
        cube_rotation = np.asarray(cube_body.xmat, dtype=np.float64).reshape(3, 3)
        half = float(self.model.geom("red_block_geom").size[0])
        local_corners = np.asarray(
            [(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)],
            dtype=np.float64,
        )
        world_corners = cube_position + local_corners @ cube_rotation.T
        napkin_corners = (world_corners - napkin_position) @ napkin_rotation
        xy_limit = napkin_size[:2] - float(footprint_edge_margin_m)
        footprint_inside = bool(np.all(np.abs(napkin_corners[:, :2]) <= xy_limit + 1e-12))
        support_error = float(np.min(napkin_corners[:, 2]) - napkin_size[2])
        velocity = np.asarray(self.data.qvel[self._block_dof:self._block_dof + 6], dtype=np.float64)
        measurement = PickPlaceMeasurement(
            pickup=pickup,
            cube_footprint_inside=footprint_inside,
            cube_support_error_m=support_error,
            cube_linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
            cube_angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
            gripper_act=float(self.current_act()[5]),
        )
        diagnostics = {
            **diagnostics,
            "napkin_local_cube_corners": napkin_corners.tolist(),
            "cube_footprint_inside": footprint_inside,
            "cube_support_error_m": support_error,
            "cube_linear_speed_m_s": measurement.cube_linear_speed_m_s,
            "cube_angular_speed_rad_s": measurement.cube_angular_speed_rad_s,
            "gripper_act": measurement.gripper_act,
        }
        return measurement, diagnostics


__all__ = [
    "CommandApplication", "MujocoTaskAdapter", "PrivilegedContactSnapshot",
    "PrivilegedStateSnapshot",
]
