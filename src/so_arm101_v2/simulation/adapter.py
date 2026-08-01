"""Standalone reward-independent MuJoCo adapter for the v2 task contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import (
    JOINT_NAMES,
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


class MujocoTaskAdapter:
    """Own one MuJoCo model/data pair without importing the legacy environment."""

    def __init__(self, model_path: str | Path, *, image_size: int = 256) -> None:
        if image_size != 256:
            raise ValueError("v2 wrist observations are fixed at 256x256")
        mujoco = _mujoco()
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        self._joint_qpos = []
        self._actuator_ids = []
        for name in JOINT_NAMES:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint < 0 or actuator < 0:
                raise ValueError(f"MuJoCo model is missing named joint/actuator {name!r}")
            self._joint_qpos.append(int(self.model.jnt_qposadr[joint]))
            self._actuator_ids.append(actuator)
        block_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        if block_body < 0:
            raise ValueError("MuJoCo model is missing red_block")
        block_joint = int(self.model.body_jntadr[block_body])
        self._block_qpos = int(self.model.jnt_qposadr[block_joint])
        self._block_body = block_body
        self._fixed_body = int(self.model.body("gripper").id)
        self._moving_body = int(self.model.body("moving_jaw_so101_v1").id)
        self.initial_cube_height = 0.0
        self.control_origin_time = 0.0
        self.control_actions = 0

    def close(self) -> None:
        self.renderer.close()

    def reset(self, scenario: SimulationScenario) -> None:
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

    def mujoco_qpos(self) -> np.ndarray:
        return np.asarray([self.data.qpos[address] for address in self._joint_qpos], dtype=np.float32)

    def current_act(self) -> np.ndarray:
        return mujoco_qpos_to_act(self.mujoco_qpos())

    def render(self, camera: str = "wrist_camera") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        image = np.asarray(self.renderer.render(), dtype=np.uint8)
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"MuJoCo rendered unexpected image shape {image.shape}")
        return image

    def observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = self.render("wrist_camera")
        return raw, preprocess_wrist_image(raw), self.current_act()

    def apply_policy_command(self, requested_act: Any) -> CommandApplication:
        current = self.current_act()
        requested = np.asarray(requested_act, dtype=np.float32)
        nonfinite = requested.shape != (6,) or not np.all(np.isfinite(requested))
        if nonfinite:
            requested = current.copy()
        evaluation = evaluate_physical_command(current, requested)
        executed = physical_normalized_to_act(evaluation.relative_limited_physical)
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
            command_bound_violation=command.command_bound_violation,
            delta_limiter_activated=command.delta_limiter_activated,
            nonfinite_command=command.nonfinite_command,
        )
        return measurement, {"grip_force_n": float(force), **diagnostics}


__all__ = ["CommandApplication", "MujocoTaskAdapter"]
