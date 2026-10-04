"""One bench environment stepped a chunk of commands at a time (retry tranche, 2026-10-04).

The per-step sequence is the evaluator's (simulation/rollout.py ``_run_pick_place_rollout``): apply the
command through the shared safety gate, advance one 30 Hz control period, measure, evaluate the contract.
Differences, all for speed and none affecting the physics:
- the wrist frame is rendered only at chunk boundaries (the policy reads nothing in between);
- no telemetry file: the episode summary (events, counters, closure in mm at step 250) is returned instead;
- an invalidated episode stops at once (the contract can no longer report success).

Imports neither torch nor any learning code: env workers stay small (host RAM is 15 GB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts import PickPlaceEvaluationState, evaluate_pick_place_step, load_pick_place_contract
from so_arm101_v2.rl.reward import RewardConfig, chunk_reward, potential
from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
from so_arm101_v2.simulation.suites import SimulationScenario

PRIVILEGED_DIM = 36


@dataclass
class EpisodeSummary:
    scenario_id: str
    actions: int = 0
    success: bool = False
    invalidated: bool = False
    timed_out: bool = False
    pickup_completed: bool = False
    strict_grasp_acquired: bool = False
    entered_placement_region: bool = False
    released: bool = False
    max_lift_m: float = 0.0
    safety_frames: int = 0
    return_total: float = 0.0
    reward_parts: dict[str, float] = field(default_factory=lambda: dict(success=0.0, safety=0.0, shaping=0.0))
    closure: dict[str, float] | None = None
    first_violation: dict[str, Any] | None = None   # which safety rule ended the episode, where, on which joint

    @property
    def failure_category(self) -> str:
        if self.success:
            return "success"
        if self.invalidated:
            return "safety_invalidation"
        if not self.pickup_completed:
            return "pickup_incomplete"
        if not self.entered_placement_region:
            return "place_region_missed"
        if not self.released:
            return "not_released"
        return "not_settled_or_retreated"


class ChunkEnv:
    def __init__(self, model_path: str | Path, reward: RewardConfig | None = None, *, closure: bool = True) -> None:
        self.model_path = Path(model_path)
        self.adapter = MujocoTaskAdapter(self.model_path)
        self.contract = load_pick_place_contract("bench_pick_replace_v1", bench_config=self.adapter.bench)
        self.reward = reward or RewardConfig()
        self._closure_kin = None
        if closure:
            from so_arm101_v2.simulation.closure import ClosureKinematics
            self._closure_kin = ClosureKinematics(self.model_path)
        self.scenario: SimulationScenario | None = None
        self.state: PickPlaceEvaluationState | None = None
        self.summary: EpisodeSummary | None = None
        self.done = True
        self._qpos_rows: list[list[float]] = []
        self._cube0: list[float] | None = None
        self._phi = 0.0
        self._place_distance = 1.0
        self._last_contact = np.zeros(3, np.float32)

    def close(self) -> None:
        self.adapter.close()

    # ------------------------------------------------------------------ observation
    def _place_distance_m(self) -> float:
        data = self.adapter.data
        napkin = data.geom("napkin")
        cube = np.asarray(data.body("red_block").xpos, dtype=np.float64)
        horizontal = float(np.linalg.norm(cube[:2] - np.asarray(napkin.xpos[:2], dtype=np.float64)))
        half = float(self.adapter.model.geom("red_block_geom").size[0])
        top = float(napkin.xpos[2] + self.adapter.model.geom("napkin").size[2])
        lift = max(float(cube[2]) - half - top, 0.0)
        return float(np.hypot(horizontal, lift))

    def privileged(self) -> np.ndarray:
        """Critic input (36): joints, joint velocities, cube pose and velocities, contact flags, square, progress, milestones."""
        snap = self.adapter.privileged_state()
        napkin = self.adapter.data.geom("napkin")
        scenario = self.scenario
        yaw = 0.0 if scenario is None else float(scenario.square_yaw_rad)
        st = self.state
        progress = (0 if st is None else st.actions_evaluated) / float(self.contract.max_actions)
        vector = np.concatenate([
            snap.current_act, snap.robot_qvel * 0.1, snap.cube_position * 10.0, snap.cube_quaternion_wxyz,
            snap.cube_linear_velocity, snap.cube_angular_velocity * 0.1, self._last_contact,
            np.asarray(napkin.xpos[:2], np.float32) * 10.0, [np.sin(4 * yaw), np.cos(4 * yaw)], [progress],
            [float(st is not None and st.pickup_state.strict_grasp_acquired), float(st is not None and st.pickup_completed)],
            [min(self._place_distance, 0.3) * 10.0],
        ]).astype(np.float32)
        if vector.shape[0] != PRIVILEGED_DIM:
            raise RuntimeError(f"privileged vector has {vector.shape[0]} dims, expected {PRIVILEGED_DIM}")
        return vector

    def observe(self, *, frame: bool) -> dict[str, Any]:
        raw, _, current = self.adapter.observation(render_pixels=frame)
        return dict(frame=raw, current_act=np.asarray(current, np.float32), action_index=0 if self.state is None else self.state.actions_evaluated,
                    privileged=self.privileged(), done=self.done)

    # ------------------------------------------------------------------ episode
    def reset(self, scenario: SimulationScenario, *, frame: bool = True) -> dict[str, Any]:
        self.adapter.reset(scenario)
        self.scenario = scenario
        self.state = PickPlaceEvaluationState()
        self.summary = EpisodeSummary(scenario_id=scenario.scenario_id)
        self.done = False
        self._qpos_rows = []
        self._cube0 = list(self.adapter.data.body("red_block").xpos)
        self._place_distance = self._place_distance_m()
        self._last_contact = self.adapter.privileged_contact_state().as_array()
        self._phi = potential(self.reward, strict_grasp_acquired=False, pickup_completed=False, place_distance_m=self._place_distance)
        return self.observe(frame=frame)

    def step_chunk(self, commands: np.ndarray, *, frame: bool = True) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        """Run up to len(commands) control steps; returns (next observation, reward parts, done, info)."""
        if self.done or self.state is None or self.summary is None:
            raise RuntimeError("step_chunk on a finished episode: reset first")
        commands = np.asarray(commands, dtype=np.float32)
        phi_start = self._phi
        summary = self.summary
        evaluation = None
        steps = 0
        for requested in commands:
            command = self.adapter.apply_policy_command(requested)
            self.adapter.advance_control_period()
            measurement, diagnostics = self.adapter.pick_place_measurement(command, footprint_edge_margin_m=self.contract.placement.footprint_edge_margin_m)
            self.state, evaluation = evaluate_pick_place_step(self.contract, measurement, self.state)
            steps += 1
            pickup = measurement.pickup
            violated = bool(pickup.command_bound_violation or pickup.delta_limiter_activated or pickup.nonfinite_command or pickup.unsafe_contact)
            summary.safety_frames += int(violated)
            if violated and summary.first_violation is None:
                summary.first_violation = dict(
                    action=int(self.state.actions_evaluated), chunk_step=steps - 1,
                    kinds=[k for k, v in (("command_bound", pickup.command_bound_violation), ("relative_limit", pickup.delta_limiter_activated),
                                          ("nonfinite", pickup.nonfinite_command), ("unsafe_contact", pickup.unsafe_contact)) if v],
                    act_clip_joints=np.flatnonzero(command.act_clip_mask).tolist(),
                    mujoco_clip_joints=np.flatnonzero(command.mujoco_clip_mask).tolist(),
                    relative_limit_joints=np.flatnonzero(command.relative_limit_mask).tolist(),
                    requested_act=np.round(np.asarray(command.requested_act, np.float64), 4).tolist())
            summary.max_lift_m = max(summary.max_lift_m, float(pickup.cube_height_gain_m))
            self._qpos_rows.append(self.adapter.mujoco_qpos().tolist())
            self._last_contact = np.asarray((pickup.any_contact, pickup.bilateral_interior_contact, pickup.strict_bilateral_grasp), np.float32)
            for event in evaluation.events:
                if event.value == "entered_placement_region":
                    summary.entered_placement_region = True
                elif event.value == "released" and self.state.pickup_completed:
                    summary.released = True
            if evaluation.terminated or evaluation.truncated or evaluation.invalidated:
                break
        assert evaluation is not None
        self._place_distance = self._place_distance_m()
        self._phi = potential(self.reward, strict_grasp_acquired=self.state.pickup_state.strict_grasp_acquired,
                              pickup_completed=self.state.pickup_completed, place_distance_m=self._place_distance)
        done = bool(evaluation.terminated or evaluation.truncated or evaluation.invalidated)
        reward = chunk_reward(self.reward, phi_start=phi_start, phi_end=self._phi, success=bool(evaluation.success),
                              invalidated=bool(evaluation.invalidated and not evaluation.success))
        summary.actions = self.state.actions_evaluated
        summary.return_total += reward["total"]
        for key in ("success", "safety", "shaping"):
            summary.reward_parts[key] += reward[key]
        summary.pickup_completed = bool(self.state.pickup_completed)
        summary.strict_grasp_acquired = bool(self.state.pickup_state.strict_grasp_acquired)
        if done:
            summary.success = bool(evaluation.success)
            summary.invalidated = bool(evaluation.invalidated) and not summary.success
            summary.timed_out = not summary.success and not summary.invalidated
            summary.closure = self._closure()
            self.done = True
        observation = self.observe(frame=frame and not done)
        return observation, reward, done, dict(steps=steps, summary=summary if done else None)

    def _closure(self) -> dict[str, float] | None:
        from so_arm101_v2.simulation.closure import CLOSURE_STEP, closure_row
        if self._closure_kin is None or len(self._qpos_rows) < CLOSURE_STEP or self.scenario is None:
            return None
        rows = [dict(robot_qpos=q, cube_position=self._cube0) for q in self._qpos_rows]
        telemetry = dict(rows=rows, placement=dict(square_yaw_rad=float(self.scenario.square_yaw_rad)))
        row = closure_row(self._closure_kin, telemetry)
        return dict(lat=float(row["lat"]), dz=float(row["dz"]), yaw_err=float(row["yaw_err"]))


__all__ = ["ChunkEnv", "EpisodeSummary", "PRIVILEGED_DIM"]
