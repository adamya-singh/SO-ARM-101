"""Simulator backend for the physical runner: drives a MujocoTaskAdapter and cross-checks the gate every step."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from so_arm101_v2.contracts.pick_place import PickPlaceEvaluationState, evaluate_pick_place_step
from so_arm101_v2.contracts.physical import HoldDecision

from .runner import Observation, PeriodOutcome, SendRecord


class ParityError(RuntimeError):
    """The runner's gate decision differed from the adapter's own application of the same request."""


class SimBackend:
    name = "simulation"

    def __init__(self, adapter: Any, scenario: Any, *, contract: Any,
                 frame_source: Callable[[int], np.ndarray] | None = None) -> None:
        self.adapter = adapter
        self.scenario = scenario
        self.contract = contract
        self.frame_source = frame_source
        self.state = PickPlaceEvaluationState()
        self.rows: list[dict[str, Any]] = []
        self.counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
        self._command = None
        self._current = None
        adapter.reset(scenario)

    def read_act(self) -> np.ndarray:
        self._current = np.asarray(self.adapter.current_act(), dtype=np.float32)
        return self._current

    def observe(self, step: int) -> Observation:
        started = time.perf_counter()
        image = self.frame_source(step) if self.frame_source is not None else self.adapter.render_wrist_observation()
        image = np.asarray(image, dtype=np.uint8)
        return Observation(image=image, frame_seq=step, frame_time=float(self.adapter.data.time), age_s=0.0,
                           raw_rgb=None, observe_ms=(time.perf_counter() - started) * 1e3)

    def send(self, step: int, decision: HoldDecision) -> SendRecord:
        started = time.perf_counter()
        # One gate application, exactly like evaluate_closed_loop: the adapter sees the raw policy output.
        command = self.adapter.apply_policy_command(decision.policy_act)
        if (not np.array_equal(command.executed_act, decision.executed_act)
                or not np.array_equal(command.requested_act, decision.requested_act)
                or command.nonfinite_command != decision.nonfinite
                or (command.command_bound_violation or command.delta_limiter_activated) != decision.held):
            raise ParityError(f"step {step}: runner decision differs from the adapter's application")
        self._command = command
        return SendRecord(sent_physical=decision.sent_physical, returned_physical=None,
                          send_ms=(time.perf_counter() - started) * 1e3)

    def wait_next_period(self, step: int) -> PeriodOutcome:
        substeps = self.adapter.advance_control_period()
        measurement, _ = self.adapter.pick_place_measurement(
            self._command, footprint_edge_margin_m=self.contract.placement.footprint_edge_margin_m)
        self.state, evaluation = evaluate_pick_place_step(self.contract, measurement, self.state)
        self.counts["clip"] += int(measurement.pickup.command_bound_violation)
        self.counts["limit"] += int(measurement.pickup.delta_limiter_activated)
        self.counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
        self.counts["unsafe"] += int(measurement.pickup.unsafe_contact)
        self.rows.append({
            "action": step + 1,
            "simulation_time_s": float(self.adapter.data.time),
            "physics_substeps": substeps,
            "current_act": self._current.tolist(),
            "requested_act": self._command.requested_act.tolist(),
            "executed_act": self._command.executed_act.tolist(),
            "robot_qpos": self.adapter.mujoco_qpos().tolist(),
            "cube_position": self.adapter.data.body("red_block").xpos.tolist(),
        })
        done = bool(evaluation.terminated or evaluation.truncated)
        return PeriodOutcome(done=done, success=(bool(evaluation.success) if done else None),
                             invalidated=bool(evaluation.invalidated), extra={"substeps": substeps})

    def close(self) -> None:
        self.adapter.close()


__all__ = ["ParityError", "SimBackend"]
