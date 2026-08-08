"""Immutable named MuJoCo evaluation scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    cube_position_m: tuple[float, float, float]
    robot_qpos_mujoco: tuple[float, ...]
    cube_quaternion_wxyz: tuple[float, float, float, float]
    fixed_appearance: bool = True

    def __post_init__(self) -> None:
        if len(self.cube_position_m) != 3 or len(self.robot_qpos_mujoco) != 6 or len(self.cube_quaternion_wxyz) != 4:
            raise ValueError("simulation scenario has malformed reset dimensions")


@dataclass(frozen=True)
class SimulationSuite:
    suite_id: str
    schema_version: int
    task_contract: str
    repeats: int
    diagnostic_only: bool
    scenarios: tuple[SimulationScenario, ...]
    recovery_probe: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.repeats <= 0 or not self.scenarios:
            raise ValueError("invalid simulation suite")
        if len({item.scenario_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("simulation scenario ids must be unique")


def _suite_from_payload(payload: dict) -> SimulationSuite:
    scenarios = tuple(SimulationScenario(
        scenario_id=str(item["scenario_id"]),
        cube_position_m=tuple(float(value) for value in item["cube_position_m"]),
        robot_qpos_mujoco=tuple(float(value) for value in item["robot_qpos_mujoco"]),
        cube_quaternion_wxyz=tuple(float(value) for value in item["cube_quaternion_wxyz"]),
        fixed_appearance=bool(item["fixed_appearance"]),
    ) for item in payload["scenarios"])
    probe = payload.get("recovery_probe")
    return SimulationSuite(
        suite_id=str(payload["suite_id"]), schema_version=int(payload["schema_version"]),
        task_contract=str(payload["task_contract"]), repeats=int(payload["repeats"]),
        diagnostic_only=bool(payload["diagnostic_only"]), scenarios=scenarios,
        recovery_probe=None if probe is None else {str(k): int(v) for k, v in probe.items()},
    )


def suite_payload(suite: SimulationSuite) -> dict:
    """Canonical JSON payload for a suite (generated-suite serialization)."""
    from dataclasses import asdict

    return {
        "suite_id": suite.suite_id,
        "schema_version": suite.schema_version,
        "task_contract": suite.task_contract,
        "repeats": suite.repeats,
        "diagnostic_only": suite.diagnostic_only,
        "scenarios": [asdict(item) for item in suite.scenarios],
        "recovery_probe": suite.recovery_probe,
    }


# Napkin footprint in the v3 scene: the place target; cube spawns must avoid it.
_NAPKIN_X = (0.0546, 0.1054)
_NAPKIN_Y = (0.2746, 0.3254)


def generate_randomized_suite(
    generator_seed: int,
    count: int,
    *,
    x_range: tuple[float, float] = (-0.06, 0.06),
    y_range: tuple[float, float] = (0.24, 0.31),
    repeats: int = 3,
    exclude_napkin: bool = True,
    screen_with_model: "str | None" = None,
) -> SimulationSuite:
    """Deterministically sample a randomized cube-pose suite for the v3 task.

    The robot start pose, cube orientation, and z height come from the
    packaged v3 nominal scenario; only the cube xy varies.  The y range is
    biased inward of the documented workspace edge (y = 0.30).
    """
    import numpy as np

    if count <= 0:
        raise ValueError("count must be positive")
    base = load_simulation_suite("fixed_pick_place_v3")
    nominal = next(item for item in base.scenarios if item.scenario_id == "nominal")
    rng = np.random.default_rng(int(generator_seed))
    screen = None
    if screen_with_model is not None:
        # IK-screen each candidate: the privileged teacher must solve its
        # full plan from the sampled pose, or the pose is rejected.  The rng
        # consumption order is deterministic, so (seed, region, model) fully
        # determine the suite.
        from so_arm101_v2.contracts import (
            PickPlaceEvaluationState,
            evaluate_pick_place_step,
            load_pick_place_contract,
        )

        from .adapter import MujocoTaskAdapter
        from .privileged import PrivilegedStagedController

        contract = load_pick_place_contract("fixed_cube_pick_place_v3")

        def screen(candidate: SimulationScenario) -> bool:
            """Full-episode admission: the teacher must EXECUTE the pose to
            success with zero safety events (an IK-only screen admits poses
            whose plans solve but whose executions trip contacts or stall)."""
            adapter = MujocoTaskAdapter(screen_with_model)
            try:
                adapter.reset(candidate)
                controller = PrivilegedStagedController()
                controller.reset(adapter)
                state = PickPlaceEvaluationState()
                evaluation = None
                for _ in range(contract.max_actions):
                    raw, _, current = adapter.observation(render_pixels=False)
                    requested = controller.predict(raw, current, adapter)
                    command = adapter.apply_policy_command(requested)
                    adapter.advance_control_period()
                    measurement, _ = adapter.pick_place_measurement(
                        command,
                        footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
                    )
                    state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                    if (
                        measurement.pickup.unsafe_contact
                        or measurement.pickup.command_bound_violation
                        or measurement.pickup.delta_limiter_activated
                        or measurement.pickup.nonfinite_command
                    ):
                        return False
                    if evaluation.terminated or evaluation.truncated:
                        break
                return bool(
                    evaluation is not None
                    and evaluation.success
                    and not evaluation.invalidated
                )
            except RuntimeError:
                return False
            finally:
                adapter.close()

    scenarios = []
    attempts = 0
    while len(scenarios) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("randomized suite sampling failed to fill the region")
        x = float(rng.uniform(*x_range))
        y = float(rng.uniform(*y_range))
        if exclude_napkin and _NAPKIN_X[0] <= x <= _NAPKIN_X[1] and _NAPKIN_Y[0] <= y <= _NAPKIN_Y[1]:
            continue
        candidate = SimulationScenario(
            scenario_id=f"random_{len(scenarios):03d}",
            cube_position_m=(x, y, nominal.cube_position_m[2]),
            robot_qpos_mujoco=nominal.robot_qpos_mujoco,
            cube_quaternion_wxyz=nominal.cube_quaternion_wxyz,
            fixed_appearance=True,
        )
        if screen is not None and not screen(candidate):
            continue
        scenarios.append(candidate)
    # Content-distinct id: two generations with the same seed/count but
    # different sampling (e.g. IK-screened vs not) must never share an
    # artifact namespace — the immutability guards would rightly collide.
    from so_arm101_v2.data._serialization import content_sha256

    scenario_hash = content_sha256(
        [list(item.cube_position_m) for item in scenarios]
    )[:8]
    return SimulationSuite(
        suite_id=f"random_pick_place_v3_seed{generator_seed}_n{count}_{scenario_hash}",
        schema_version=1,
        task_contract=base.task_contract,
        repeats=repeats,
        diagnostic_only=False,
        scenarios=tuple(scenarios),
        recovery_probe=None,
    )


def load_suite_from_path(path) -> SimulationSuite:
    """Load a generated suite JSON (content-hash verified)."""
    from pathlib import Path as _Path

    from so_arm101_v2.data._serialization import content_sha256

    payload = json.loads(_Path(path).read_text(encoding="utf-8"))
    stated = payload.pop("content_sha256")
    if content_sha256(payload) != stated:
        raise ValueError("generated suite content hash mismatch")
    return _suite_from_payload(payload)


def load_simulation_suite(name: str) -> SimulationSuite:
    if name not in {
        "fixed_pickup_contract_v1",
        "fixed_pickup_recovery_probe_v1",
        "fixed_pick_place_v3",
    }:
        raise ValueError(f"unknown simulation suite: {name!r}")
    resource = files("so_arm101_v2.data.resources").joinpath(f"{name}.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    scenarios = tuple(SimulationScenario(
        scenario_id=str(item["scenario_id"]),
        cube_position_m=tuple(float(value) for value in item["cube_position_m"]),
        robot_qpos_mujoco=tuple(float(value) for value in item["robot_qpos_mujoco"]),
        cube_quaternion_wxyz=tuple(float(value) for value in item["cube_quaternion_wxyz"]),
        fixed_appearance=bool(item["fixed_appearance"]),
    ) for item in payload["scenarios"])
    probe = payload.get("recovery_probe")
    return SimulationSuite(
        suite_id=str(payload["suite_id"]), schema_version=int(payload["schema_version"]),
        task_contract=str(payload["task_contract"]), repeats=int(payload["repeats"]),
        diagnostic_only=bool(payload["diagnostic_only"]), scenarios=scenarios,
        recovery_probe=None if probe is None else {str(k): int(v) for k, v in probe.items()},
    )


__all__ = ["SimulationScenario", "SimulationSuite", "load_simulation_suite"]
