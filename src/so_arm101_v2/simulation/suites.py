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


def load_simulation_suite(name: str) -> SimulationSuite:
    if name not in {"fixed_pickup_contract_v1", "fixed_pickup_recovery_probe_v1"}:
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
