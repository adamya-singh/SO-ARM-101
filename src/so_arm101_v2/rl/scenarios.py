"""Training and monitoring scenarios for RL, disjoint from every evaluation and capture suite (2026-10-04).

Training poses come from teacher-screened bench suites (simulation/bench.generate_bench_suite with appearance
and placement randomization) on seeds no earlier suite used: the held-out suite is seed 8, the captures are
seeds 12 (ladder, 2400) and 14 (4800). Training shards use seeds 30-37 (screened in parallel, one shard per
process), the monitoring suite seed 21. Every RL run walks the same fixed order of training poses (common
random numbers across methods): episode i always uses ``order[i % len(order)]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.simulation.suites import SimulationScenario, load_suite_from_path

RESERVED_SEEDS = (8, 12, 14)          # held-out suite, ladder capture, 4800 capture
TRAIN_SEEDS = tuple(range(30, 38))
MONITOR_SEED = 21
ORDER_SEED = 20261004


def scenario_key(s: SimulationScenario) -> tuple:
    """Identity of a pose for disjointness: cube position, square placement and appearance seed."""
    return (tuple(round(float(v), 6) for v in s.cube_position_m),
            None if s.square_center_xy is None else tuple(round(float(v), 6) for v in s.square_center_xy),
            round(float(s.square_yaw_rad), 6), s.appearance_seed)


def assert_disjoint(train: Iterable[SimulationScenario], others: Iterable[SimulationScenario]) -> None:
    taken = {scenario_key(s) for s in others}
    clash = [s.scenario_id for s in train if scenario_key(s) in taken]
    if clash:
        raise ValueError(f"{len(clash)} training scenarios coincide with evaluation/capture scenarios: {clash[:5]}")


class ScenarioPool:
    """All training shards, deduplicated, in one fixed order."""

    def __init__(self, scenarios: list[SimulationScenario], *, order_seed: int = ORDER_SEED) -> None:
        seen, unique = set(), []
        for s in scenarios:
            key = scenario_key(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        if not unique:
            raise ValueError("empty scenario pool")
        self.scenarios = unique
        self.order = np.random.default_rng(order_seed).permutation(len(unique))
        self.order_sha256 = content_sha256(dict(ids=[unique[i].scenario_id + str(scenario_key(unique[i])) for i in self.order]))

    def __len__(self) -> int:
        return len(self.scenarios)

    def episode(self, index: int) -> SimulationScenario:
        return self.scenarios[int(self.order[index % len(self.order)])]

    @classmethod
    def from_index(cls, path: str | Path) -> "ScenarioPool":
        payload = json.loads(Path(path).read_text())
        scenarios: list[SimulationScenario] = []
        for suite_path in payload["train_suites"]:
            scenarios.extend(load_suite_from_path(suite_path).scenarios)
        return cls(scenarios, order_seed=int(payload.get("order_seed", ORDER_SEED)))


def load_monitor(path: str | Path) -> list[SimulationScenario]:
    payload = json.loads(Path(path).read_text())
    return list(load_suite_from_path(payload["monitor_suite"]).scenarios)


__all__ = ["MONITOR_SEED", "ORDER_SEED", "RESERVED_SEEDS", "ScenarioPool", "TRAIN_SEEDS", "assert_disjoint", "load_monitor", "scenario_key"]
