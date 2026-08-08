"""Randomized suite generation and hash-verified path loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.simulation.suites import (
    generate_randomized_suite,
    load_simulation_suite,
    load_suite_from_path,
    suite_payload,
)

_NAPKIN_X = (0.0546, 0.1054)
_NAPKIN_Y = (0.2746, 0.3254)


def test_generator_is_deterministic_and_respects_the_region() -> None:
    first = generate_randomized_suite(7, 25, repeats=1)
    second = generate_randomized_suite(7, 25, repeats=1)
    assert suite_payload(first) == suite_payload(second)
    assert first.suite_id.startswith("random_pick_place_v3_seed7_n25_")
    assert len(first.suite_id.rsplit("_", 1)[1]) == 8
    assert len(first.scenarios) == 25 and first.repeats == 1
    nominal = next(
        item for item in load_simulation_suite("fixed_pick_place_v3").scenarios
        if item.scenario_id == "nominal"
    )
    for scenario in first.scenarios:
        x, y, z = scenario.cube_position_m
        assert -0.06 <= x <= 0.06 and 0.24 <= y <= 0.31
        assert z == nominal.cube_position_m[2]
        assert not (_NAPKIN_X[0] <= x <= _NAPKIN_X[1] and _NAPKIN_Y[0] <= y <= _NAPKIN_Y[1])
        assert scenario.robot_qpos_mujoco == nominal.robot_qpos_mujoco
    different = generate_randomized_suite(8, 25, repeats=1)
    assert suite_payload(different)["scenarios"] != suite_payload(first)["scenarios"]


def test_load_suite_from_path_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    suite = generate_randomized_suite(7, 5, repeats=2)
    payload = suite_payload(suite)
    payload["generator"] = {"generator_seed": 7, "count": 5}
    payload["content_sha256"] = content_sha256(payload)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_suite_from_path(path)
    assert suite_payload(loaded) == suite_payload(suite)
    tampered = dict(payload)
    tampered["repeats"] = 5
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_suite_from_path(path)
