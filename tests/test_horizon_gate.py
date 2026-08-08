"""Horizon-alignment tranche: teacher hold-tail pins, cells, and resolver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
from so_arm101_v2.simulation import MujocoTaskAdapter, load_simulation_suite
from so_arm101_v2.simulation.horizon import (
    ALIGNED_RECIPE_OVERRIDES,
    ALIGNED_TEACHER_HORIZON,
    HORIZON_SEEDS,
    resolve_horizon_status,
)
from so_arm101_v2.simulation.privileged import PrivilegedStagedController
from so_arm101_v2.simulation.scaling import FROZEN_RECIPE

from test_chunked_promotion import _write_tiny_oracle_manifest
from test_precision_gate import _cell

SCENE = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"


def test_registered_cells_are_pinned_and_constructible() -> None:
    assert ALIGNED_TEACHER_HORIZON == 480
    assert ALIGNED_RECIPE_OVERRIDES == {
        "hidden_width": 512, "max_steps": 90_000, "lr_schedule": "cosine_floor_v1",
    }
    assert HORIZON_SEEDS == (101, 202, 303)
    for seed in HORIZON_SEEDS:
        ChunkedCloneConfig(**{**FROZEN_RECIPE, **ALIGNED_RECIPE_OVERRIDES, "seed": seed})


def test_resolver_requires_all_three_seeds_to_pass() -> None:
    all_pass = [_cell(f"seed{seed}", 15, 0) for seed in HORIZON_SEEDS]
    assert resolve_horizon_status(all_pass) == "horizon_promoted_robust"
    one_fails = [_cell("seed101", 15, 0), _cell("seed202", 15, 0), _cell("seed303", 14, 3)]
    assert resolve_horizon_status(one_fails) == "horizon_not_resolved"
    with pytest.raises(ValueError, match="registered seed order"):
        resolve_horizon_status(list(reversed(all_pass)))
    with pytest.raises(ValueError, match="evaluation evidence"):
        broken = [_cell(f"seed{seed}", 15, 0) for seed in HORIZON_SEEDS]
        broken[0] = {**broken[0], "stage_a_evaluation_content_sha256": None}
        resolve_horizon_status(broken)


def test_teacher_hold_tail_is_behavior_neutral() -> None:
    """The pre-registered invariant: schedule prefix <= 450 unchanged; every
    action in 451-480 emits exactly the action-450 command."""
    scenario = next(
        item for item in load_simulation_suite("fixed_pick_place_v3").scenarios
        if item.scenario_id == "nominal"
    )
    adapter = MujocoTaskAdapter(SCENE)
    try:
        adapter.reset(scenario)
        controller = PrivilegedStagedController()
        controller.reset(adapter)
        assert controller.boundaries == (
            0, 70, 110, 145, 175, 205, 255, 285, 315, 350, 365,
            386, 403, 419, 424, 450,
        )
        # Consume the recorded plan without physics: predict() is a pure
        # function of the action index once the plan is solved.
        blank = np.empty((0,), dtype=np.uint8)
        current = adapter.current_act()
        commands = [controller.predict(blank, current, adapter) for _ in range(480)]
        anchor = commands[449]                      # action 450
        for tail_command in commands[450:]:         # actions 451-480
            assert np.array_equal(tail_command, anchor)
    finally:
        adapter.close()


def test_chunked_policy_loads_480_horizon_checkpoint(tmp_path: Path) -> None:
    from so_arm101_v2.simulation.chunked import ChunkedClonePolicy

    manifest_path, manifest = _write_tiny_oracle_manifest(tmp_path)
    # Rewrite the tiny fixture as a 480-horizon manifest twin.
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body["teacher_horizon"] = 480
    from so_arm101_v2.data._serialization import content_sha256
    body.pop("content_sha256")
    body["content_sha256"] = content_sha256(body)
    manifest_path.write_text(json.dumps(body), encoding="utf-8")

    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
        numerics=None,
    )
    policy = ChunkedClonePolicy(result.checkpoint)
    assert policy.teacher_horizon == 480
    # Deployment progress clock derives from the checkpoint horizon.
    assert policy.policy_id == "chunked_h2.seed101"
