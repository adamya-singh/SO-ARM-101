"""Precision tranche: registered cells, selection rules, and LR-schedule pins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.chunked import (
    CHUNKED_LR_FLOOR,
    ChunkedCloneConfig,
    build_chunked_clone_model,
    chunked_learning_rate,
    train_chunked_clone,
)
from so_arm101_v2.simulation.precision import (
    BUDGET_CANDIDATES,
    PRECISION_CELL_ORDER,
    SCHEDULE_CANDIDATES,
    SEED_CANDIDATES,
    cell_metrics,
    resolve_precision_stage_status,
    resolve_schedule_selection,
    select_best_cell,
)
from so_arm101_v2.simulation.scaling import FROZEN_RECIPE

from test_chunked_promotion import _write_tiny_oracle_manifest


def _cell(cell_id: str, successes: int, frames: int, *, passed: bool | None = None) -> dict:
    rollouts = [
        {
            "success": index < successes,
            "invalidated": index >= successes,
            "safety_counts": {
                "clipping_frames": frames if index == successes else 0,
                "limiting_frames": 0, "nonfinite_frames": 0, "unsafe_contact_frames": 0,
            },
        }
        for index in range(15)
    ]
    if successes >= 15:
        for item in rollouts:
            item["safety_counts"]["clipping_frames"] = 0
        frames = 0
    return {
        "cell_id": cell_id,
        "stage_a_rollouts": rollouts,
        "stage_a_passed": bool(passed if passed is not None else successes >= 15),
        "stage_a_evaluation_content_sha256": "e" * 64,
    }


def test_registered_cells_are_pinned_and_constructible() -> None:
    assert SCHEDULE_CANDIDATES == (
        ("cosine_w512_90k",
         {"hidden_width": 512, "max_steps": 90_000, "lr_schedule": "cosine_floor_v1"}),
    )
    assert BUDGET_CANDIDATES == (
        ("steps150k", {"hidden_width": 512, "max_steps": 150_000}),
        ("steps300k", {"hidden_width": 512, "max_steps": 300_000}),
        ("width1024", {"hidden_width": 1024, "max_steps": 90_000}),
    )
    assert SEED_CANDIDATES == (202, 303)
    for _, overrides in (*SCHEDULE_CANDIDATES, *BUDGET_CANDIDATES):
        ChunkedCloneConfig(**{**FROZEN_RECIPE, **overrides})
    for seed in SEED_CANDIDATES:
        ChunkedCloneConfig(**{**FROZEN_RECIPE, "hidden_width": 512, "seed": seed})


def test_chunked_learning_rate_pins() -> None:
    assert chunked_learning_rate("fixed", 1, 90_000, 1e-3) == 1e-3
    assert chunked_learning_rate("fixed", 90_000, 90_000, 1e-3) == 1e-3
    assert chunked_learning_rate("cosine_floor_v1", 1, 90_000, 1e-3) == 1e-3
    assert chunked_learning_rate("cosine_floor_v1", 90_000, 90_000, 1e-3) == CHUNKED_LR_FLOOR
    values = [
        chunked_learning_rate("cosine_floor_v1", step, 1_000, 1e-3)
        for step in range(1, 1_001)
    ]
    assert all(a >= b for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError, match="unknown chunked lr schedule"):
        chunked_learning_rate("linear", 1, 10, 1e-3)
    with pytest.raises(ValueError, match="step must lie"):
        chunked_learning_rate("fixed", 0, 10, 1e-3)
    with pytest.raises(ValueError, match="step must lie"):
        chunked_learning_rate("fixed", 11, 10, 1e-3)
    with pytest.raises(ValueError, match="above the floor"):
        chunked_learning_rate("cosine_floor_v1", 1, 10, CHUNKED_LR_FLOOR)


def test_width_1024_model_builds_with_zero_initialized_head() -> None:
    torch = pytest.importorskip("torch")

    model = build_chunked_clone_model(10, 1024, 2)
    with torch.inference_mode():
        output = model(torch.zeros((3, 10)))
    assert output.shape == (3, 12)
    assert torch.equal(output, torch.zeros_like(output))
    with pytest.raises(ValueError, match="hidden_width"):
        build_chunked_clone_model(10, 384, 2)


def test_noise_penalty_config_identity_excludes_fixed_lr_schedule(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    config = ChunkedCloneConfig(
        chunk_horizon=2, max_steps=2, saturation_mode="noise_penalty_v1",
        decoder_eta=1.0, margin_act=0.002, noise_sigma=0.05, penalty_weight=1.0,
    )
    result = train_chunked_clone(
        manifest_path, tmp_path / "output", config=config, numerics=None,
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    # The exact 10-key noise-penalty set: protects every stored v2 scaling
    # digest from the new lr_schedule field.
    assert set(report["config"]) == {
        "chunk_horizon", "seed", "hidden_width", "learning_rate", "max_steps",
        "saturation_mode", "decoder_eta", "margin_act", "noise_sigma",
        "penalty_weight",
    }
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "config", "target", "chunk_padding", "offline_role",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]


def test_cosine_schedule_enters_identity_and_report(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    fixed = train_chunked_clone(
        manifest_path, tmp_path / "fixed",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2), numerics=None,
    )
    cosine = train_chunked_clone(
        manifest_path, tmp_path / "cosine",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2, lr_schedule="cosine_floor_v1"),
        numerics=None,
    )
    report = json.loads(cosine.report_json.read_text(encoding="utf-8"))
    assert report["config"]["lr_schedule"] == "cosine_floor_v1"
    assert report["learning_rate_schedule"]["floor"] == CHUNKED_LR_FLOOR
    assert report["learning_rate_schedule"]["final_learning_rate"] == CHUNKED_LR_FLOOR
    assert cosine.directory.name != fixed.directory.name


def test_chunked_policy_id_reflects_config_seed(tmp_path: Path) -> None:
    from so_arm101_v2.simulation.chunked import ChunkedClonePolicy

    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2, seed=202),
        numerics=None,
    )
    assert ChunkedClonePolicy(result.checkpoint).policy_id == "chunked_h2.seed202"


def test_schedule_selection_truth_table() -> None:
    control = _cell("control_fixed_w512_90k", 9, 210)
    assert resolve_schedule_selection(control, _cell("cosine_w512_90k", 10, 300)) == "cosine_floor_v1"
    assert resolve_schedule_selection(control, _cell("cosine_w512_90k", 9, 100)) == "cosine_floor_v1"
    assert resolve_schedule_selection(control, _cell("cosine_w512_90k", 9, 210)) == "fixed"
    assert resolve_schedule_selection(control, _cell("cosine_w512_90k", 8, 0)) == "fixed"


def test_select_best_cell_tiebreaks() -> None:
    cells = [
        _cell("control_fixed_w512_90k", 9, 210),
        _cell("steps150k", 9, 100),
        _cell("steps300k", 9, 100),
        _cell("width1024", 12, 400),
    ]
    assert select_best_cell(cells) == "width1024"
    cells[3] = _cell("width1024", 9, 100)
    # Three-way tie on metrics: earliest in the registered change order wins.
    assert select_best_cell(cells) == "steps150k"
    with pytest.raises(ValueError, match="unregistered cell"):
        select_best_cell([_cell("mystery", 1, 1)])


def test_resolve_precision_stage_status() -> None:
    control = _cell("control_fixed_w512_90k", 9, 210)
    improved = _cell("cosine_w512_90k", 12, 40)
    passed = _cell("cosine_w512_90k", 15, 0)
    assert resolve_precision_stage_status("schedule", [improved], control=control) == "schedule_selected_cosine_floor_v1"
    assert resolve_precision_stage_status("schedule", [passed], control=control) == "schedule_selected_cosine_floor_v1_stage_a_passed"
    assert resolve_precision_stage_status("schedule", [_cell("cosine_w512_90k", 9, 210)], control=control) == "schedule_selected_fixed"

    budget = [_cell("steps150k", 9, 50), _cell("steps300k", 15, 0), _cell("width1024", 15, 0)]
    assert resolve_precision_stage_status("budget", budget) == "budget_promoted_steps300k"
    budget_fail = [_cell("steps150k", 9, 50), _cell("steps300k", 12, 10), _cell("width1024", 11, 20)]
    assert resolve_precision_stage_status("budget", budget_fail) == "budget_not_resolved"
    with pytest.raises(ValueError, match="registered order"):
        resolve_precision_stage_status("budget", list(reversed(budget)))

    best = _cell("steps300k", 15, 0)
    seeds_pass = [_cell("seed202", 15, 0), _cell("seed303", 15, 0)]
    assert resolve_precision_stage_status("seeds", seeds_pass, control=best) == "seeds_robust"
    seeds_fail = [_cell("seed202", 15, 0), _cell("seed303", 14, 3)]
    assert resolve_precision_stage_status("seeds", seeds_fail, control=best) == "seeds_not_resolved"
    with pytest.raises(ValueError, match="evaluation evidence"):
        resolve_precision_stage_status("budget", [{**_cell("steps150k", 1, 1), "stage_a_evaluation_content_sha256": None},
                                                   _cell("steps300k", 1, 1), _cell("width1024", 1, 1)])


def test_best_cell_selection_spans_all_registered_stages() -> None:
    # The schedule stage's cell must compete in later-stage selection: a
    # budget stage that only ranked its own cells against the control would
    # violate BEST_CELL_RULE (this pins the defect fixed on 2026-08-06).
    pool = [
        _cell("control_fixed_w512_90k", 9, 210),
        _cell("cosine_w512_90k", 12, 12),
        _cell("steps150k", 9, 33),
        _cell("steps300k", 9, 15),
        _cell("width1024", 3, 66),
    ]
    assert select_best_cell(pool) == "cosine_w512_90k"


def test_seed_cells_fork_digests(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    a = train_chunked_clone(
        manifest_path, tmp_path / "a",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2, seed=202), numerics=None,
    )
    b = train_chunked_clone(
        manifest_path, tmp_path / "b",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2, seed=303), numerics=None,
    )
    assert a.directory.name != b.directory.name
    assert cell_metrics(_cell("width1024", 3, 7)) == (3, 7)
    assert PRECISION_CELL_ORDER[0] == "control_fixed_w512_90k"
