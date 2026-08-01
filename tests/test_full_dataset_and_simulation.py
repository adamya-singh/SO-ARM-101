from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import REPOSITORY_ROOT
from conftest import PHYSICAL_DATASET_ROOT
from so_arm101_v2.data import build_split_manifests
from so_arm101_v2.data.samples import _episode_video_offsets
from so_arm101_v2.learning import (
    ImageSignalStatus,
    Lead3TrainingCache,
    SmallModelKind,
    build_small_model,
    classify_image_signal,
    cross_episode_image_indices,
)
from so_arm101_v2.learning.full_dataset import _cache_rows
from so_arm101_v2.simulation import (
    MujocoTaskAdapter,
    evaluate_face_grasp_contacts,
    load_simulation_suite,
)


def _synthetic_cache(tmp_path: Path) -> Lead3TrainingCache:
    count = 8
    images = np.zeros((count, 256, 256, 3), dtype=np.uint8)
    episodes = np.array([0, 0, 1, 1, 10, 10, 11, 11], dtype=np.int64)
    return Lead3TrainingCache(
        directory=tmp_path, metadata_path=tmp_path / "metadata.json",
        images_path=tmp_path / "images.npy", arrays_path=tmp_path / "anchors.npz",
        dataset_digest="synthetic", train_count=4, validation_count=4,
        images=images, episode_ids=episodes,
        frame_indices=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        split_codes=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
        current_states=np.zeros((count, 6), dtype=np.float32),
        targets=np.ones((count, 6), dtype=np.float32),
    )


def test_small_model_architectures_have_expected_shapes_and_gradients() -> None:
    bias = np.zeros(6, dtype=np.float32)
    for kind in SmallModelKind:
        model = build_small_model(kind, bias)
        images = torch.rand(3, 3, 256, 256)
        states = torch.rand(3, 6)
        output = model(images, states)
        assert output.shape == (3, 6)
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_cross_episode_image_derangement_never_keeps_episode(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path)
    validation = cache.indices("validation")
    donors = cross_episode_image_indices(cache, validation)
    assert donors.shape == validation.shape
    assert np.all(cache.episode_ids[donors] != cache.episode_ids[validation])


def test_real_lead3_cache_rows_exclude_test_and_match_future_state(physical_inventory) -> None:
    import pyarrow.parquet as pq

    manifests = build_split_manifests(physical_inventory)
    offsets = _episode_video_offsets(PHYSICAL_DATASET_ROOT, pq, physical_inventory.fps)
    rows = _cache_rows(physical_inventory, manifests, offsets)
    assert int(np.sum(rows["split_codes"] == 0)) == 33125
    assert int(np.sum(rows["split_codes"] == 1)) == 3852
    assert not set(rows["episode_ids"]).intersection(manifests["test"].episode_ids)
    np.testing.assert_array_equal(
        rows["targets"], physical_inventory.states[rows["global_rows"] + 3]
    )


@pytest.mark.parametrize(
    ("clean", "state", "blank", "shuffled", "expected"),
    [
        (0.80, 1.00, 1.10, 1.10, ImageSignalStatus.USEFUL),
        (1.00, 1.00, 1.01, 0.99, ImageSignalStatus.NOT_USED),
        (1.10, 1.00, 1.30, 1.30, ImageSignalStatus.USED_NOT_HELPFUL),
    ],
)
def test_image_signal_classifications(
    clean: float, state: float, blank: float, shuffled: float, expected: ImageSignalStatus
) -> None:
    shape = (3, 10)
    status, evidence = classify_image_signal(
        np.full(shape, clean), np.full(shape, state),
        np.full(shape, blank), np.full(shape, shuffled),
    )
    assert status is expected
    assert set(evidence["episode_bootstrap_95ci"]) == {
        "state_only_minus_clean", "blank_minus_clean", "shuffled_minus_clean"
    }


def test_strict_face_detector_requires_centered_opposing_forces() -> None:
    contacts = [
        {"side": "fixed", "position_local": [-0.0125, 0.0, 0.0], "normal_local": [-1, 0, 0], "force": 0.2},
        {"side": "moving", "position_local": [0.0125, 0.0, 0.0], "normal_local": [1, 0, 0], "force": 0.2},
    ]
    strict, force, diagnostics = evaluate_face_grasp_contacts(contacts, np.array([1.0, 0.0, 0.0]))
    assert strict and force == pytest.approx(0.2)
    assert diagnostics["bilateral_interior_face_contact"]
    # contacts closer than the 2 mm corner margin to a cube edge stay rejected
    # (the margin was 4 mm for the legacy blade jaws; the Menagerie fingertip
    # pads legitimately contact within ~2.5 mm of the edges of a 25 mm cube)
    contacts[1]["position_local"] = [0.0125, 0.0115, 0.0115]
    assert not evaluate_face_grasp_contacts(contacts, np.array([1.0, 0.0, 0.0]))[0]


def test_named_suites_are_small_fixed_and_diagnostic() -> None:
    primary = load_simulation_suite("fixed_pickup_contract_v1")
    recovery = load_simulation_suite("fixed_pickup_recovery_probe_v1")
    assert len(primary.scenarios) == 5 and primary.repeats == 3 and not primary.diagnostic_only
    assert len(recovery.scenarios) == 1 and recovery.diagnostic_only
    assert recovery.recovery_probe["recovery_window_actions"] == 60
    for scenario in primary.scenarios:
        assert np.linalg.norm(np.asarray(scenario.cube_position_m[:2]) - np.array([0.0, 0.3])) <= 0.002


def test_mujoco_adapter_reset_rate_and_reward_independent_measurement() -> None:
    adapter = MujocoTaskAdapter(REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml")
    try:
        scenario = load_simulation_suite("fixed_pickup_contract_v1").scenarios[0]
        adapter.reset(scenario)
        start = float(adapter.data.time)
        current = adapter.current_act()
        command = adapter.apply_policy_command(current)
        substeps = adapter.advance_control_period()
        measurement, diagnostics = adapter.measurement(command)
        assert substeps in {16, 17}
        assert adapter.data.time - start == pytest.approx(1 / 30, abs=0.002)
        assert not measurement.command_bound_violation
        assert "grip_force_n" in diagnostics
        assert not hasattr(measurement, "reward")
    finally:
        adapter.close()
