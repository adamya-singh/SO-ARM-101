from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from conftest import PHYSICAL_DATASET_ROOT, REPOSITORY_ROOT
from so_arm101_v2.contracts import JOINT_NAMES, act_to_mujoco_qpos, clip_mujoco_qpos
from so_arm101_v2.data import (
    SampleReference,
    build_split_manifests,
    load_future_state_samples,
)
from so_arm101_v2.learning import (
    TinyModelConfig,
    build_memorization_subset,
    evaluate_baselines,
    run_tiny_overfit,
)
from so_arm101_v2.learning.baselines import baseline_resource_name
from so_arm101_v2.learning.tiny_model import MEMORIZATION_REFERENCES
from so_arm101_v2.learning.tiny_model import _build_model
from so_arm101_v2.visualization import apply_mujoco_pose, build_sample_report
from so_arm101_v2.visualization.mujoco_viewer import _SamplePlaybackController


@pytest.fixture(scope="module")
def canonical_sample(physical_inventory):
    return load_future_state_samples(
        PHYSICAL_DATASET_ROOT,
        [SampleReference(45, 168, 3)],
        inventory=physical_inventory,
    )[0]


def test_sample_report_contains_every_requested_boundary(
    canonical_sample, tmp_path: Path
) -> None:
    paths = build_sample_report(canonical_sample, tmp_path)
    payload = json.loads(paths.json.read_text())
    page = paths.html.read_text()
    assert payload["reference"] == {"episode_id": 45, "frame_index": 168, "lead_steps": 3}
    assert payload["hardware_io_performed"] is False
    assert len(payload["observed_states"]) == 4
    for key in (
        "requested_act", "clipped_mujoco", "relative_limited_physical",
        "raw_goal_ticks", "relative_limit_mask",
    ):
        assert len(payload["conversion"][key]) == 6
    assert "Raw image" in page
    assert "Final model input" in page
    assert "data:image/png;base64," in page
    build_sample_report(canonical_sample, tmp_path)


def test_mujoco_pose_adapter_assigns_named_joints(canonical_sample) -> None:
    model = mujoco.MjModel.from_xml_path(
        str(REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml")
    )
    data = mujoco.MjData(model)
    qpos, _ = clip_mujoco_qpos(act_to_mujoco_qpos(canonical_sample.future_target))
    apply_mujoco_pose(model, data, qpos)
    for name, expected in zip(JOINT_NAMES, qpos):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(float(expected))


def test_mujoco_viewer_keys_select_current_target_and_playback() -> None:
    trajectory = [np.full(6, index, dtype=np.float32) for index in range(4)]
    controller = _SamplePlaybackController(trajectory)
    np.testing.assert_array_equal(controller.handle_key(ord("T")), trajectory[-1])
    assert controller.index == 3 and not controller.playing
    np.testing.assert_array_equal(controller.handle_key(ord("C")), trajectory[0])
    assert controller.index == 0 and not controller.playing
    assert controller.handle_key(32) is None
    assert controller.playing
    np.testing.assert_array_equal(controller.advance(), trajectory[1])


def test_real_baselines_are_train_fitted_and_complete(
    physical_inventory, tmp_path: Path
) -> None:
    manifests = build_split_manifests(physical_inventory)
    result = evaluate_baselines(physical_inventory, manifests, output_dir=tmp_path)
    payload = result.to_dict()
    assert payload["mean_fit_split"] == "train"
    expected = {
        "1": {"train": 33285, "validation": 3872, "test": 4374},
        "3": {"train": 33125, "validation": 3852, "test": 4354},
        "5": {"train": 32965, "validation": 3832, "test": 4334},
    }
    for lead, split_counts in expected.items():
        for split, count in split_counts.items():
            assert payload["leads"][lead]["splits"][split]["samples"] == count
    path = tmp_path / baseline_resource_name(physical_inventory.dataset_digest)
    assert json.loads(path.read_text()) == payload


def test_memorization_subset_is_fixed_and_tiny_gate_passes(
    physical_inventory, tmp_path: Path
) -> None:
    manifests = build_split_manifests(physical_inventory)
    baseline_dir = tmp_path / "baselines"
    evaluate_baselines(physical_inventory, manifests, output_dir=baseline_dir)
    baseline_path = baseline_dir / baseline_resource_name(physical_inventory.dataset_digest)
    samples = build_memorization_subset(
        PHYSICAL_DATASET_ROOT, physical_inventory, sample_count=8
    )
    assert tuple(sample.reference for sample in samples) == MEMORIZATION_REFERENCES[:8]

    config = TinyModelConfig(sample_count=8, max_steps=500)
    result = run_tiny_overfit(
        PHYSICAL_DATASET_ROOT,
        physical_inventory,
        baseline_path,
        tmp_path / "tiny",
        config=config,
    )
    assert result.passed
    assert result.normalized_mse <= config.normalized_mse_threshold
    assert result.max_act_error <= config.max_act_error_threshold
    report = json.loads(result.report_json.read_text())
    assert report["gradient_norms_at_step_1"]["image_branch"] > 0
    assert report["gradient_norms_at_step_1"]["state_branch"] > 0
    checkpoint = torch.load(result.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["dataset_digest"] == physical_inventory.dataset_digest
    replay = run_tiny_overfit(
        PHYSICAL_DATASET_ROOT,
        physical_inventory,
        baseline_path,
        tmp_path / "tiny",
        config=config,
    )
    assert replay == result


def test_tiny_network_shape_and_samples_are_independent() -> None:
    torch.manual_seed(101)
    model = _build_model(torch, np.zeros(6, dtype=np.float32))
    images = torch.rand(2, 3, 256, 256)
    states = torch.rand(2, 6)
    with torch.inference_mode():
        batch = model(images, states)
        first_alone = model(images[:1], states[:1])
    assert batch.shape == (2, 6)
    torch.testing.assert_close(batch[:1], first_alone)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())


def test_tiny_gate_requires_matching_baseline(physical_inventory, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"dataset_digest": "wrong"}))
    with pytest.raises(ValueError, match="digest mismatch"):
        run_tiny_overfit(
            PHYSICAL_DATASET_ROOT,
            physical_inventory,
            bad,
            tmp_path / "tiny",
            config=TinyModelConfig(sample_count=1, max_steps=1),
        )
