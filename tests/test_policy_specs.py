"""PolicySpec registry: picklability, construction, and pixel requirements."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from so_arm101_v2.simulation.policy_specs import PolicySpec, build_policy


def test_spec_pickle_round_trip() -> None:
    spec = PolicySpec(
        kind="torch_checkpoint",
        checkpoint="/some/model.pt",
        options=(("black_image", True),),
    )
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown policy spec kind"):
        PolicySpec(kind="nonexistent").build()


def test_checkpoint_kinds_require_checkpoint() -> None:
    for kind in ("torch_checkpoint", "oracle_clone"):
        with pytest.raises(ValueError, match="requires a checkpoint path"):
            PolicySpec(kind=kind).build()


def test_constant_pose_requires_target_act() -> None:
    with pytest.raises(ValueError, match="target_act"):
        PolicySpec(kind="constant_pose").build()


def test_pixel_free_builders_construct_and_mark_requires_pixels() -> None:
    current = build_policy(PolicySpec(kind="current_pose"))
    assert current.requires_pixels is False

    constant = build_policy(PolicySpec(
        kind="constant_pose",
        options=(("target_act", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)),),
    ))
    assert constant.requires_pixels is False
    assert np.asarray(constant.target_act).dtype == np.float32

    privileged = build_policy(PolicySpec(kind="privileged_staged"))
    assert privileged.requires_pixels is False


def test_torch_checkpoint_requires_pixels_depends_on_kind(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from so_arm101_v2.learning.full_dataset import build_small_model
    from so_arm101_v2.learning.tiny_model import normalize_act
    from so_arm101_v2.simulation.rollout import TorchCheckpointPolicy

    mean = np.zeros(6, dtype=np.float32)
    for kind, black_image, expected in (
        ("state_only", False, False),
        ("image_state", True, False),
        ("image_state", False, True),
    ):
        model = build_small_model(kind, normalize_act(mean))
        checkpoint = tmp_path / f"{kind}.{black_image}.pt"
        torch.save({
            "model_kind": kind,
            "train_mean_target_act": mean.tolist(),
            "state_dict": model.state_dict(),
        }, checkpoint)
        policy = TorchCheckpointPolicy(checkpoint, black_image=black_image, device="cpu")
        assert policy.requires_pixels is expected, (kind, black_image)
