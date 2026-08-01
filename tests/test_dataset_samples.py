from __future__ import annotations

import numpy as np
import pytest

from conftest import PHYSICAL_DATASET_ROOT
from so_arm101_v2.data import (
    SampleReference,
    load_future_state_samples,
    preprocess_wrist_image,
)


def test_canonical_sample_loads_exact_image_state_and_target(physical_inventory) -> None:
    ref = SampleReference(45, 168, 3)
    sample = load_future_state_samples(
        PHYSICAL_DATASET_ROOT, [ref], inventory=physical_inventory
    )[0]
    record = physical_inventory.episodes[45]
    row = record.global_from_index + 168
    assert sample.reference == ref
    assert sample.raw_image.shape == (256, 256, 3)
    assert sample.raw_image.dtype == np.uint8
    assert sample.model_image.shape == (3, 256, 256)
    assert sample.model_image.dtype == np.float32
    assert float(sample.model_image.min()) >= 0
    assert float(sample.model_image.max()) <= 1
    np.testing.assert_array_equal(sample.model_image, preprocess_wrist_image(sample.raw_image))
    np.testing.assert_array_equal(sample.current_state, physical_inventory.states[row])
    np.testing.assert_array_equal(sample.future_target, physical_inventory.states[row + 3])
    np.testing.assert_array_equal(sample.observed_states, physical_inventory.states[row : row + 4])
    np.testing.assert_array_equal(
        sample.model_image[0],
        sample.raw_image[:, :, 0].astype(np.float32) / np.float32(255.0),
    )


def test_sample_batch_preserves_order_and_never_crosses_episode(physical_inventory) -> None:
    refs = [SampleReference(1, 10, 1), SampleReference(0, 20, 5)]
    samples = load_future_state_samples(
        PHYSICAL_DATASET_ROOT, refs, inventory=physical_inventory
    )
    assert [sample.reference for sample in samples] == refs
    assert samples[0].source_video_frame != samples[1].source_video_frame
    final_frame = physical_inventory.episodes[0].frame_count - 1
    with pytest.raises(ValueError, match="crosses episode end"):
        load_future_state_samples(
            PHYSICAL_DATASET_ROOT,
            [SampleReference(0, final_frame, 1)],
            inventory=physical_inventory,
        )


def test_sample_and_preprocessing_validation() -> None:
    with pytest.raises(ValueError, match="1, 3, or 5"):
        SampleReference(0, 0, 2)
    with pytest.raises(ValueError, match="RGB uint8"):
        preprocess_wrist_image(np.zeros((256, 256, 3), dtype=np.float32))
