from __future__ import annotations

from pathlib import Path

import pytest

from so_arm101_v2.data import inventory_physical_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_DATASET_ROOT = (
    REPOSITORY_ROOT / "imitation-learning/datasets/so101_pickplace_v1"
)


@pytest.fixture(scope="session")
def physical_inventory():
    return inventory_physical_dataset(PHYSICAL_DATASET_ROOT)
