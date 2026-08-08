from __future__ import annotations

import os
from pathlib import Path

import pytest

# The suite pins the LEGACY cpu-eager numerics lane: identity/digest tests
# assert byte-stable legacy artifacts, and unmarked tests must never fork
# onto the GPU regime just because CUDA is live on this box.  Regime-v2
# tests opt in explicitly (tests/test_numerics.py).
os.environ.setdefault("SO_ARM101_V2_NUMERICS", "legacy")

from so_arm101_v2.data import inventory_physical_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_DATASET_ROOT = (
    REPOSITORY_ROOT / "imitation-learning/datasets/so101_pickplace_v1"
)


@pytest.fixture(scope="session")
def physical_inventory():
    return inventory_physical_dataset(PHYSICAL_DATASET_ROOT)
