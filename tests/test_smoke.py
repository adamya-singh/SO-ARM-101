from __future__ import annotations

import pytest

import so_arm101_v2
from so_arm101_v2.data import load_json_resource


def test_package_imports_and_resources_load_from_install() -> None:
    assert so_arm101_v2.__version__ == "0.4.0"
    assert load_json_resource("fixed_cube_pickup_v1.json")["task_id"] == "fixed_cube_pickup_v1"


def test_unknown_resources_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown so_arm101_v2 resource"):
        load_json_resource("../../legacy.json")
