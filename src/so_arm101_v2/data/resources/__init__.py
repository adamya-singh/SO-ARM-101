"""Versioned JSON resources shipped with :mod:`so_arm101_v2`."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def _resource_names() -> frozenset[str]:
    return frozenset(
        entry.name
        for entry in files(__package__).iterdir()
        if entry.is_file() and entry.name.endswith(".json")
    )


RESOURCE_NAMES = _resource_names()


def read_resource_bytes(name: str) -> bytes:
    """Read a known v2 resource without relying on a repository-relative path."""
    if not name.endswith(".json") or "/" in name or "\\" in name or name not in RESOURCE_NAMES:
        raise ValueError(f"Unknown so_arm101_v2 resource: {name!r}")
    return files(__package__).joinpath(name).read_bytes()


def load_json_resource(name: str) -> Any:
    """Load a known UTF-8 JSON resource."""
    return json.loads(read_resource_bytes(name).decode("utf-8"))


__all__ = ["RESOURCE_NAMES", "load_json_resource", "read_resource_bytes"]
