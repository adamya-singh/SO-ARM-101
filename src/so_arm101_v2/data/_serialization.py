"""Canonical JSON and immutable artifact helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    else:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    return text.encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_immutable_json(path: Path, value: Any) -> None:
    """Create a canonical JSON artifact, rejecting conflicting regeneration."""
    expected = canonical_json_bytes(value, pretty=True)
    if path.exists():
        if path.read_bytes() != expected:
            raise FileExistsError(
                f"immutable artifact already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


__all__ = [
    "canonical_json_bytes",
    "content_sha256",
    "write_immutable_json",
]
