"""Canonical JSON and immutable artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


def write_immutable_bytes(
    path: Path, data: bytes, *, conflict_message: str | None = None
) -> None:
    """Atomically create an immutable artifact, rejecting conflicting regeneration.

    Publishes via hard link so concurrent writers race on an atomic
    create-exclusive: identical content from a concurrent writer is accepted,
    divergent content raises, and a killed writer leaves only a stray temp file
    (never a torn artifact).
    """

    def _conflict() -> FileExistsError:
        return FileExistsError(
            conflict_message
            or f"immutable artifact already exists with different content: {path}"
        )

    if path.exists():
        if path.read_bytes() != data:
            raise _conflict()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise _conflict() from None
    finally:
        os.unlink(temp_name)


def _files_identical(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as a, second.open("rb") as b:
        while True:
            block_a = a.read(1 << 22)
            block_b = b.read(1 << 22)
            if block_a != block_b:
                return False
            if not block_a:
                return True


def write_immutable_file(
    path: Path, source: Path, *, conflict_message: str | None = None
) -> None:
    """Streaming variant of write_immutable_bytes for artifacts too large to
    hold in memory (e.g. multi-GB frames sidecars).  Same atomic
    create-exclusive publish and conflict semantics; content is compared by
    streaming, never fully materialized."""
    import shutil

    def _conflict() -> FileExistsError:
        return FileExistsError(
            conflict_message
            or f"immutable artifact already exists with different content: {path}"
        )

    if path.exists():
        if not _files_identical(path, source):
            raise _conflict()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, handle, length=1 << 22)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if not _files_identical(path, source):
                raise _conflict() from None
    finally:
        os.unlink(temp_name)


def write_immutable_json(path: Path, value: Any) -> None:
    """Create a canonical JSON artifact, rejecting conflicting regeneration."""
    write_immutable_bytes(path, canonical_json_bytes(value, pretty=True))


__all__ = [
    "canonical_json_bytes",
    "content_sha256",
    "write_immutable_bytes",
    "write_immutable_file",
    "write_immutable_json",
]
