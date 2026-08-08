"""Atomic immutable-write guarantees under concurrency and crashes."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from so_arm101_v2.data._serialization import (
    canonical_json_bytes,
    write_immutable_bytes,
    write_immutable_json,
)


def _hammer(arguments: tuple[str, int]) -> str:
    path, _ = arguments
    write_immutable_json(Path(path), {"value": 7, "nested": [1, 2, 3]})
    return "ok"


def test_concurrent_identical_writers_all_succeed(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=8) as pool:
        results = pool.map(_hammer, [(str(target), index) for index in range(16)])
    assert results == ["ok"] * 16
    assert json.loads(target.read_text()) == {"value": 7, "nested": [1, 2, 3]}
    assert target.read_bytes() == canonical_json_bytes(
        {"value": 7, "nested": [1, 2, 3]}, pretty=True
    )
    residue = [item for item in tmp_path.iterdir() if item.name != "artifact.json"]
    assert residue == []


def test_existing_identical_file_is_a_noop(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_immutable_json(target, {"a": 1})
    write_immutable_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_existing_conflicting_file_raises_with_legacy_message(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_immutable_json(target, {"a": 1})
    with pytest.raises(FileExistsError, match="immutable artifact already exists"):
        write_immutable_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}


def test_custom_conflict_message(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    write_immutable_bytes(target, b"one")
    with pytest.raises(FileExistsError, match="custom label"):
        write_immutable_bytes(target, b"two", conflict_message="custom label")


def test_link_race_identical_content_succeeds(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "artifact.bin"
    real_link = os.link

    def racing_link(source: str, destination: str) -> None:
        # A concurrent writer publishes the same bytes just before us.
        Path(destination).write_bytes(b"payload")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    write_immutable_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert [item.name for item in tmp_path.iterdir()] == ["artifact.bin"]


def test_link_race_conflicting_content_raises(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "artifact.bin"
    real_link = os.link

    def racing_link(source: str, destination: str) -> None:
        Path(destination).write_bytes(b"different")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        write_immutable_bytes(target, b"payload")
    assert target.read_bytes() == b"different"
    assert [item.name for item in tmp_path.iterdir()] == ["artifact.bin"]


def test_crash_during_publish_leaves_no_partial_artifact(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "artifact.bin"

    def broken_link(source: str, destination: str) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(os, "link", broken_link)
    with pytest.raises(RuntimeError, match="simulated crash"):
        write_immutable_bytes(target, b"payload")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
