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


def test_delegating_wrappers_preserve_message_prefixes(tmp_path: Path) -> None:
    from so_arm101_v2.learning.chunked import _write_immutable_bytes as chunked_write
    from so_arm101_v2.simulation.correction import _write_immutable_bytes as correction_write
    from so_arm101_v2.simulation.recovery import _write_immutable_bytes as recovery_write

    for writer, prefix in (
        (chunked_write, "immutable chunked artifact differs"),
        (correction_write, "immutable correction artifact differs"),
        (recovery_write, "immutable recovery artifact differs"),
    ):
        target = tmp_path / f"{prefix.split()[1]}.bin"
        writer(target, b"one")
        writer(target, b"one")
        with pytest.raises(FileExistsError, match=prefix):
            writer(target, b"two")


def test_write_immutable_file_streams_and_rejects_conflicts(tmp_path) -> None:
    from so_arm101_v2.data._serialization import write_immutable_file

    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (2 << 20))
    destination = tmp_path / "out" / "images.npy"
    write_immutable_file(destination, source)
    assert destination.read_bytes() == source.read_bytes()
    write_immutable_file(destination, source)  # identical re-publish accepted
    divergent = tmp_path / "divergent.bin"
    divergent.write_bytes(b"y" * (2 << 20))
    with pytest.raises(FileExistsError, match="different content"):
        write_immutable_file(destination, divergent)
    # same size, different bytes: streamed comparison must still catch it
    shifted = tmp_path / "shifted.bin"
    shifted.write_bytes(b"x" * ((2 << 20) - 1) + b"z")
    with pytest.raises(FileExistsError, match="different content"):
        write_immutable_file(destination, shifted)
    assert not list(destination.parent.glob(".*.tmp"))


def test_write_immutable_file_move_publishes_without_copy(tmp_path):
    """2026-09-11: a 226 GB frames sidecar died copying itself; move=True hard-links the source into place."""
    import os
    from so_arm101_v2.data._serialization import write_immutable_file
    source = tmp_path / "images.npy"; source.write_bytes(b"frames" * 1000)
    target = tmp_path / "oracle" / "images.npy"
    write_immutable_file(target, source, move=True)
    assert target.read_bytes() == b"frames" * 1000 and not source.exists()
    # Same content again: accepted, the new source is consumed.
    again = tmp_path / "again.npy"; again.write_bytes(b"frames" * 1000)
    write_immutable_file(target, again, move=True)
    assert not again.exists() and target.read_bytes() == b"frames" * 1000
    # Different content: refused, target untouched.
    other = tmp_path / "other.npy"; other.write_bytes(b"other")
    import pytest
    with pytest.raises(FileExistsError):
        write_immutable_file(target, other, move=True)
    assert target.read_bytes() == b"frames" * 1000 and other.exists()
    assert os.stat(target).st_nlink == 1


def test_truncate_npy_in_place_matches_np_save_of_the_slice(tmp_path):
    import numpy as np
    from so_arm101_v2.simulation.oracle import truncate_npy_in_place
    rng = np.random.default_rng(0)
    for total, keep in ((7, 3), (480 * 4, 480 * 3), (12, 12), (1000, 1)):
        path = tmp_path / f"frames_{total}.npy"
        data = rng.integers(0, 255, size=(total, 4, 5, 3), dtype=np.uint8)
        np.save(path, data)
        expected = tmp_path / f"expected_{keep}.npy"; np.save(expected, data[:keep])
        assert truncate_npy_in_place(path, keep)
        assert path.read_bytes() == expected.read_bytes()
        assert np.load(path).shape == (keep, 4, 5, 3)
    path = tmp_path / "grow.npy"; np.save(path, data)
    assert not truncate_npy_in_place(path, total + 1)   # cannot grow; file untouched
    assert np.load(path).shape == (total, 4, 5, 3)
