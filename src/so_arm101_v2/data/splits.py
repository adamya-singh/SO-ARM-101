"""Deterministic, provenance-stratified episode manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ._serialization import content_sha256, write_immutable_json

if TYPE_CHECKING:
    from .inventory import DatasetInventory


SPLIT_VERSION = "split_v1"
SPLIT_QUOTAS: Mapping[str, tuple[int, int, int]] = {
    "repo_batch_000_016": (13, 2, 2),
    "repo_batch_017_049": (27, 3, 3),
    "repo_batch_050_099": (40, 5, 5),
}


@dataclass(frozen=True)
class SplitManifest:
    """An immutable episode-level split tied to one dataset digest."""

    schema_version: int
    split_version: str
    dataset_id: str
    dataset_digest: str
    split: str
    episode_ids: tuple[int, ...]
    episode_count: int
    frame_count: int
    provenance_batch_counts: tuple[tuple[str, int], ...]
    ranking_key: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {self.split!r}")
        if tuple(sorted(set(self.episode_ids))) != self.episode_ids:
            raise ValueError("episode_ids must be sorted and unique")
        if self.episode_count != len(self.episode_ids):
            raise ValueError("episode_count does not match episode_ids")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_version": self.split_version,
            "dataset_id": self.dataset_id,
            "dataset_digest": self.dataset_digest,
            "split": self.split,
            "episode_ids": list(self.episode_ids),
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "provenance_batch_counts": dict(self.provenance_batch_counts),
            "ranking_key": self.ranking_key,
            "content_sha256": self.content_sha256,
        }


def _rank(dataset_id: str, episode_id: int) -> str:
    text = f"{dataset_id}:{SPLIT_VERSION}:{episode_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest(
    inventory: "DatasetInventory", split: str, episode_ids: list[int]
) -> SplitManifest:
    episode_ids.sort()
    records = {record.episode_id: record for record in inventory.episodes}
    batch_counts = tuple(
        (batch.batch_id, sum(records[index].provenance_batch == batch.batch_id for index in episode_ids))
        for batch in inventory.provenance_batches
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "split_version": SPLIT_VERSION,
        "dataset_id": inventory.dataset_id,
        "dataset_digest": inventory.dataset_digest,
        "split": split,
        "episode_ids": episode_ids,
        "episode_count": len(episode_ids),
        "frame_count": sum(records[index].frame_count for index in episode_ids),
        "provenance_batch_counts": dict(batch_counts),
        "ranking_key": "sha256(<dataset_id>:split_v1:<episode_id>)",
    }
    return SplitManifest(
        episode_ids=tuple(episode_ids),
        provenance_batch_counts=batch_counts,
        content_sha256=content_sha256(body),
        **{key: value for key, value in body.items() if key not in {"episode_ids", "provenance_batch_counts"}},
    )


def split_resource_name(manifest: SplitManifest) -> str:
    return (
        f"{manifest.dataset_id}.split.{manifest.split}.v1."
        f"{manifest.dataset_digest[:12]}.json"
    )


def build_split_manifests(
    inventory: "DatasetInventory", *, output_dir: str | Path | None = None
) -> dict[str, SplitManifest]:
    """Build fixed 80/10/10 splits and optionally write them immutably."""
    by_batch: dict[str, list[int]] = {key: [] for key in SPLIT_QUOTAS}
    for episode in inventory.episodes:
        if episode.provenance_batch not in by_batch:
            raise ValueError(f"unexpected provenance batch: {episode.provenance_batch}")
        by_batch[episode.provenance_batch].append(episode.episode_id)

    selected = {"train": [], "validation": [], "test": []}
    for batch_id, quotas in SPLIT_QUOTAS.items():
        ranked = sorted(
            by_batch[batch_id], key=lambda episode_id: _rank(inventory.dataset_id, episode_id)
        )
        train_count, validation_count, test_count = quotas
        if len(ranked) != sum(quotas):
            raise ValueError(
                f"batch {batch_id} has {len(ranked)} episodes, expected {sum(quotas)}"
            )
        selected["train"].extend(ranked[:train_count])
        selected["validation"].extend(
            ranked[train_count : train_count + validation_count]
        )
        selected["test"].extend(ranked[-test_count:])

    manifests = {
        split: _manifest(inventory, split, episode_ids)
        for split, episode_ids in selected.items()
    }
    if output_dir is not None:
        destination = Path(output_dir)
        for manifest in manifests.values():
            write_immutable_json(destination / split_resource_name(manifest), manifest.to_dict())
    return manifests


__all__ = [
    "SPLIT_QUOTAS",
    "SPLIT_VERSION",
    "SplitManifest",
    "build_split_manifests",
    "split_resource_name",
]
