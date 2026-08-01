"""Physical-data inventory, immutable splits, and explicit future targets."""

from .inventory import DatasetInventory, EpisodeRecord, inventory_physical_dataset
from .resources import load_json_resource, read_resource_bytes
from .samples import (
    FutureStateSample,
    SampleReference,
    load_future_state_samples,
    preprocess_wrist_image,
)
from .splits import SplitManifest, build_split_manifests
from .targets import (
    FutureTargetIndex,
    FutureTargetSpec,
    PaddingPolicy,
    audit_future_target_candidates,
    build_future_target_index,
    gather_future_state_targets,
)

__all__ = [
    "DatasetInventory",
    "EpisodeRecord",
    "FutureTargetIndex",
    "FutureTargetSpec",
    "FutureStateSample",
    "PaddingPolicy",
    "SampleReference",
    "SplitManifest",
    "audit_future_target_candidates",
    "build_future_target_index",
    "build_split_manifests",
    "gather_future_state_targets",
    "inventory_physical_dataset",
    "load_future_state_samples",
    "load_json_resource",
    "read_resource_bytes",
    "preprocess_wrist_image",
]
