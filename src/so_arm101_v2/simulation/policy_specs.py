"""Picklable policy construction specs for process-parallel rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PolicySpec:
    """Reconstructible description of a simulation policy.

    Rollout worker processes rebuild policies from specs, so every field must
    stay picklable and position-independent (checkpoint paths are resolved
    strings, options are sorted primitive pairs).
    """

    kind: str
    checkpoint: str | None = None
    options: tuple[tuple[str, Any], ...] = ()

    def option(self, name: str, default: Any = None) -> Any:
        for key, value in self.options:
            if key == name:
                return value
        return default

    def build(self) -> Any:
        try:
            builder = _POLICY_BUILDERS[self.kind]
        except KeyError:
            raise ValueError(f"unknown policy spec kind: {self.kind!r}") from None
        return builder(self)


def _required_checkpoint(spec: PolicySpec) -> str:
    if not spec.checkpoint:
        raise ValueError(f"policy spec kind {spec.kind!r} requires a checkpoint path")
    return spec.checkpoint


def _build_privileged_staged(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController

    return PrivilegedStagedController()


def _build_current_pose(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.rollout import CurrentPosePolicy

    return CurrentPosePolicy()


def _build_constant_pose(spec: PolicySpec) -> Any:
    import numpy as np

    from so_arm101_v2.simulation.rollout import ConstantPosePolicy

    target = spec.option("target_act")
    if target is None:
        raise ValueError("constant_pose spec requires a target_act option")
    return ConstantPosePolicy(np.asarray(target, dtype=np.float32))


def _build_torch_checkpoint(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.rollout import TorchCheckpointPolicy

    return TorchCheckpointPolicy(
        _required_checkpoint(spec),
        black_image=bool(spec.option("black_image", False)),
    )


def _build_chunked_clone(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.chunked import ChunkedClonePolicy

    return ChunkedClonePolicy(
        _required_checkpoint(spec),
        clamp_channels=tuple(spec.option("clamp_channels", ())),
    )


def _build_oracle_clone(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.clone_policy import OracleCloneCheckpointPolicy

    return OracleCloneCheckpointPolicy(_required_checkpoint(spec))


def _build_vision_chunked(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy

    return VisionChunkedPolicy(
        _required_checkpoint(spec),
        black_image=bool(spec.option("black_image", False)),
        clamp_channels=tuple(spec.option("clamp_channels", ())),
    )


def _build_diagnostic_clone(spec: PolicySpec) -> Any:
    from so_arm101_v2.simulation.correction import NonPromotableDiagnosticClonePolicy

    return NonPromotableDiagnosticClonePolicy(_required_checkpoint(spec))


_POLICY_BUILDERS: dict[str, Callable[[PolicySpec], Any]] = {
    "privileged_staged": _build_privileged_staged,
    "current_pose": _build_current_pose,
    "constant_pose": _build_constant_pose,
    "torch_checkpoint": _build_torch_checkpoint,
    "chunked_clone": _build_chunked_clone,
    "vision_chunked": _build_vision_chunked,
    "oracle_clone": _build_oracle_clone,
    "diagnostic_clone": _build_diagnostic_clone,
}


def build_policy(spec: PolicySpec) -> Any:
    return spec.build()


__all__ = ["PolicySpec", "build_policy"]
