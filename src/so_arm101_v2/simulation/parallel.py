"""Shared auto-sizing for process-parallel rollouts.

Worker count can never change report bytes: rollout results are collected in
submission order and every rollout owns its digest-keyed output paths, so
sizing is purely a wall-clock decision.
"""

from __future__ import annotations

import os


def default_worker_count(*, record_video: bool, task_count: int | None = None) -> int:
    """min(task_count, 10 with video, cores-2 without), floor 1.

    Video encoding holds an EGL context plus a libx264 encoder per rollout:
    cap 10 on this class of machine.  Without video, leave two cores for the
    parent process and the OS.
    """
    cores = os.cpu_count() or 1
    cap = max(1, cores - 2)
    if record_video:
        cap = min(cap, 10)
    if task_count is not None:
        cap = min(cap, max(1, task_count))
    return cap


def resolve_workers(
    requested: int | None, *, record_video: bool, task_count: int | None = None
) -> int:
    """None means auto; an explicit integer (including 1) is honored as-is."""
    if requested is not None:
        return requested
    return default_worker_count(record_video=record_video, task_count=task_count)


__all__ = ["default_worker_count", "resolve_workers"]
