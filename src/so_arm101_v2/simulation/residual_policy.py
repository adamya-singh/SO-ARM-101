"""Closed-loop policy for any residual-chunk model of the retry tranche (learning/vla_models.py), 2026-10-04.

The contract of ``Pi05ChunkPolicy`` (simulation/pi05_policy.py) without its physical-arm extras (servo-lag
compensation): the live wrist frame, ``normalize_act(current)`` and the open-loop progress clock go in, a
90-step residual chunk comes out, ``execute`` commands run before the next frame is read, the gripper
channel is clamped (gripper_clamp_v1). ``Pi05ChunkPolicy`` stays the policy for live pi0.5 runs.

``infer_batch`` is the batched entry the RL collector uses: one forward pass for every environment that
reached a chunk boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds

TEACHER_HORIZON = 480


def chunk_state(current_act: np.ndarray, action_index: int) -> np.ndarray:
    """The 7-dim model state: normalised current joints + progress on the teacher's 480-step clock."""
    from so_arm101_v2.learning.tiny_model import normalize_act
    progress = np.float32(min(int(action_index), TEACHER_HORIZON - 1) / (TEACHER_HORIZON - 1))
    return np.concatenate([normalize_act(np.asarray(current_act, np.float32)), [progress]]).astype(np.float32)


def lead_features(current_act: np.ndarray, history: list[np.ndarray], count: int) -> np.ndarray:
    """Previous-command input: for k = 1..count, the command sent k actions ago minus the current joints, normalised
    act units; zeros where no command has been sent yet. Mirrors ``CaptureSource.state`` (learning/pi05.py)."""
    from so_arm101_v2.learning.tiny_model import normalize_act
    current_norm = normalize_act(np.asarray(current_act, np.float32))
    out = np.zeros((count, 6), dtype=np.float32)
    for k in range(1, min(count, len(history)) + 1):
        out[k - 1] = normalize_act(np.asarray(history[-k], np.float32)) - current_norm
    return out.reshape(-1)


def residual_to_commands(current_act: np.ndarray, residual: np.ndarray, clamp_channels: tuple[int, ...] = (5,)) -> np.ndarray:
    """(H, 6) residual in normalised units -> (H, 6) ACT commands, gripper clamped to the safe bounds."""
    from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act
    current_norm = normalize_act(np.asarray(current_act, np.float32))
    commands = denormalize_act((current_norm[None, :] + np.asarray(residual, np.float32)).astype(np.float32)).astype(np.float32)
    low, high = effective_safe_act_bounds()
    for channel in clamp_channels:
        commands[:, channel] = np.clip(commands[:, channel], low[channel], high[channel])
    return commands


class ResidualChunkPolicy:
    requires_pixels = True

    def __init__(self, model: Any, *, execute: int = 90, clamp_channels: tuple[int, ...] = (5,), noise_seed: int = 0,
                 num_steps: int | None = None, tag: str | None = None) -> None:
        if not 1 <= execute <= 90:
            raise ValueError("execute must be in [1, 90]")
        self.model, self.execute = model, int(execute)
        self.clamp_channels = tuple(int(c) for c in clamp_channels)
        self.noise_seed, self.num_steps = int(noise_seed), num_steps
        self.tag = tag or model.kind
        self.chunk_horizon = self.execute
        self.teacher_horizon = TEACHER_HORIZON
        self.report_content_sha256 = None
        self.action_index = 0
        self.inferences = 0
        self._buffer: list[np.ndarray] = []
        self.prev_commands = int(getattr(getattr(model, "config", None), "extra", {}).get("prev_commands", 0))
        self._history: list[np.ndarray] = []

    @property
    def policy_id(self) -> str:
        return f"{self.tag}.exec{self.execute}" + (".gripper_clamp_v1" if self.clamp_channels == (5,) else "")

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._buffer = []
        self._history = []

    @property
    def needs_frame(self) -> bool:
        return not self._buffer

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del adapter
        if not self._buffer:
            state = chunk_state(current_act, self.action_index)
            if self.prev_commands:
                state = np.concatenate([state, lead_features(current_act, self._history, self.prev_commands)])
            state = state[None, :]
            residual = self.model.sample(np.asarray(image, np.uint8)[None], state, seed=self.noise_seed + self.action_index,
                                         num_steps=self.num_steps)[0]
            self._buffer = list(residual_to_commands(current_act, residual[: self.execute], self.clamp_channels))
            self.inferences += 1
        self.action_index += 1
        command = np.asarray(self._buffer.pop(0), dtype=np.float32)
        if self.prev_commands:
            self._history = (self._history + [command])[-self.prev_commands:]
        return command


__all__ = ["ResidualChunkPolicy", "TEACHER_HORIZON", "chunk_state", "lead_features", "residual_to_commands"]
