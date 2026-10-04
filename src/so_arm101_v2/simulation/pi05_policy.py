"""Closed-loop policy for a pi0.5 residual-chunk fine-tune (``learning/pi05.py``).

Same contract as ``VisionChunkedPolicy``: the live 256x256 wrist frame, ``normalize_act(current)``
and the open-loop progress clock go in; a 90-step residual chunk comes out. ``execute`` sets how many
of its commands run before the next frame is read (90 = the CNN runs' schedule; 30 = replan at every
row the training captures stored a frame for). GPU inference: run rollouts with ``workers=1`` and a
callable factory that shares one loaded model (a 3B model per spawn worker does not fit the card).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds


class Pi05ChunkPolicy:
    requires_pixels = True

    def __init__(self, policy: Any, config: Any, stats: Any, prompt: Any, *, execute: int = 90,
                 clamp_channels: tuple[int, ...] = (5,), noise_seed: int = 0, tag: str = "pi05") -> None:
        from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act

        if not 1 <= execute <= config.chunk_horizon:
            raise ValueError("execute must be in [1, chunk_horizon]")
        self.policy, self.config, self.stats, self.prompt = policy, config, stats, prompt
        self.execute = int(execute)
        self.chunk_horizon = self.execute          # commands per camera read (what the physical dry-pass gate iterates)
        self.report_content_sha256 = None
        self.teacher_horizon = 480
        self.noise_seed = int(noise_seed)
        self.tag = tag
        self._normalize_act, self._denormalize_act = normalize_act, denormalize_act
        self.clamp_channels = tuple(int(c) for c in clamp_channels)
        low, high = effective_safe_act_bounds()
        self._clamp_low, self._clamp_high = low, high
        self.device = next(policy.parameters()).device
        self.action_index = 0
        self.inferences = 0
        self._buffer: list[np.ndarray] = []

    @property
    def policy_id(self) -> str:
        return f"{self.tag}.exec{self.execute}" + (".gripper_clamp_v1" if self.clamp_channels == (5,) else "")

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._buffer = []

    @property
    def needs_frame(self) -> bool:
        return not self._buffer

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del adapter
        if not self._buffer:
            from so_arm101_v2.learning.pi05 import prepare_images, sample_residual

            current = np.asarray(current_act, dtype=np.float32)
            current_norm = self._normalize_act(current)
            progress = np.float32(min(self.action_index, self.teacher_horizon - 1) / (self.teacher_horizon - 1))
            state = np.concatenate([current_norm, [progress]]).astype(np.float32)[None, :]
            tokens, masks = self.prompt(self.stats.norm_state(state), self.device)
            images = prepare_images(np.asarray(image, dtype=np.uint8)[None], self.device)
            residual = sample_residual(self.policy, self.stats, images, tokens, masks,
                                       seed=self.noise_seed + self.action_index).cpu().numpy()[0]
            commands = self._denormalize_act((current_norm[None, :] + residual[: self.execute]).astype(np.float32))
            self._buffer = [np.asarray(row, dtype=np.float32) for row in commands]
            self.inferences += 1
        self.action_index += 1
        command = self._buffer.pop(0)
        for channel in self.clamp_channels:
            command[channel] = np.clip(command[channel], self._clamp_low[channel], self._clamp_high[channel])
        return command


__all__ = ["Pi05ChunkPolicy"]
