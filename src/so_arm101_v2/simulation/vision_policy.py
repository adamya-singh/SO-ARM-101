"""Closed-loop policy for vision-conditioned chunked clones.

Consumes the live wrist frame at each chunk boundary (``requires_pixels``
makes the rollout render and pass raw HWC uint8), plus proprioception and
the open-loop progress clock. No privileged simulator state is read.
CPU inference — the evaluation lane stays CPU-pinned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds
from so_arm101_v2.data import preprocess_wrist_image
from so_arm101_v2.data._serialization import content_sha256


class VisionChunkedPolicy:
    """Execute a vision chunked clone: one wrist frame per H actions."""

    requires_pixels = True

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        black_image: bool = False,
        clamp_channels: tuple[int, ...] = (),
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("vision inference requires the 'learn' extra") from exc
        from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act
        from so_arm101_v2.learning.vision import build_vision_chunked_model

        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported vision clone checkpoint schema")
        self.chunk_horizon = int(payload["chunk_horizon"])
        if payload.get("model_kind") != f"vision_h{self.chunk_horizon}":
            raise ValueError("vision clone checkpoint kind metadata is malformed")
        self.hidden_width = int(payload["hidden_width"])
        self.encoder = str(payload.get("encoder", "v1"))
        self.teacher_horizon = int(payload["teacher_horizon"])
        self.config_seed = int(payload.get("config", {}).get("seed", 101))
        if self.teacher_horizon not in (450, 480):
            raise ValueError("vision clone checkpoint metadata is malformed")
        report_path = checkpoint_path.with_name("report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stated = report.pop("content_sha256")
        if content_sha256(report) != stated:
            raise ValueError("vision clone report content hash mismatch")
        if stated != payload.get("report_content_sha256"):
            raise ValueError("vision clone checkpoint and report disagree")
        self.report_content_sha256 = stated
        self.model = build_vision_chunked_model(self.hidden_width, self.chunk_horizon, encoder=self.encoder)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.torch = torch
        self._normalize_act = normalize_act
        self._denormalize_act = denormalize_act
        self.action_index = 0
        self._buffer: list[np.ndarray] = []
        # Ablation lever: zero the frame at inference to measure how much
        # the policy actually relies on pixels.  Skips rendering entirely.
        self.black_image = bool(black_image)
        if self.black_image:
            self.requires_pixels = False
        # Same clamp semantics as ChunkedClonePolicy (gripper-only registered
        # scope; see notes/gripper-clamp-proposal.md).
        self.clamp_channels = tuple(int(channel) for channel in clamp_channels)
        if any(not 0 <= channel < 6 for channel in self.clamp_channels):
            raise ValueError("clamp_channels must be joint indices in [0, 6)")
        if self.clamp_channels:
            low, high = effective_safe_act_bounds()
            self._clamp_low = low
            self._clamp_high = high

    @property
    def policy_id(self) -> str:
        base = f"vision_h{self.chunk_horizon}.seed{self.config_seed}"
        if self.black_image:
            base = f"{base}.black"
        if self.clamp_channels == (5,):
            return f"{base}.gripper_clamp_v1"
        if self.clamp_channels:
            channels = "_".join(str(channel) for channel in self.clamp_channels)
            return f"{base}.clamp{channels}_v1"
        return base

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._buffer = []

    @property
    def needs_frame(self) -> bool:
        """True when the next ``predict`` will run the network and therefore needs a real frame."""
        return not self._buffer

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del adapter
        if not self._buffer:
            current = np.asarray(current_act, dtype=np.float32)
            progress = np.float32(
                min(self.action_index, self.teacher_horizon - 1)
                / (self.teacher_horizon - 1)
            )
            state = np.concatenate(
                [self._normalize_act(current), np.asarray([progress], dtype=np.float32)]
            )[None, :]
            if self.black_image:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            chw = preprocess_wrist_image(np.asarray(image, dtype=np.uint8))
            current_norm = self._normalize_act(current)
            with self.torch.inference_mode():
                residual = self.model(
                    self.torch.from_numpy(np.array(chw, copy=True)[None, :]),
                    self.torch.from_numpy(state),
                ).numpy()[0]
            chunk_norm = current_norm[None, :] + residual.reshape(self.chunk_horizon, 6)
            commands = self._denormalize_act(chunk_norm.astype(np.float32))
            self._buffer = [np.asarray(row, dtype=np.float32) for row in commands]
        self.action_index += 1
        command = self._buffer.pop(0)
        for channel in self.clamp_channels:
            command[channel] = np.clip(
                command[channel], self._clamp_low[channel], self._clamp_high[channel]
            )
        return command


__all__ = ["VisionChunkedPolicy"]
