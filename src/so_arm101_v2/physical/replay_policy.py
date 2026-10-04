"""Replay a recorded command sequence through the physical runner (no network).

``ReplayPolicy`` returns the ``executed_act`` rows of a simulator telemetry file one per step: the way to run the
privileged teacher's own commands on the real arm, so the measured joints can be compared with the joints the
simulator predicted for the same commands. It asks for a camera frame at step 0 and at the survey read
(``bench.observation_steps``) only, so the guards see the survey frame; the commands do not depend on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ReplayPolicy:
    requires_pixels = False
    report_content_sha256 = None

    def __init__(self, telemetry_path: str | Path, bench: Any) -> None:
        payload = json.loads(Path(telemetry_path).read_text())
        self.commands = np.asarray([row["executed_act"] for row in payload["rows"]], dtype=np.float32)
        if self.commands.ndim != 2 or self.commands.shape[1] != 6 or not np.isfinite(self.commands).all():
            raise ValueError("telemetry rows must carry finite six-joint executed_act commands")
        self.frame_steps = (0, int(bench.observation_steps))
        self.chunk_horizon = int(bench.observation_steps)
        self.teacher_horizon = len(self.commands)
        self.policy_id = f"replay.{Path(telemetry_path).stem}"
        self.action_index = 0

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0

    @property
    def needs_frame(self) -> bool:
        return self.action_index in self.frame_steps

    def predict(self, image: Any, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del image, current_act, adapter
        command = self.commands[min(self.action_index, len(self.commands) - 1)].copy()
        self.action_index += 1
        return command


__all__ = ["ReplayPolicy"]
