"""Closed-loop policy for a pi0.5 residual-chunk fine-tune (``learning/pi05.py``).

Same contract as ``VisionChunkedPolicy``: the live 256x256 wrist frame, ``normalize_act(current)``
and the open-loop progress clock go in; a 90-step residual chunk comes out. ``execute`` sets how many
of its commands run before the next frame is read (90 = the CNN runs' schedule; 30 = replan at every
row the training captures stored a frame for). GPU inference: run rollouts with ``workers=1`` and a
callable factory that shares one loaded model (a 3B model per spawn worker does not fit the card).

Servo lag. A chunk is added to the measured joints, as in training, where the simulated joints follow a
command about 3.5 steps behind. A real elbow follows 6-13 steps behind, and a re-read in the middle of a move
restarts the plan from that lagging joint: the lost ground stays lost (physical/episode_20, 2026-10-04: the
survey move stopped 13 elbow units short). Two ways to deal with it here:

- ``execute=90`` (or ``first_execute=90`` for the survey move only): re-read only where the teacher's plan has
  the arm nearly at rest, as the CNN runs did.
- ``servo_response``: lag compensation. Two first-order servo models follow the commands sent so far: the
  simulated servo (``SIM_SERVO_RESPONSE`` per step) and the real one (the same servo behind a per-joint
  first-order response, the form ``tools/fit_servo_lag.py`` fits). The chunk is added to
  ``measured + (simulated - real)``: the measured pose with the modelled extra lag put back. Both models settle
  on the command, so a resting offset (gravity sag) stays in the anchor, as it was in training; an anchor that
  drops it and uses the command or a servo model alone adds the sag again at every re-read (tried 2026-10-04:
  0/10 on the certified scene). The correction never exceeds ``anchor_lead_cap_act`` and the gripper is never
  corrected (it is meant to stall on the cube). A response of 1.0 for a joint means no correction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds

# Least-squares first-order response of the certified scene's arm joints (ACT space, per 30 Hz step) over the 30
# held-out exec-30 rollouts of pi05_lora16_10k_20261002: 0.26, 0.27, 0.26, 0.29, 0.30 for joints 0-4.
SIM_SERVO_RESPONSE = 0.27
ANCHOR_LEAD_CAP_ACT = 12.0 * np.pi / 100.0     # 12 physical units; the approach phase caps its lead at 10


class Pi05ChunkPolicy:
    requires_pixels = True

    def __init__(self, policy: Any, config: Any, stats: Any, prompt: Any, *, execute: int = 90,
                 clamp_channels: tuple[int, ...] = (5,), noise_seed: int = 0, tag: str = "pi05",
                 first_execute: int | None = None, servo_response: tuple[float, ...] | None = None,
                 anchor_lead_cap_act: float = ANCHOR_LEAD_CAP_ACT) -> None:
        from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act

        if not 1 <= execute <= config.chunk_horizon:
            raise ValueError("execute must be in [1, chunk_horizon]")
        self.policy, self.config, self.stats, self.prompt = policy, config, stats, prompt
        if servo_response is not None:
            servo_response = tuple(float(v) for v in servo_response)
            if len(servo_response) != 6 or any(not 0.0 < v <= 1.0 for v in servo_response):
                raise ValueError("servo_response must be six values in (0, 1]")
        if first_execute is not None and not 1 <= first_execute <= config.chunk_horizon:
            raise ValueError("first_execute must be in [1, chunk_horizon]")
        self.execute = int(execute)
        self.servo_response = None if servo_response is None else np.asarray(servo_response, dtype=np.float64)
        self.first_execute = None if first_execute is None else int(first_execute)
        self.anchor_lead_cap_act = float(anchor_lead_cap_act)
        self._models: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None    # simulated servo, real servo, its filtered command
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
        return (f"{self.tag}.exec{self.execute}" + ("" if self.first_execute is None else f".first{self.first_execute}")
                + ("" if self.servo_response is None else ".lagcomp")
                + (".gripper_clamp_v1" if self.clamp_channels == (5,) else ""))

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._buffer = []
        self._models = None

    def _anchor(self, measured: np.ndarray) -> np.ndarray:
        if self._models is None:
            return measured
        simulated, real, _ = self._models
        correction = np.clip(simulated - real, -self.anchor_lead_cap_act, self.anchor_lead_cap_act)
        correction[5] = 0.0
        return (measured + correction).astype(np.float32)

    @property
    def needs_frame(self) -> bool:
        return not self._buffer

    def predict(self, image: np.ndarray, current_act: np.ndarray, adapter: Any | None = None) -> np.ndarray:
        del adapter
        if not self._buffer:
            from so_arm101_v2.learning.pi05 import prepare_images, sample_residual

            current = self._anchor(np.asarray(current_act, dtype=np.float32))
            current_norm = self._normalize_act(current)
            progress = np.float32(min(self.action_index, self.teacher_horizon - 1) / (self.teacher_horizon - 1))
            state = np.concatenate([current_norm, [progress]]).astype(np.float32)[None, :]
            tokens, masks = self.prompt(self.stats.norm_state(state), self.device)
            images = prepare_images(np.asarray(image, dtype=np.uint8)[None], self.device)
            residual = sample_residual(self.policy, self.stats, images, tokens, masks,
                                       seed=self.noise_seed + self.action_index).cpu().numpy()[0]
            execute = self.first_execute if self.action_index == 0 and self.first_execute is not None else self.execute
            commands = self._denormalize_act((current_norm[None, :] + residual[:execute]).astype(np.float32))
            self._buffer = [np.asarray(row, dtype=np.float32) for row in commands]
            self.inferences += 1
        self.action_index += 1
        command = self._buffer.pop(0)
        for channel in self.clamp_channels:
            command[channel] = np.clip(command[channel], self._clamp_low[channel], self._clamp_high[channel])
        if self.servo_response is not None:
            if self._models is None:
                start = np.asarray(current_act, dtype=np.float64)
                self._models = (start.copy(), start.copy(), start.copy())
            simulated, real, filtered = self._models
            target = command.astype(np.float64)
            simulated += SIM_SERVO_RESPONSE * (target - simulated)
            filtered += self.servo_response * (target - filtered)
            real += SIM_SERVO_RESPONSE * (filtered - real)
        return command


__all__ = ["ANCHOR_LEAD_CAP_ACT", "Pi05ChunkPolicy", "SIM_SERVO_RESPONSE"]
