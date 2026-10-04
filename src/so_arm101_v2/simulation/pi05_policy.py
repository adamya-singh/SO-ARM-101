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
  ``servo_response="adaptive"`` measures the response during the episode instead (``estimate_servo_response``):
  the elbow was about three times slower on 2026-10-04 than on 2026-09-10, so a fixed value is wrong on one
  day or the other. A joint is corrected only when it has moved enough to judge and reads clearly slower than
  the simulator's; otherwise the chunk is added to the measured pose, exactly as without the option.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds

# Least-squares first-order response of the certified scene's arm joints (ACT space, per 30 Hz step) over the 30
# held-out exec-30 rollouts of pi05_lora16_10k_20261002: 0.26, 0.27, 0.26, 0.29, 0.30 for joints 0-4.
SIM_SERVO_RESPONSE = 0.27
ANCHOR_LEAD_CAP_ACT = 12.0 * np.pi / 100.0     # 12 physical units; the approach phase caps its lead at 10
RESPONSE_GRID = np.array([0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.32, 0.40, 0.50, 0.65, 0.80, 1.0])
# Estimates on the certified scene are 0.65-0.8 for the joints that move (the simulated servo is not exactly first
# order) and 0.5-0.65 on the real arm of 2026-09-10; the slow elbow of 2026-10-04 reads 0.20-0.25. At or below this
# value a joint is treated as slower than the simulator's.
SLOW_RESPONSE = 0.35
EXCITATION_ACT = 0.2                            # a joint's commands must span this much (6.4 units) before it is judged


def servo_models(commands: np.ndarray, start: float | np.ndarray, response: np.ndarray) -> np.ndarray:
    """The modelled joint value before each command and after the last: shape (len(commands) + 1, *response.shape).

    The simulated servo (first order, ``SIM_SERVO_RESPONSE`` per step) behind a first-order ``response``, started
    at ``start``. ``commands`` is (T,) for one joint with a grid of responses, or (T, J) with one response per joint.
    """
    response = np.asarray(response, dtype=np.float64)
    filtered = np.broadcast_to(np.asarray(start, dtype=np.float64), response.shape).copy()
    value = filtered.copy()
    out = np.empty((len(commands) + 1, *response.shape))
    out[0] = value
    for index, command in enumerate(np.asarray(commands, dtype=np.float64)):
        filtered = filtered + response * (command - filtered)
        value = value + SIM_SERVO_RESPONSE * (filtered - value)
        out[index + 1] = value
    return out


def estimate_servo_response(commands: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Per-joint response of the real servo from an episode so far: 1.0 unless a joint is clearly slow.

    ``commands[t]`` and ``measured[t]`` (the joints read just before command t), ACT space, shape (T, 6). For each
    arm joint whose commands span ``EXCITATION_ACT``, the grid value whose model reproduces the measured step-to-step
    motion best (differences, so a resting offset does not matter). Values above ``SLOW_RESPONSE`` and the gripper
    return 1.0. Checked 2026-10-04 on recorded runs: always 1.0 on the certified scene; 0.16 on the lag stress
    (true 0.16) from step 30; 0.20-0.25 on physical/episode_20 from step 30; 1.0 on episodes 10 and 13.
    """
    commands, measured = np.asarray(commands, dtype=np.float64), np.asarray(measured, dtype=np.float64)
    response = np.ones(6)
    if len(commands) < 8:
        return response
    for joint in range(5):
        c, m = commands[:, joint], measured[:, joint]
        if c.max() - c.min() < EXCITATION_ACT:
            continue
        modelled = servo_models(c[:-1], m[0], RESPONSE_GRID)
        error = ((np.diff(m)[:, None] - np.diff(modelled, axis=0)) ** 2).sum(axis=0)
        best = float(RESPONSE_GRID[int(np.argmin(error))])
        if best <= SLOW_RESPONSE:
            response[joint] = best
    return response


class Pi05ChunkPolicy:
    requires_pixels = True

    def __init__(self, policy: Any, config: Any, stats: Any, prompt: Any, *, execute: int = 90,
                 clamp_channels: tuple[int, ...] = (5,), noise_seed: int = 0, tag: str = "pi05",
                 first_execute: int | None = None, servo_response: tuple[float, ...] | str | None = None,
                 anchor_lead_cap_act: float = ANCHOR_LEAD_CAP_ACT) -> None:
        from so_arm101_v2.learning.tiny_model import denormalize_act, normalize_act

        if not 1 <= execute <= config.chunk_horizon:
            raise ValueError("execute must be in [1, chunk_horizon]")
        self.policy, self.config, self.stats, self.prompt = policy, config, stats, prompt
        self.adaptive = isinstance(servo_response, str)
        if self.adaptive:
            if servo_response != "adaptive":
                raise ValueError('servo_response must be six values in (0, 1], "adaptive" or None')
        elif servo_response is not None:
            servo_response = tuple(float(v) for v in servo_response)
            if len(servo_response) != 6 or any(not 0.0 < v <= 1.0 for v in servo_response):
                raise ValueError("servo_response must be six values in (0, 1]")
        if first_execute is not None and not 1 <= first_execute <= config.chunk_horizon:
            raise ValueError("first_execute must be in [1, chunk_horizon]")
        self.execute = int(execute)
        self.servo_response = None if servo_response is None or self.adaptive else np.asarray(servo_response, dtype=np.float64)
        self.compensate = servo_response is not None
        self.response_estimates: list[dict[str, Any]] = []      # adaptive: (step, response) at every re-read, for the run record
        self.first_execute = None if first_execute is None else int(first_execute)
        self.anchor_lead_cap_act = float(anchor_lead_cap_act)
        self._commands: list[np.ndarray] = []       # this episode's commands and the joints measured just before each
        self._measured: list[np.ndarray] = []
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
                + (".lagadapt" if self.adaptive else "" if self.servo_response is None else ".lagcomp")
                + (".gripper_clamp_v1" if self.clamp_channels == (5,) else ""))

    def reset(self, adapter: Any | None = None) -> None:
        del adapter
        self.action_index = 0
        self._buffer = []
        self._commands, self._measured = [], []
        self.response_estimates = []

    def _anchor(self, measured: np.ndarray) -> np.ndarray:
        if not self.compensate or not self._commands:
            return measured
        commands, history = np.asarray(self._commands), np.asarray(self._measured)
        if self.adaptive:
            response = estimate_servo_response(commands, history)
            self.response_estimates.append(dict(step=self.action_index, response=[round(float(v), 2) for v in response]))
        else:
            response = self.servo_response
        if np.all(response >= 1.0):
            return measured
        simulated = servo_models(commands, history[0], np.ones(6))[-1]
        real = servo_models(commands, history[0], response)[-1]
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
        if self.compensate:
            self._measured.append(np.asarray(current_act, dtype=np.float64).copy())
            self._commands.append(command.astype(np.float64))
        return command


__all__ = ["ANCHOR_LEAD_CAP_ACT", "Pi05ChunkPolicy", "SIM_SERVO_RESPONSE", "SLOW_RESPONSE", "estimate_servo_response", "servo_models"]
