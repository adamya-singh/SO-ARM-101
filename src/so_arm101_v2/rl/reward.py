"""Chunk reward for the bench pick-and-replace task (retry tranche, 2026-10-04).

Terminal: +1 on contract success, -1 on a safety invalidation (any clip, delta limit, nonfinite command or
unsafe contact invalidates the episode under the contract), 0 on a timeout.

Shaping: potential-based, F = Phi(end of chunk) - Phi(start of chunk), so over an episode it sums to
Phi(end) - Phi(start) and cannot change which policy is optimal (Ng et al. 1999). Phi is at most 0.3 and is
built only from milestones that need contact with the cube:

- 0.10 once a strict grasp has been acquired (contract event, latched),
- 0.10 once the pickup is complete (lifted and held, latched),
- 0.10 x closeness of the cube to the square (1 - distance / 10 cm; distance = horizontal offset from the
  square's centre combined with the height of the cube's lowest corner above the square), after the pickup.

Nothing rewards approaching or aligning above the cube. The April 2026 reward (notes/reward-redesign-
contact-grasp-lift.md) paid for alignment and PPO found the hover: high alignment reward, zero contact. Here a
hovering policy collects 0 and a success collects 1 + (Phi_end - Phi_start) >= 0.7.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    success: float = 1.0
    invalidation: float = -1.0
    grasp_bonus: float = 0.10
    pickup_bonus: float = 0.10
    carry_bonus: float = 0.10
    carry_scale_m: float = 0.10
    version: str = "bench_reward_v1"

    def __post_init__(self) -> None:
        if self.grasp_bonus + self.pickup_bonus + self.carry_bonus > 0.5 * self.success:
            raise ValueError("shaping must stay well below the success reward (anti-hover)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def potential(config: RewardConfig, *, strict_grasp_acquired: bool, pickup_completed: bool, place_distance_m: float) -> float:
    phi = 0.0
    if strict_grasp_acquired:
        phi += config.grasp_bonus
    if pickup_completed:
        phi += config.pickup_bonus
        closeness = 1.0 - min(max(float(place_distance_m), 0.0) / config.carry_scale_m, 1.0)
        phi += config.carry_bonus * closeness
    return phi


def chunk_reward(config: RewardConfig, *, phi_start: float, phi_end: float, success: bool, invalidated: bool) -> dict[str, float]:
    """Reward parts of one chunk; ``total`` is what PPO sees."""
    terminal = config.success if success else (config.invalidation if invalidated else 0.0)
    shaping = phi_end - phi_start
    return dict(total=terminal + shaping, success=config.success if success else 0.0,
                safety=config.invalidation if invalidated else 0.0, shaping=shaping)


__all__ = ["RewardConfig", "chunk_reward", "potential"]
