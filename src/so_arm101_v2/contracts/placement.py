"""Placement randomization regime: where the white square (and the cube on it) may lie on the bench.

Why (2026-09-10): the first live success (episode 10) was followed by a miss
(episode 13) with the cube ~20 mm from the task pose; the policy had trained on
cube offsets of +-10 mm around one fixed square. The user's direction: square
and cube move together, placed at random inside a rectangle 14 in wide (x,
side to side) by 10 in deep (y, forward) whose near edge is 2 in forward of
the base front edge, with the square yawed up to +-45 deg and the cube keeping
its small jitter on the square.

This module is numpy only (like ``contracts.appearance``): the regime is the
hashed block stored as ``BenchConfig.placement``; ``placement_draws`` is the
deterministic stream of (x, y, yaw) draws (stream 2, separate from the +-10 mm
cube-offset stream and the appearance streams 0/1); ``yaw_quaternion`` is the
cube/square orientation. Reachability is the teacher's business: suite
generation screens every draw with a full teacher episode and records the
rejections, so the effective training region is the reachable part of the
rectangle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

PLACEMENT_RESOLVER_VERSION = "bench_placement_v1"
_SALT = 0x5C0A3E
INCH = 0.0254


def _pair(value, name: str) -> tuple[float, float]:
    lo, hi = float(value[0]), float(value[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"placement regime {name} must be a finite (low, high) pair")
    return lo, hi


@dataclass(frozen=True)
class PlacementRegime:
    """The user's rectangle in the bench frame (x lateral, y forward from the base front edge)."""

    name: str = "bench_placement"
    version: int = 1
    x_range_m: tuple[float, float] = (-7 * INCH, 7 * INCH)                    # 14 in side to side
    y_range_m: tuple[float, float] = (0.0646353 + 2 * INCH, 0.0646353 + 12 * INCH)   # near edge 2 in from the base front edge, 10 in deep
    base_gap_m: float = 2 * INCH
    depth_m: float = 10 * INCH
    cube_offset_m: float = 0.010          # cube jitter about the square centre (the existing +-10 mm)
    yaw_range_deg: float = 45.0           # square (and cube) yaw drawn uniformly in [-yaw, +yaw]
    survey_visibility_margin_px: float = 20.0

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported placement regime version")
        object.__setattr__(self, "x_range_m", _pair(self.x_range_m, "x_range_m"))
        object.__setattr__(self, "y_range_m", _pair(self.y_range_m, "y_range_m"))
        for field_name in ("base_gap_m", "depth_m", "cube_offset_m", "yaw_range_deg", "survey_visibility_margin_px"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"placement regime {field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        if abs((self.y_range_m[1] - self.y_range_m[0]) - self.depth_m) > 1e-9:
            raise ValueError("placement regime depth_m must equal the y range extent")
        if self.cube_offset_m > 0.02 or self.yaw_range_deg > 90.0:
            raise ValueError("placement regime cube offset must stay within 20 mm and yaw within 90 degrees")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlacementRegime":
        return cls(**dict(raw))

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    def validate_against(self, base_front_edge_y_m: float, nominal_square_xy) -> None:
        """The near edge must sit base_gap_m beyond the base front edge and the nominal square inside the rectangle."""
        if abs(self.y_range_m[0] - (float(base_front_edge_y_m) + self.base_gap_m)) > 1e-9:
            raise ValueError("placement rectangle near edge must be base_gap_m forward of the base front edge")
        if not self.contains(nominal_square_xy):
            raise ValueError("the nominal square must lie inside the placement rectangle")

    def contains(self, xy) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return self.x_range_m[0] <= x <= self.x_range_m[1] and self.y_range_m[0] <= y <= self.y_range_m[1]

    def distance_to_edge_mm(self, xy) -> float:
        """Signed distance to the rectangle boundary in mm: positive inside, negative outside."""
        x, y = float(xy[0]), float(xy[1])
        dx = min(x - self.x_range_m[0], self.x_range_m[1] - x)
        dy = min(y - self.y_range_m[0], self.y_range_m[1] - y)
        if dx >= 0 and dy >= 0:
            return 1000.0 * min(dx, dy)
        return -1000.0 * float(np.hypot(min(dx, 0.0), min(dy, 0.0)))


def placement_draws(seed: int, count: int, regime: PlacementRegime, *, stream: int = 2) -> tuple[tuple[float, float, float], ...]:
    """``count`` uniform (x, y, yaw_rad) draws over the rectangle; a pure function of (seed, stream, regime)."""
    rng = np.random.default_rng([int(seed), _SALT, int(stream), regime.version])
    xs = rng.uniform(regime.x_range_m[0], regime.x_range_m[1], size=int(count))
    ys = rng.uniform(regime.y_range_m[0], regime.y_range_m[1], size=int(count))
    yaws = np.deg2rad(rng.uniform(-regime.yaw_range_deg, regime.yaw_range_deg, size=int(count)))
    return tuple((float(x), float(y), float(w)) for x, y, w in zip(xs, ys, yaws))


def yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    """MuJoCo (w, x, y, z) quaternion for a rotation of ``yaw_rad`` about +z."""
    half = 0.5 * float(yaw_rad)
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def fold_cube_yaw(yaw_rad: float) -> float:
    """The cube is 4-fold symmetric about z: the equivalent grasp yaw in (-pi/4, pi/4]."""
    quarter = np.pi / 2
    folded = (float(yaw_rad) + quarter / 2) % quarter - quarter / 2
    return float(folded)


__all__ = ["INCH", "PLACEMENT_RESOLVER_VERSION", "PlacementRegime", "fold_cube_yaw", "placement_draws", "yaw_quaternion"]
