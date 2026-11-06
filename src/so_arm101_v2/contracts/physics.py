"""Physics randomization regimes for the bench task (2026-11-05, notes/handoff-20261105.md step 2).

A scenario names a regime (``physics_regime``) and carries a ``physics_seed``; ``resolve_physics(regime, seed)``
is a pure function of the two. Nothing here touches the scene or its hash: the draw scales the scene's own
values at reset (simulation/adapter.py), so on the measured-servo scene (bench_pick_replace_v1_servo) the
servo multipliers are centred on the October servo model (servo_model_20261004) and on the nominal scene on
kp 50 / frictionloss 0.1.

Gravity sag has no parameter of its own: the resting error is (gravity torque + friction) / kp, so it is
randomized through kp, friction and the arm's link masses.

Every draw that varies during an episode is keyed on (seed, index): the inference pause on the chunk boundary,
the sensor noise on the control step. Nothing depends on how many renders or calls happened, which keeps
captures that skip renders (oracle.py) and parallel captures byte-identical.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

PHYSICS_RESOLVER_VERSION = "bench_physics_v1"
_SALT = 0x9A51C5
_EPISODE, _PAUSE, _NOISE = 0, 1, 2

# Physical normalized units (the servo's -100..100 for joints, 0..100 for the gripper) per act unit.
_UNITS_PER_ACT = np.array([200.0 / (2 * np.pi)] * 5 + [100.0 / 1.7], dtype=np.float64)


def _range(pair: Any, name: str, *, positive: bool = False) -> tuple[float, float]:
    low, high = (float(v) for v in pair)
    if not (np.isfinite(low) and np.isfinite(high)) or low > high or low < 0 or (positive and low <= 0):
        raise ValueError(f"physics regime {name} must be a finite {'positive ' if positive else ''}range low <= high, got {pair!r}")
    return low, high


@dataclass(frozen=True)
class PhysicsRegime:
    """Ranges of one versioned physics randomization; a range with low == high is a fixed value."""

    name: str
    version: int = 1
    # log-uniform, per joint, times the scene's kp. Low end 0.7, not 0.5: the 2026-11-05 feasibility scan failed 70% of
    # draws in the softest elbow quintile (the fitted real elbow, kp 6.5, is already the softest joint by far).
    kp_scale: tuple[float, float] = (0.7, 2.0)
    kv_scale: tuple[float, float] = (0.5, 2.0)          # log-uniform, per joint, times the scene's kv
    friction_scale: tuple[float, float] = (0.5, 2.0)    # log-uniform, per joint, times the scene's frictionloss
    arm_mass_scale: tuple[float, float] = (0.8, 1.25)   # uniform, one factor for every arm link (gravity sag)
    latency_ms: tuple[float, float] = (0.0, 100.0)      # uniform, command delay to the actuators
    pause_ms: tuple[float, float] = (0.0, 1000.0)       # uniform, drawn per chunk boundary: physics runs, command held
    sensor_noise_units: float = 0.3                     # Gaussian sd per joint per step on the observed joint state
    cube_mass_kg: tuple[float, float] = (0.005, 0.06)   # log-uniform; the printed cube weighs a few grams
    cube_friction: tuple[float, float] = (0.5, 1.5)     # uniform, sliding friction of the cube's geoms

    def __post_init__(self) -> None:
        if not self.name or self.version != 1:
            raise ValueError("physics regime needs a name and version 1")
        for field in ("kp_scale", "kv_scale", "friction_scale", "arm_mass_scale", "cube_mass_kg"):
            object.__setattr__(self, field, _range(getattr(self, field), field, positive=True))
        for field in ("latency_ms", "pause_ms", "cube_friction"):
            object.__setattr__(self, field, _range(getattr(self, field), field))
        if self.latency_ms[1] > 500.0 or self.pause_ms[1] > 5000.0:
            raise ValueError("physics regime latency/pause out of the plausible range")
        noise = float(self.sensor_noise_units)
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("sensor_noise_units must be a finite non-negative number")
        object.__setattr__(self, "sensor_noise_units", noise)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PhysicsRegime":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})

    def identity(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicsParams:
    """One episode's draw."""

    regime: str
    seed: int
    kp_scale: tuple[float, ...]
    kv_scale: tuple[float, ...]
    friction_scale: tuple[float, ...]
    arm_mass_scale: float
    latency_s: float
    cube_mass_kg: float
    cube_friction: float
    pause_ms: tuple[float, float]
    sensor_noise_units: float

    def as_record(self) -> dict[str, Any]:
        return asdict(self)

    def pause_s(self, boundary: int) -> float:
        """Inference pause at chunk boundary ``boundary`` (the action index where the policy reads a frame)."""
        low, high = self.pause_ms
        if high <= 0.0:
            return 0.0
        rng = np.random.default_rng([int(self.seed), _SALT, _PAUSE, int(boundary)])
        return float(rng.uniform(low, high)) / 1000.0

    def sensor_noise_act(self, step: int) -> np.ndarray:
        """Additive noise on the observed joint state at control step ``step``, in act units."""
        if self.sensor_noise_units <= 0.0:
            return np.zeros(6, dtype=np.float32)
        rng = np.random.default_rng([int(self.seed), _SALT, _NOISE, int(step)])
        units = rng.normal(0.0, self.sensor_noise_units, size=6)
        return (units / _UNITS_PER_ACT).astype(np.float32)


def _log_uniform(rng: np.random.Generator, pair: tuple[float, float], size: int | None = None) -> Any:
    low, high = pair
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def resolve_physics(regime: PhysicsRegime, seed: int) -> PhysicsParams:
    """Deterministic draw of one episode's physics; a pure function of ``(regime, seed)``."""
    seed = int(seed)
    if not 0 <= seed < 2 ** 31:
        raise ValueError("physics seed must be in [0, 2**31)")
    rng = np.random.default_rng([seed, _SALT, _EPISODE, regime.version])
    kp = _log_uniform(rng, regime.kp_scale, 6)
    kv = _log_uniform(rng, regime.kv_scale, 6)
    friction = _log_uniform(rng, regime.friction_scale, 6)
    mass = float(rng.uniform(*regime.arm_mass_scale))
    latency = float(rng.uniform(*regime.latency_ms)) / 1000.0
    cube_mass = float(_log_uniform(rng, regime.cube_mass_kg))
    cube_friction = float(rng.uniform(*regime.cube_friction))
    return PhysicsParams(
        regime=regime.name, seed=seed,
        kp_scale=tuple(float(v) for v in kp), kv_scale=tuple(float(v) for v in kv),
        friction_scale=tuple(float(v) for v in friction), arm_mass_scale=mass, latency_s=latency,
        cube_mass_kg=cube_mass, cube_friction=cube_friction, pause_ms=regime.pause_ms,
        sensor_noise_units=regime.sensor_noise_units,
    )


def physics_seeds(seed: int, count: int, *, stream: int = 4) -> tuple[int, ...]:
    """``count`` per-scenario physics seeds from a stream separate from the pose, appearance and teacher streams."""
    rng = np.random.default_rng([int(seed), _SALT, int(stream)])
    return tuple(int(v) for v in rng.integers(0, 2 ** 31, size=int(count)))


# Named regimes. Adding one is safe; changing one forks every suite and capture that names it (bump the name).
REGIMES: dict[str, PhysicsRegime] = {
    # Training: wide ranges centred on the scene's servo model (handoff step 2; pause range chosen by the user 2026-11-05).
    "bench_physics_v1": PhysicsRegime(name="bench_physics_v1"),
    # Evaluation stand-in for the real arm running a fast policy (CNN / ACT infer in milliseconds): the scene's servo
    # model as measured, 50 ms command latency, no inference pause, sensor noise on, a printed few-gram cube.
    "real_like_v1": PhysicsRegime(
        name="real_like_v1", kp_scale=(1.0, 1.0), kv_scale=(1.0, 1.0), friction_scale=(1.0, 1.0),
        arm_mass_scale=(1.0, 1.0), latency_ms=(50.0, 50.0), pause_ms=(0.0, 0.0), sensor_noise_units=0.3,
        cube_mass_kg=(0.008, 0.008), cube_friction=(1.0, 1.0)),
}


def physics_regime(name: str) -> PhysicsRegime:
    try:
        return REGIMES[name]
    except KeyError:
        raise ValueError(f"unknown physics regime {name!r}; known: {sorted(REGIMES)}") from None


__all__ = ["PHYSICS_RESOLVER_VERSION", "PhysicsParams", "PhysicsRegime", "REGIMES", "physics_regime", "physics_seeds",
           "resolve_physics"]
