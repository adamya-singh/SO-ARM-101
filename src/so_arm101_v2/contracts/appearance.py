"""Appearance randomization regime for the bench wrist camera: ranges, per-scenario draws, photometric ops.

Why: the first physical trials (2026-09-09) showed the vision policy is brittle
to the exact rendered look it trained on (flat grey background, one light, a
pure-white square). This module defines a *regime* (the ranges, versioned and
hashed into the bench config like the lens block) and a pure-numpy *resolver*
that turns ``(regime, seed)`` into concrete parameters. Applying them to a
MuJoCo model is ``simulation.appearance``; this module has no MuJoCo dependency
so the regime, the draws and the photometric operators are testable and
hashable on their own.

Determinism: every draw comes from ``np.random.default_rng([seed, version,
SALT])`` in a fixed order, so a scenario's look is a pure function of its seed.
The photometric noise is keyed on ``(seed, frame_index)`` so rendering the same
control step twice gives the same pixels regardless of how many other frames
were rendered in between (the evaluation renders every step, the physical
runner's simulator backend only at chunk boundaries).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

APPEARANCE_RESOLVER_VERSION = "bench_appearance_v1"
_SALT = 0xA99EA
_NOISE_TAG = 0x50F7
MATERIAL_NAMES = ("white", "black", "groundplane", "black_pla")


def _pair(value, name: str) -> tuple[float, float]:
    lo, hi = (float(value[0]), float(value[1]))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"appearance regime {name} must be a finite (low, high) pair")
    return lo, hi


@dataclass(frozen=True)
class AppearanceRegime:
    """Ranges the resolver draws from. Widen or narrow in a new version; the identity is hashed."""

    name: str = "bench_appearance"
    version: int = 1
    headlight_ambient: tuple[float, float] = (0.10, 0.55)
    headlight_diffuse: tuple[float, float] = (0.35, 0.85)
    headlight_specular: tuple[float, float] = (0.0, 0.25)
    key_light_cone_deg: float = 40.0
    key_light_diffuse: tuple[float, float] = (0.4, 1.0)
    key_light_ambient: tuple[float, float] = (0.0, 0.15)
    key_light_specular: tuple[float, float] = (0.0, 0.6)
    shadow_probability: float = 0.8
    # Per-material albedo: a grey level from the range times a per-channel factor in
    # [1 - chroma, 1 + chroma] (clipped to [0, 1]), so near-neutral looks like the real bench are
    # common and colour casts still occur.
    arm_albedo: tuple[float, float] = (0.03, 0.20)
    motor_albedo: tuple[float, float] = (0.03, 0.20)
    ground_albedo: tuple[float, float] = (0.005, 0.12)
    cube_albedo: tuple[float, float] = (0.005, 0.12)
    material_chroma: float = 0.4
    paper_chroma: float = 0.12
    material_specular: tuple[float, float] = (0.0, 0.7)
    material_shininess: tuple[float, float] = (0.0, 0.8)
    material_reflectance: tuple[float, float] = (0.0, 0.25)
    ground_reflectance_cap: float = 0.3
    napkin_albedo: tuple[float, float] = (0.65, 1.0)
    towel_probability: float = 0.85
    towel_scale: tuple[float, float] = (0.9, 1.6)
    towel_yaw_deg: float = 10.0
    towel_albedo: tuple[float, float] = (0.6, 1.0)
    ground_texture_probability: float = 0.7
    ground_texture_contrast: tuple[float, float] = (0.0, 0.5)
    ground_texture_repeat_per_m: tuple[float, float] = (5.0, 40.0)   # 256-texel tiles per metre: 20 cm .. 2.5 cm
    skybox_off_probability: float = 0.5
    camera_pos_jitter_m: float = 0.002
    camera_pitch_jitter_deg: float = 4.0
    camera_yaw_roll_jitter_deg: float = 2.0
    camera_fovy_jitter_fraction: float = 0.02
    gain: tuple[float, float] = (0.6, 1.6)
    gamma: tuple[float, float] = (0.7, 1.4)
    channel_balance: float = 0.10
    noise_sigma: tuple[float, float] = (0.0, 2.0)
    blur_sigma_px: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported appearance regime version")
        for field_name in ("headlight_ambient", "headlight_diffuse", "headlight_specular", "key_light_diffuse", "key_light_ambient",
                           "key_light_specular", "arm_albedo", "motor_albedo", "ground_albedo", "cube_albedo", "material_specular",
                           "material_shininess", "material_reflectance", "napkin_albedo", "towel_scale", "towel_albedo",
                           "ground_texture_contrast", "ground_texture_repeat_per_m", "gain", "gamma", "noise_sigma", "blur_sigma_px"):
            object.__setattr__(self, field_name, _pair(getattr(self, field_name), field_name))
        for field_name in ("shadow_probability", "towel_probability", "ground_texture_probability", "skybox_off_probability"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"appearance regime {field_name} must be a probability")
            object.__setattr__(self, field_name, value)
        for field_name in ("key_light_cone_deg", "towel_yaw_deg", "camera_pitch_jitter_deg", "camera_yaw_roll_jitter_deg"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 90.0:
                raise ValueError(f"appearance regime {field_name} must be within [0, 90] degrees")
            object.__setattr__(self, field_name, value)
        if not 0.0 <= float(self.camera_pos_jitter_m) <= 0.02 or not 0.0 <= float(self.camera_fovy_jitter_fraction) <= 0.2:
            raise ValueError("appearance regime camera jitter out of range")
        if not 0.0 <= float(self.channel_balance) <= 0.5 or not 0.0 <= float(self.ground_reflectance_cap) <= 1.0:
            raise ValueError("appearance regime channel balance / reflectance cap out of range")
        if not 0.0 <= float(self.material_chroma) <= 1.0 or not 0.0 <= float(self.paper_chroma) <= 1.0:
            raise ValueError("appearance regime chroma must be within [0, 1]")
        if self.ground_texture_repeat_per_m[0] <= 0:
            raise ValueError("ground texture repeat must be positive")
        if self.gain[0] <= 0 or self.gamma[0] <= 0:
            raise ValueError("gain and gamma must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AppearanceRegime":
        return cls(**dict(raw))

    def identity(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhotometricParams:
    gain: float
    gamma: float
    balance: tuple[float, float, float]
    noise_sigma: float
    blur_sigma_px: float


@dataclass(frozen=True)
class AppearanceParams:
    """Resolved per-scenario look; everything a scene applier needs, recorded in evidence."""

    headlight_ambient: float
    headlight_diffuse: float
    headlight_specular: float
    light_dir: tuple[float, float, float]
    light_diffuse: float
    light_ambient: float
    light_specular: float
    castshadow: bool
    materials: tuple[tuple[str, tuple[float, float, float], float, float, float], ...]   # name, rgb, specular, shininess, reflectance
    napkin_rgb: tuple[float, float, float]
    towel: tuple[float, float, float, tuple[float, float, float]] | None                    # scale_x, scale_y, yaw_rad, rgb
    ground_texture: tuple[int, float, float] | None                                       # (texture seed, contrast, repeat per metre)
    skybox: str | tuple[tuple[float, float, float], tuple[float, float, float]]            # "off" | (top rgb, bottom rgb)
    camera_pos_offset: tuple[float, float, float]
    camera_euler_deg: tuple[float, float, float]                                            # pitch, yaw, roll jitter
    fovy_scale: float
    photometric: PhotometricParams

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def appearance_seeds(seed: int, count: int, *, stream: int = 0) -> tuple[int, ...]:
    """``count`` independent per-scenario seeds from a stream separate from the pose RNG (Python ints)."""
    rng = np.random.default_rng([int(seed), _SALT, int(stream)])
    return tuple(int(v) for v in rng.integers(0, 2 ** 31, size=int(count)))


def resolve_appearance(regime: AppearanceRegime, seed: int) -> AppearanceParams:
    """Deterministic draw of one look from the regime; a pure function of ``(regime, seed)``."""
    seed = int(seed)
    if not 0 <= seed < 2 ** 31:
        raise ValueError("appearance seed must be in [0, 2**31)")
    rng = np.random.default_rng([seed, regime.version, _SALT])
    u = lambda pair: float(rng.uniform(*pair))

    def tinted(pair, chroma):
        # Grey level times per-channel factors; the tint amplitude itself is drawn (squared uniform) so
        # near-neutral looks like the real bench dominate and strong casts remain in the tail.
        grey = rng.uniform(*pair)
        amplitude = chroma * rng.uniform() ** 2
        factors = 1.0 + amplitude * rng.uniform(-1.0, 1.0, size=3)
        return tuple(float(v) for v in np.clip(grey * factors, 0.0, 1.0))
    headlight = (u(regime.headlight_ambient), u(regime.headlight_diffuse), u(regime.headlight_specular))
    # Key light direction: uniform in a cone around straight down.
    cone = np.deg2rad(regime.key_light_cone_deg)
    theta = float(np.arccos(1 - rng.uniform() * (1 - np.cos(cone))))
    phi = float(rng.uniform(0, 2 * np.pi))
    light_dir = (float(np.sin(theta) * np.cos(phi)), float(np.sin(theta) * np.sin(phi)), float(-np.cos(theta)))
    light = (u(regime.key_light_diffuse), u(regime.key_light_ambient), u(regime.key_light_specular), bool(rng.uniform() < regime.shadow_probability))
    materials = []
    for name, albedo in zip(MATERIAL_NAMES, (regime.arm_albedo, regime.motor_albedo, regime.ground_albedo, regime.cube_albedo)):
        rgb = tinted(albedo, regime.material_chroma)
        specular, shininess, reflectance = u(regime.material_specular), u(regime.material_shininess), u(regime.material_reflectance)
        if name == "groundplane":
            reflectance = min(reflectance, regime.ground_reflectance_cap)
        materials.append((name, rgb, specular, shininess, reflectance))
    napkin_rgb = tinted(regime.napkin_albedo, regime.paper_chroma)
    towel = None
    if rng.uniform() < regime.towel_probability:
        scale = tuple(float(v) for v in rng.uniform(regime.towel_scale[0], regime.towel_scale[1], size=2))
        yaw = float(np.deg2rad(rng.uniform(-regime.towel_yaw_deg, regime.towel_yaw_deg)))
        towel_rgb = tinted(regime.towel_albedo, regime.paper_chroma)
        towel = (scale[0], scale[1], yaw, towel_rgb)
    ground_texture = None
    if rng.uniform() < regime.ground_texture_probability:
        ground_texture = (int(rng.integers(0, 2 ** 31)), u(regime.ground_texture_contrast), u(regime.ground_texture_repeat_per_m))
    if rng.uniform() < regime.skybox_off_probability:
        skybox: Any = "off"
    else:
        skybox = (tuple(float(v) for v in rng.uniform(0.0, 1.0, size=3)), tuple(float(v) for v in rng.uniform(0.0, 0.3, size=3)))
    pos = tuple(float(v) for v in rng.uniform(-regime.camera_pos_jitter_m, regime.camera_pos_jitter_m, size=3))
    euler = (float(rng.uniform(-regime.camera_pitch_jitter_deg, regime.camera_pitch_jitter_deg)),
             float(rng.uniform(-regime.camera_yaw_roll_jitter_deg, regime.camera_yaw_roll_jitter_deg)),
             float(rng.uniform(-regime.camera_yaw_roll_jitter_deg, regime.camera_yaw_roll_jitter_deg)))
    fovy_scale = float(rng.uniform(1 - regime.camera_fovy_jitter_fraction, 1 + regime.camera_fovy_jitter_fraction))
    photometric = PhotometricParams(
        gain=u(regime.gain), gamma=u(regime.gamma),
        balance=tuple(float(v) for v in rng.uniform(1 - regime.channel_balance, 1 + regime.channel_balance, size=3)),
        noise_sigma=u(regime.noise_sigma), blur_sigma_px=u(regime.blur_sigma_px),
    )
    return AppearanceParams(
        headlight_ambient=headlight[0], headlight_diffuse=headlight[1], headlight_specular=headlight[2],
        light_dir=light_dir, light_diffuse=light[0], light_ambient=light[1], light_specular=light[2], castshadow=light[3],
        materials=tuple(materials), napkin_rgb=napkin_rgb, towel=towel, ground_texture=ground_texture, skybox=skybox,
        camera_pos_offset=pos, camera_euler_deg=euler, fovy_scale=fovy_scale, photometric=photometric,
    )


# ----------------------------------------------------------------------------- photometric operators

def photometric_lut(params: PhotometricParams) -> np.ndarray:
    """(3, 256) uint8 lookup table for gain, gamma and per-channel balance (float64 math, rounded once)."""
    x = np.arange(256, dtype=np.float64) / 255.0
    table = np.empty((3, 256), dtype=np.uint8)
    for channel in range(3):
        y = np.power(np.clip(x * params.gain * params.balance[channel], 0.0, 1.0), params.gamma)
        table[channel] = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    return table


def _blur_kernel(sigma: float) -> np.ndarray | None:
    if sigma <= 1e-6:
        return None
    taps = np.exp(-0.5 * (np.arange(-2, 3, dtype=np.float64) / sigma) ** 2)
    return (taps / taps.sum()).astype(np.float32)


def apply_photometric(image: np.ndarray, params: PhotometricParams, *, seed: int, frame_index: int) -> np.ndarray:
    """LUT (gain/gamma/balance) -> separable 5-tap blur -> Gaussian noise keyed on (seed, frame_index); uint8 in, uint8 out."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("photometric ops expect a uint8 HxWx3 image")
    lut = photometric_lut(params)
    out = np.stack([lut[c][image[:, :, c]] for c in range(3)], axis=-1).astype(np.float32)
    kernel = _blur_kernel(params.blur_sigma_px)
    if kernel is not None:
        padded = np.pad(out, ((2, 2), (0, 0), (0, 0)), mode="edge")
        rows = kernel[0] * padded[0:-4] + kernel[1] * padded[1:-3] + kernel[2] * padded[2:-2] + kernel[3] * padded[3:-1] + kernel[4] * padded[4:]
        padded = np.pad(rows, ((0, 0), (2, 2), (0, 0)), mode="edge")
        out = kernel[0] * padded[:, 0:-4] + kernel[1] * padded[:, 1:-3] + kernel[2] * padded[:, 2:-2] + kernel[3] * padded[:, 3:-1] + kernel[4] * padded[:, 4:]
    if params.noise_sigma > 1e-6:
        rng = np.random.default_rng([int(seed), int(frame_index), _NOISE_TAG])
        out = out + rng.standard_normal(out.shape, dtype=np.float32) * np.float32(params.noise_sigma)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def ground_texture(seed: int, width: int, height: int, *, contrast: float, base_rgb: Sequence[float]) -> np.ndarray:
    """Procedural two-octave speckle, darkness baked in (textures modulate material rgba, so the material is set to 1)."""
    rng = np.random.default_rng([int(seed), 0x7EC5])
    coarse = rng.uniform(-1, 1, size=(height // 8, width // 8))
    coarse = np.kron(coarse, np.ones((8, 8)))[:height, :width]
    fine = rng.uniform(-1, 1, size=(height, width))
    field = 1.0 + float(contrast) * (0.6 * coarse + 0.4 * fine)
    base = np.asarray(base_rgb, dtype=np.float64).reshape(1, 1, 3)
    return np.clip(np.rint(field[:, :, None] * base * 255.0), 0, 255).astype(np.uint8)


def skybox_texture(top_rgb: Sequence[float], bottom_rgb: Sequence[float], width: int, height: int) -> np.ndarray:
    """Vertical gradient in the builtin skybox layout (a tall strip of six faces; a gradient down the strip is enough)."""
    ramp = np.linspace(1.0, 0.0, height)[:, None, None]
    top = np.asarray(top_rgb, dtype=np.float64).reshape(1, 1, 3)
    bottom = np.asarray(bottom_rgb, dtype=np.float64).reshape(1, 1, 3)
    image = (top * ramp + bottom * (1 - ramp)) * 255.0
    return np.clip(np.rint(np.repeat(image, width, axis=1)), 0, 255).astype(np.uint8)


__all__ = [
    "APPEARANCE_RESOLVER_VERSION", "MATERIAL_NAMES", "AppearanceParams", "AppearanceRegime", "PhotometricParams",
    "appearance_seeds", "apply_photometric", "ground_texture", "photometric_lut", "resolve_appearance", "skybox_texture",
]
