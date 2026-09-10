"""Appearance regime, resolver and photometric ops (numpy only), plus the scenario/config plumbing."""
from __future__ import annotations

from dataclasses import replace
import json
import time

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.appearance import (
    APPEARANCE_RESOLVER_VERSION,
    MATERIAL_NAMES,
    AppearanceRegime,
    PhotometricParams,
    appearance_seeds,
    apply_photometric,
    ground_texture,
    photometric_lut,
    resolve_appearance,
    skybox_texture,
)
from so_arm101_v2.simulation.suites import SimulationScenario, SimulationSuite, _suite_from_payload, suite_payload


def test_resolver_is_deterministic_within_ranges_and_seeds_differ():
    regime = AppearanceRegime()
    a, b, c = resolve_appearance(regime, 7), resolve_appearance(regime, 7), resolve_appearance(regime, 8)
    assert a == b and a != c
    assert regime.headlight_ambient[0] <= a.headlight_ambient <= regime.headlight_ambient[1]
    assert abs(np.linalg.norm(a.light_dir) - 1) < 1e-9 and a.light_dir[2] < 0
    assert np.degrees(np.arccos(-a.light_dir[2])) <= regime.key_light_cone_deg + 1e-6
    assert tuple(m[0] for m in a.materials) == MATERIAL_NAMES
    for name, rgb, specular, shininess, reflectance in a.materials:
        lo, hi = {"white": regime.arm_albedo, "black": regime.motor_albedo, "groundplane": regime.ground_albedo, "black_pla": regime.cube_albedo}[name]
        assert all(lo <= v <= hi for v in rgb) and 0 <= specular <= 0.7 and 0 <= reflectance <= 0.3
    assert all(0.65 <= v <= 1.0 for v in a.napkin_rgb)
    if a.towel is not None:
        assert 0.9 <= a.towel[0] <= 1.6 and abs(np.degrees(a.towel[2])) <= 10
    assert 0.98 <= a.fovy_scale <= 1.02 and all(abs(v) <= 0.002 for v in a.camera_pos_offset)
    assert 0.6 <= a.photometric.gain <= 1.6 and 0.7 <= a.photometric.gamma <= 1.4
    # Identity round trip and JSON-ability of the record.
    json.dumps(a.as_record())
    assert AppearanceRegime.from_mapping(json.loads(json.dumps(regime.identity()))) == regime
    assert APPEARANCE_RESOLVER_VERSION == "bench_appearance_v1"
    # Spread: over 200 seeds, towel/texture/skybox toggles all occur.
    draws = [resolve_appearance(regime, s) for s in range(200)]
    assert any(d.towel is None for d in draws) and any(d.towel is not None for d in draws)
    assert any(d.skybox == "off" for d in draws) and any(d.skybox != "off" for d in draws)
    assert any(d.ground_texture is None for d in draws) and any(d.ground_texture is not None for d in draws)


def test_regime_validation_and_seed_stream():
    with pytest.raises(ValueError):
        AppearanceRegime(gain=(1.6, 0.6))
    with pytest.raises(ValueError):
        AppearanceRegime(shadow_probability=1.5)
    seeds = appearance_seeds(12, 5)
    assert all(type(s) is int and 0 <= s < 2 ** 31 for s in seeds) and len(set(seeds)) == 5
    assert seeds == appearance_seeds(12, 5) and seeds != appearance_seeds(12, 5, stream=1) and seeds != appearance_seeds(8, 5)
    with pytest.raises(ValueError):
        resolve_appearance(AppearanceRegime(), -1)


def test_photometric_ops_are_deterministic_identity_neutral_and_fast():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    identity = PhotometricParams(gain=1.0, gamma=1.0, balance=(1.0, 1.0, 1.0), noise_sigma=0.0, blur_sigma_px=0.0)
    assert np.array_equal(apply_photometric(image, identity, seed=1, frame_index=0), image)
    lut = photometric_lut(identity)
    assert np.array_equal(lut[0], np.arange(256, dtype=np.uint8))
    params = PhotometricParams(gain=1.3, gamma=0.8, balance=(1.05, 1.0, 0.95), noise_sigma=1.5, blur_sigma_px=0.7)
    a = apply_photometric(image, params, seed=5, frame_index=90)
    assert np.array_equal(a, apply_photometric(image, params, seed=5, frame_index=90))
    assert not np.array_equal(a, apply_photometric(image, params, seed=5, frame_index=91))
    assert not np.array_equal(a, apply_photometric(image, params, seed=6, frame_index=90))
    assert a.dtype == np.uint8 and a.shape == image.shape
    # A brighter LUT is monotone.
    assert np.all(np.diff(photometric_lut(params)[1].astype(int)) >= 0)
    started = time.perf_counter()
    for _ in range(10):
        apply_photometric(image, params, seed=5, frame_index=0)
    assert (time.perf_counter() - started) / 10 < 0.010
    # Textures.
    tex = ground_texture(3, 256, 256, contrast=0.4, base_rgb=(0.1, 0.1, 0.12))
    assert tex.shape == (256, 256, 3) and tex.dtype == np.uint8 and tex.std() > 0
    assert np.array_equal(tex, ground_texture(3, 256, 256, contrast=0.4, base_rgb=(0.1, 0.1, 0.12)))
    sky = skybox_texture((0.3, 0.5, 0.7), (0, 0, 0), 512, 3072)
    assert sky.shape == (3072, 512, 3) and sky[0, 0].tolist() == [76, 128, 178] and sky[-1, 0].tolist() == [0, 0, 0]


def test_scenario_seed_field_round_trips_and_is_hashed():
    base = SimulationScenario("s", (0.0, 0.28, 0.011), (0.0,) * 6, (1.0, 0.0, 0.0, 0.0))
    seeded = replace(base, scenario_id="s_seeded", fixed_appearance=False, appearance_seed=123)
    with pytest.raises(ValueError, match="fixed_appearance"):
        replace(base, appearance_seed=5)
    with pytest.raises(ValueError, match="appearance_seed"):
        replace(base, fixed_appearance=False, appearance_seed=-1)
    suite = SimulationSuite("suite", 1, "bench_pick_replace_v1", 1, False, (base, seeded))
    payload = suite_payload(suite)
    assert payload["scenarios"][1]["appearance_seed"] == 123 and payload["scenarios"][0]["appearance_seed"] is None
    again = _suite_from_payload(json.loads(json.dumps(payload)))
    assert again.scenarios == suite.scenarios
    # Packaged suites (no key) still load.
    legacy = dict(payload["scenarios"][0]); legacy.pop("appearance_seed")
    assert _suite_from_payload({**payload, "scenarios": [legacy]}).scenarios[0] == base
    from so_arm101_v2.data._serialization import content_sha256
    assert content_sha256(suite_payload(suite)) != content_sha256(suite_payload(SimulationSuite("suite", 1, "bench_pick_replace_v1", 1, False, (base, replace(seeded, appearance_seed=124)))))


def test_bench_config_carries_the_regime_and_the_scene_hash_covers_it(tmp_path):
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.bench import BenchConfig, scene_bench_config, scene_dependency_hash
    from tools.prepare_bench_scene import prepare
    live = scene_bench_config(REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml")
    regime = AppearanceRegime()
    with_regime = replace(live, appearance=regime.identity())
    assert with_regime.appearance_regime == regime and with_regime.appearance == regime.identity()
    with pytest.raises(ValueError):
        replace(live, appearance={"gain": (2.0, 1.0)})
    plain = prepare(tmp_path / "plain", live)
    randomized = prepare(tmp_path / "rand", with_regime)
    assert scene_dependency_hash(plain) != scene_dependency_hash(randomized)
    assert BenchConfig.load(tmp_path / "rand" / "bench_config.json").appearance_regime == regime
