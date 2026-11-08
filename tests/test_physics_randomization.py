"""Physics randomization (2026-11-05, notes/handoff-20261105.md step 2): contract, adapter, capture, screening, RL pool."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.contracts.physics import REGIMES, PhysicsRegime, physics_regime, physics_seeds, resolve_physics
from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.simulation.suites import SimulationScenario, SimulationSuite, _suite_from_payload, scenario_record, suite_payload

SERVO_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1_servo/scene_bench_pick_replace_v1.xml"
REGIME = "bench_physics_v1"


def test_regime_draws_are_deterministic_and_inside_their_ranges():
    regime = physics_regime(REGIME)
    assert PhysicsRegime.from_mapping(json.loads(json.dumps(regime.identity()))) == regime
    for seed in physics_seeds(3, 200):
        p = resolve_physics(regime, seed)
        assert p == resolve_physics(regime, seed)
        for scale, (low, high) in ((p.kp_scale, regime.kp_scale), (p.kv_scale, regime.kv_scale), (p.friction_scale, regime.friction_scale)):
            assert all(low <= v <= high for v in scale)
        assert regime.arm_mass_scale[0] <= p.arm_mass_scale <= regime.arm_mass_scale[1]
        assert 0.0 <= p.latency_s <= 0.1 and regime.cube_mass_kg[0] <= p.cube_mass_kg <= regime.cube_mass_kg[1]
        assert 0.0 <= p.pause_s(90) <= 1.0 and p.pause_s(90) == p.pause_s(90)
        assert np.array_equal(p.sensor_noise_act(17), p.sensor_noise_act(17))
    fixed = resolve_physics(REGIMES["real_like_v1"], 5)
    assert fixed.kp_scale == (1.0,) * 6 and fixed.latency_s == pytest.approx(0.05) and fixed.pause_s(90) == 0.0
    with pytest.raises(ValueError):
        PhysicsRegime(name="bad", kp_scale=(2.0, 1.0))


def test_physics_fields_round_trip_and_unset_records_are_unchanged():
    base = SimulationScenario("s", (0.0, 0.28, 0.011), (0.0,) * 6, (1.0, 0.0, 0.0, 0.0))
    assert "physics_regime" not in scenario_record(base) and "physics_seed" not in scenario_record(base)
    drawn = replace(base, scenario_id="p", physics_regime=REGIME, physics_seed=42)
    suite = SimulationSuite("suite", 1, "bench_pick_replace_v1", 1, False, (base, drawn))
    assert _suite_from_payload(json.loads(json.dumps(suite_payload(suite)))).scenarios == suite.scenarios
    with pytest.raises(ValueError):
        replace(base, physics_regime=REGIME)            # a regime without a seed
    with pytest.raises(ValueError):
        replace(base, physics_regime="nope", physics_seed=1)


def _scenario(index=0, seed=None, regime=REGIME):
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import certification_suite
    s = certification_suite(scene_bench_config(SERVO_SCENE), repeats=1).scenarios[index]
    return s if seed is None else replace(s, physics_regime=regime, physics_seed=seed)


def _teacher_actions(adapter, scenario, steps=200):
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    adapter.reset(scenario)
    controller = PrivilegedStagedController()
    controller.reset(adapter)
    out = []
    for t in range(steps):
        out.append(controller.predict(None, adapter.current_act(), adapter))
        if t % 90 == 0:
            adapter.inference_pause(t)
        adapter.apply_policy_command(out[-1])
        adapter.advance_control_period()
    return np.asarray(out), adapter.mujoco_qpos().copy()


def test_adapter_applies_a_draw_restores_the_scene_and_is_a_no_op_when_off():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.adapter import _PHYSICS_ARRAYS, MujocoTaskAdapter
    fresh = MujocoTaskAdapter(SERVO_SCENE)
    reused = MujocoTaskAdapter(SERVO_SCENE)
    try:
        pristine = {name: getattr(fresh.model, name).copy() for name in _PHYSICS_ARRAYS}
        reused.reset(_scenario(seed=11))
        assert reused.physics_record is not None and reused.physics_params.seed == 11
        assert not np.array_equal(reused.model.actuator_gainprm, pristine["actuator_gainprm"])
        # An adapter that ran a draw and a fresh one behave bit for bit alike once reset without physics.
        a, qa = _teacher_actions(reused, _scenario())
        b, qb = _teacher_actions(fresh, _scenario())
        assert np.array_equal(a, b) and np.array_equal(qa, qb)
        assert all(np.array_equal(getattr(reused.model, n), pristine[n]) for n in _PHYSICS_ARRAYS)
        assert reused.physics_record is None and reused.inference_pause(0) == 0.0
        assert np.array_equal(reused.observed_act(), reused.current_act())
    finally:
        fresh.close(); reused.close()


def test_latency_delays_the_actuator_write_and_the_pause_runs_physics_with_the_clock_held():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    adapter = MujocoTaskAdapter(SERVO_SCENE)
    try:
        scenario = _scenario(seed=next(s for s in physics_seeds(1, 500) if resolve_physics(physics_regime(REGIME), s).latency_s > 0.05))
        adapter.reset(scenario)
        latency = adapter.physics_params.latency_s
        before = np.asarray(adapter.data.ctrl[adapter._actuator_ids]).copy()
        target = adapter.current_act().copy(); target[0] += 0.05
        adapter.apply_policy_command(target)
        assert np.array_equal(adapter.data.ctrl[adapter._actuator_ids], before)        # not yet at the servos
        while adapter.data.time < adapter.control_origin_time + latency - 0.004:
            adapter._physics_step()
        assert np.array_equal(adapter.data.ctrl[adapter._actuator_ids], before)
        while adapter.data.time < adapter.control_origin_time + latency + 0.004:
            adapter._physics_step()
        assert not np.array_equal(adapter.data.ctrl[adapter._actuator_ids], before)
        actions, t0 = adapter.control_actions, adapter.data.time
        boundary = next(b for b in range(0, 900, 90) if adapter.physics_params.pause_s(b) > 0.1)
        seconds = adapter.inference_pause(boundary)
        assert seconds == pytest.approx(adapter.physics_params.pause_s(boundary))
        assert adapter.data.time - t0 == pytest.approx(seconds, abs=0.003) and adapter.control_actions == actions
    finally:
        adapter.close()


def test_compensated_teacher_succeeds_under_a_draw_and_screening_admits_it():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.bench import screen_scenario
    regime = physics_regime(REGIME)
    # A soft-ish elbow, some latency and a long pause, the conditions the scan found hardest.
    seed = next(s for s in physics_seeds(2, 2000) if (lambda p: p.kp_scale[2] < 1.0 and p.latency_s > 0.04 and p.pause_s(180) > 0.5)(resolve_physics(regime, s)))
    ok = [screen_scenario(SERVO_SCENE, _scenario(i, seed=seed))[0] for i in range(4)]
    assert sum(ok) >= 3, ok


def test_physics_capture_is_stride_and_worker_invariant_and_records_the_draw(tmp_path):
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations
    from so_arm101_v2.simulation.bench import screen_scenario
    candidates = [_scenario(i, seed=s) for i in range(3) for s in physics_seeds(9, 3)]
    scenarios = tuple(replace(s, scenario_id=f"p{k}") for k, s in enumerate(c for c in candidates if screen_scenario(SERVO_SCENE, c)[0]))[:2]
    assert len(scenarios) == 2
    suite = SimulationSuite("bench_pick_replace_v1_test_physics", 1, "bench_pick_replace_v1", 1, False, scenarios)
    report = {"environment_proven": True, "deterministic": True, "suite": {"suite_id": suite.suite_id},
              "rollouts": [{"scenario_id": s.scenario_id, "success": True} for s in scenarios]}
    report["content_sha256"] = content_sha256(report)
    (tmp_path / "pre.json").write_text(json.dumps(report))
    runs = {(stride, workers): capture_oracle_demonstrations(SERVO_SCENE, suite, tmp_path / "pre.json", tmp_path / f"s{stride}w{workers}",
                                                             scenario="all", record_video=False, teacher_horizon=480, store_frames=True,
                                                             workers=workers, frame_row_stride=stride)
            for stride, workers in ((1, 1), (30, 2))}
    a, b = (json.loads(Path(r.manifest).read_text()) for r in runs.values())
    assert a["arrays"]["sha256"] == b["arrays"]["sha256"] and a["episodes"] == b["episodes"]
    assert a["physics"]["regimes"][REGIME]["kp_scale"] == list(physics_regime(REGIME).kp_scale) or a["physics"]["regimes"][REGIME]["kp_scale"] == physics_regime(REGIME).kp_scale
    assert a["episodes"][0]["physics"]["params"]["seed"] == scenarios[0].physics_seed
    fa = np.load(Path(runs[(1, 1)].directory) / "images.npy", mmap_mode="r")
    fb = np.load(Path(runs[(30, 2)].directory) / "images.npy", mmap_mode="r")
    for episode in range(2):
        delta = np.abs(fa[episode * 480:(episode + 1) * 480:30].astype(np.int16) - fb[episode * 16:(episode + 1) * 16].astype(np.int16))
        assert int(delta.max()) <= 1 and float((delta > 0).mean()) < 1e-3


def test_parallel_screening_builds_the_same_physics_suite(tmp_path):
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import generate_bench_suite
    bench = scene_bench_config(SERVO_SCENE)
    kwargs = dict(seed=16, count=3, repeats=1, randomize_appearance=bench.appearance is not None, randomize_placement=True, physics_regime=REGIME)
    first, a = generate_bench_suite(SERVO_SCENE, bench, output_dir=tmp_path / "a", **kwargs)
    second, b = generate_bench_suite(SERVO_SCENE, bench, output_dir=tmp_path / "b", workers=3, **kwargs)
    assert a.read_bytes() == b.read_bytes() and "_physics_" in first.suite_id
    assert all(s.physics_regime == REGIME and s.physics_seed is not None for s in first.scenarios)
    assert len({s.physics_seed for s in first.scenarios}) == 3


def test_rl_pool_gives_each_episode_its_own_draw():
    from so_arm101_v2.rl.scenarios import ScenarioPool
    base = [SimulationScenario(f"s{i}", (0.01 * i, 0.28, 0.011), (0.0,) * 6, (1.0, 0.0, 0.0, 0.0)) for i in range(3)]
    plain, drawn = ScenarioPool(base), ScenarioPool(base, physics_regime=REGIME)
    assert plain.episode(4) is base[int(plain.order[1])] and plain.episode(4).physics_seed is None
    assert drawn.episode(1).scenario_id == drawn.episode(4).scenario_id and drawn.episode(1).physics_seed != drawn.episode(4).physics_seed
    assert drawn.episode(4).physics_seed == ScenarioPool(base, physics_regime=REGIME).episode(4).physics_seed


def test_regime_v2_spans_the_soft_servo_and_the_stiff_nominal_scene():
    """bench_physics_v2 (2026-11-08): every joint's kp and friction range on the SO-101 servo scene contains both the
    servo model's value and the nominal scene's (kp 50, friction 0.1); v1 did not (elbow kp 4.5-13)."""
    import numpy as np
    from so_arm101_v2.data.resources import load_json_resource
    servo = load_json_resource("servo_model_20261108_so101.json")
    regime = physics_regime("bench_physics_v2")
    for scales, base, nominal in ((regime.kp_scale, servo["kp"], 50.0), (regime.friction_scale, servo["frictionloss"], 0.1)):
        for (low, high), value in zip(scales, base):
            assert low * value <= value <= high * value
            assert low * value <= nominal <= high * value
    draws = np.array([resolve_physics(regime, s).kp_scale for s in range(500)])
    assert draws[:, 2].max() > 5.0 and np.delete(draws, [0, 2, 3], axis=1).max() <= 2.0
