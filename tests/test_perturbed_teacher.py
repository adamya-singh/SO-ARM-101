"""Perturbed-teacher capture (2026-11-05, notes/handoff-20261105.md step 1): teacher, scenario fields, DART labels, metric."""
from __future__ import annotations

from dataclasses import replace
import glob
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT
from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.simulation.suites import SimulationScenario, SimulationSuite, _suite_from_payload, scenario_record, suite_payload

LIVE_SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
EXPERIMENTS = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments"
DESCEND = 0.020   # feasibility scan 2026-11-05: 79/80 admitted at 20 mm, 48/80 at the clean 8 mm


def test_teacher_fields_round_trip_and_unperturbed_records_are_unchanged():
    base = SimulationScenario("s", (0.0, 0.28, 0.011), (0.0,) * 6, (1.0, 0.0, 0.0, 0.0))
    record = scenario_record(base)
    assert "teacher_offset_xy_m" not in record and "teacher_descend_height_m" not in record
    assert base.teacher_reset_kwargs() == {}
    perturbed = replace(base, scenario_id="p", teacher_offset_xy_m=(0.01, -0.005), teacher_descend_height_m=DESCEND)
    assert perturbed.teacher_reset_kwargs() == {"lateral_offset_xy_m": (0.01, -0.005), "perturbed_descend_height_m": DESCEND}
    suite = SimulationSuite("suite", 1, "bench_pick_replace_v1", 1, False, (base, perturbed))
    again = _suite_from_payload(json.loads(json.dumps(suite_payload(suite))))
    assert again.scenarios == suite.scenarios
    with pytest.raises(ValueError):
        replace(base, teacher_offset_xy_m=(0.1, 0.0))
    with pytest.raises(ValueError):
        replace(base, teacher_descend_height_m=0.02)   # a height without an offset


def test_offset_draws_are_deterministic_and_inside_the_disk():
    from so_arm101_v2.simulation.bench import teacher_offset_draws
    a = teacher_offset_draws(15, 2000, 0.015)
    assert a == teacher_offset_draws(15, 2000, 0.015) and a != teacher_offset_draws(16, 2000, 0.015)
    r = np.hypot(*np.asarray(a).T)
    assert r.max() <= 0.015 and np.median(r) == pytest.approx(0.015 / np.sqrt(2), rel=0.05)   # uniform in area


def test_targets_use_the_clean_label_and_drop_starts_inside_the_survey_chunk():
    from so_arm101_v2.learning.chunked import PERTURBED_LABEL_ROWS, build_chunked_targets, perturbed_start_mask
    from so_arm101_v2.simulation.oracle import perturbation_label_rows
    assert PERTURBED_LABEL_ROWS == perturbation_label_rows(90)
    rng = np.random.default_rng(0)
    rows = 480
    arrays = {"executed_act": rng.uniform(-1, 1, (rows, 6)).astype(np.float32), "current_act": rng.uniform(-1, 1, (rows, 6)).astype(np.float32),
              "action_index": np.arange(rows)}
    before = build_chunked_targets(arrays, 90, episode_lengths=[rows])
    assert perturbed_start_mask(arrays).all()
    label = arrays["executed_act"].copy()
    label[90:180] = 0.25
    dart = {**arrays, "label_act": label, "teacher_perturbed": np.ones(rows, dtype=bool)}
    after = build_chunked_targets(dart, 90, episode_lengths=[rows])
    assert np.array_equal(after[180:], before[180:])               # the correction is learned from what the teacher did
    assert not np.array_equal(after[90], before[90])                # the survey chunk copies the clean plan
    mask = perturbed_start_mask(dart)
    assert mask[90] and mask[180] and not mask[91:180].any() and mask[:90].all() and mask[180:].all()
    assert perturbed_start_mask({**dart, "teacher_perturbed": np.zeros(rows, dtype=bool)}).all()


def _jaw(adapter):
    return (adapter.data.site("fixed_jaw_tip").xpos + adapter.data.site("moving_jaw_tip").xpos) / 2


def _episode(scenario, *, rows=260):
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.privileged import PrivilegedStagedController
    adapter = MujocoTaskAdapter(LIVE_SCENE)
    try:
        adapter.reset(scenario)
        controller = PrivilegedStagedController()
        controller.reset(adapter, **scenario.teacher_reset_kwargs())
        actions, jaw = [], {}
        for row in range(rows):
            actions.append(controller.predict(None, adapter.current_act(), adapter))
            adapter.apply_policy_command(actions[-1])
            adapter.advance_control_period()
            if row in (180, 250):
                jaw[row] = _jaw(adapter).copy()
        return controller, np.asarray(actions), jaw
    finally:
        adapter.close()


@pytest.mark.parametrize("direction_deg", [0.0, 135.0])   # 0 deg failed every placement before the recentre stage
def test_perturbed_teacher_shows_the_offset_at_180_and_corrects_by_250(direction_deg):
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import certification_suite, screen_scenario
    base = certification_suite(scene_bench_config(LIVE_SCENE), repeats=1).scenarios[0]
    a = np.deg2rad(direction_deg)
    scenario = replace(base, teacher_offset_xy_m=(0.015 * np.cos(a), 0.015 * np.sin(a)), teacher_descend_height_m=DESCEND)
    clean, clean_actions, clean_jaw = _episode(base)
    perturbed, actions, jaw = _episode(scenario)
    assert perturbed.boundaries[:6] == clean.boundaries[:6] and perturbed.boundaries[8:] == clean.boundaries[7:]
    assert np.array_equal(actions[:90], clean_actions[:90])
    assert not np.allclose(actions[90:180], clean_actions[90:180])
    assert np.array_equal(np.stack([perturbed.clean_action(r) for r in range(len(actions))]), clean_actions)
    assert np.hypot(*(jaw[180] - clean_jaw[180])[:2]) > 0.012
    assert np.hypot(*(jaw[250] - clean_jaw[250])[:2]) < 0.002
    assert screen_scenario(LIVE_SCENE, scenario) == (True, "success")


def test_perturbed_capture_writes_dart_labels_and_unperturbed_capture_is_unchanged(tmp_path):
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import certification_suite
    from so_arm101_v2.simulation.oracle import capture_oracle_demonstrations
    base = certification_suite(scene_bench_config(LIVE_SCENE), repeats=1).scenarios[3]
    scenario = replace(base, teacher_offset_xy_m=(-0.009, 0.011), teacher_descend_height_m=DESCEND)
    clean, clean_actions, _ = _episode(base, rows=480)
    for label, scenarios in (("clean", (base,)), ("perturbed", (scenario,))):
        suite = SimulationSuite(f"bench_pick_replace_v1_test_{label}", 1, "bench_pick_replace_v1", 1, False, scenarios)
        report = {"environment_proven": True, "deterministic": True, "suite": {"suite_id": suite.suite_id},
                  "rollouts": [{"scenario_id": s.scenario_id, "success": True} for s in scenarios]}
        report["content_sha256"] = content_sha256(report)
        (tmp_path / f"{label}.json").write_text(json.dumps(report))
        result = capture_oracle_demonstrations(LIVE_SCENE, suite, tmp_path / f"{label}.json", tmp_path / label, scenario="all",
                                               record_video=False, teacher_horizon=480, workers=1)
        manifest = json.loads(Path(result.manifest).read_text())
        arrays = np.load(result.arrays)
        if label == "clean":
            assert "label_act" not in arrays.files and "teacher_perturbation" not in manifest
            assert "teacher_perturbation" not in manifest["episodes"][0]
            assert np.abs(arrays["executed_act"] - clean_actions).max() <= 1e-6   # executed = requested to the capture's 1e-6 check
            assert np.array_equal(arrays["requested_act"], clean_actions)
        else:
            assert manifest["teacher_perturbation"]["columns"] == ["label_act", "teacher_perturbed"]
            assert manifest["episodes"][0]["teacher_perturbation"]["label_rows"] == [90, 180]
            assert arrays["teacher_perturbed"].all()
            assert np.array_equal(arrays["label_act"][90:180], clean_actions[90:180])   # clean_action reproduces the clean teacher bit for bit
            assert np.array_equal(arrays["label_act"][:90], arrays["executed_act"][:90])
            assert np.array_equal(arrays["label_act"][180:], arrays["executed_act"][180:])


def test_correction_summary_reproduces_the_2026_09_18_analysis():
    pytest.importorskip("mujoco")
    from so_arm101_v2.simulation.closure import correction_summary
    runs = {"augmentation_20260912/evaluations/shift12": (11.4, 11.2), "augmentation_shift12_4800_20260912/evaluations/shift12": (9.7, 8.1)}
    for run, (at_179, at_250) in runs.items():
        paths = sorted(glob.glob(str(EXPERIMENTS / run / "policies/*/telemetry/*.json")))
        if not paths:
            pytest.skip("tranche telemetry not on this machine")
        summary = correction_summary(LIVE_SCENE, paths)
        assert round(summary["lateral_rms_mm_179"], 1) == at_179 and round(summary["lateral_rms_mm_250"], 1) == at_250   # shrinkage.txt


def test_parallel_screening_builds_the_same_perturbed_suite(tmp_path):
    pytest.importorskip("mujoco")
    from so_arm101_v2.contracts.bench import scene_bench_config
    from so_arm101_v2.simulation.bench import generate_bench_suite
    bench = scene_bench_config(LIVE_SCENE)
    kwargs = dict(seed=15, count=3, repeats=1, randomize_appearance=bench.appearance is not None, randomize_placement=True,
                  perturb_teacher_radius_m=0.015, teacher_descend_height_m=DESCEND)
    sequential, a = generate_bench_suite(LIVE_SCENE, bench, output_dir=tmp_path / "a", **kwargs)
    parallel, b = generate_bench_suite(LIVE_SCENE, bench, output_dir=tmp_path / "b", workers=3, **kwargs)
    assert a.read_bytes() == b.read_bytes()
    assert sequential.suite_id.endswith(sequential.suite_id[-12:]) and "_perturbed_" in sequential.suite_id
    assert all(s.teacher_offset_xy_m is not None and s.teacher_descend_height_m == DESCEND for s in sequential.scenarios)
