"""The runner on the simulator backend reproduces the closed-loop evaluation that scored the policy."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPOSITORY_ROOT

SCENE = REPOSITORY_ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"
LENS_RUN = REPOSITORY_ROOT / "artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/seed202_120k_lens_20260909"
CHECKPOINT = LENS_RUN / "models/vision_h90/5e96018881d4140f/model.pt"
STORED = LENS_RUN / "evaluations/nominal/policies/bench_pick_replace_v1_nominal_eval_3de9de823621/telemetry/vision.nominal.repeat0.json"


def _renderer_available() -> bool:
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        mujoco.Renderer(model, height=256, width=256).close()
        return True
    except Exception:
        return False


needs_artifacts = pytest.mark.skipif(not (CHECKPOINT.exists() and _renderer_available()), reason="lens-run checkpoint or EGL renderer unavailable")


@needs_artifacts
def test_runner_reproduces_evaluate_closed_loop_with_pinned_frames(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from so_arm101_v2.contracts import load_pick_place_contract
    from so_arm101_v2.physical.runner import run_episode
    from so_arm101_v2.physical.sim_backend import SimBackend
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.policy_specs import PolicySpec
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    torch.set_num_threads(1)
    # Record every wrist observation the reference evaluation used, keyed by control step.
    recorded: dict[int, np.ndarray] = {}
    original = MujocoTaskAdapter.render_wrist_observation

    def recording(self):
        image = original(self)
        recorded[int(self.control_actions)] = np.array(image, copy=True)
        return image

    monkeypatch.setattr(MujocoTaskAdapter, "render_wrist_observation", recording)
    bench = MujocoTaskAdapter(SCENE).bench
    suite = bench_suite(bench, [(0, 0)], label="runner_parity", repeats=1)
    spec = PolicySpec(kind="vision_chunked", checkpoint=str(CHECKPOINT), options=(("clamp_channels", (5,)), ("black_image", False)))
    result = evaluate_closed_loop(SCENE, suite, {"vision": spec}, tmp_path / "reference", environment_proven=True, record_video=False, workers=1)
    report = json.loads(Path(result.report_json).read_text())
    rollout = report["rollouts"][0]
    reference_rows = json.loads((tmp_path / "reference" / rollout["telemetry_path"]).read_text())["rows"] if not Path(rollout["telemetry_path"]).is_absolute() else json.loads(Path(rollout["telemetry_path"]).read_text())["rows"]
    monkeypatch.setattr(MujocoTaskAdapter, "render_wrist_observation", original)

    adapter = MujocoTaskAdapter(SCENE)
    contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
    backend = SimBackend(adapter, suite.scenarios[0], contract=contract, frame_source=lambda step: recorded[step])
    policy = VisionChunkedPolicy(CHECKPOINT, black_image=False, clamp_channels=(5,))
    try:
        episode = run_episode(backend, policy, bench=adapter.bench, max_actions=contract.max_actions)
    finally:
        backend.close()
    assert episode.success == rollout["success"] and episode.actions == len(reference_rows)
    assert episode.hold_frames == 0 and episode.aborted_reason is None
    for ours, theirs in zip(backend.rows, reference_rows):
        assert ours["current_act"] == theirs["current_act"]
        assert ours["requested_act"] == theirs["requested_act"]
        assert ours["executed_act"] == theirs["executed_act"]
        assert ours["robot_qpos"] == theirs["robot_qpos"]


@pytest.mark.skipif(not (CHECKPOINT.exists() and STORED.exists() and _renderer_available()), reason="stored lens-run telemetry or renderer unavailable")
def test_runner_with_live_renders_matches_the_stored_nominal_rollout() -> None:
    torch = pytest.importorskip("torch")
    from so_arm101_v2.contracts import load_pick_place_contract
    from so_arm101_v2.physical.runner import run_episode
    from so_arm101_v2.physical.sim_backend import SimBackend
    from so_arm101_v2.simulation.adapter import MujocoTaskAdapter
    from so_arm101_v2.simulation.bench import bench_suite
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy
    torch.set_num_threads(1)
    stored = json.loads(STORED.read_text())["rows"]
    adapter = MujocoTaskAdapter(SCENE)
    suite = bench_suite(adapter.bench, [(0, 0)], label="nominal_eval", repeats=3)
    contract = load_pick_place_contract(suite.task_contract, bench_config=adapter.bench)
    backend = SimBackend(adapter, suite.scenarios[0], contract=contract)
    policy = VisionChunkedPolicy(CHECKPOINT, black_image=False, clamp_channels=(5,))
    try:
        episode = run_episode(backend, policy, bench=adapter.bench, max_actions=contract.max_actions)
    finally:
        backend.close()
    assert episode.success is True and episode.hold_frames == 0 and backend.counts == {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    assert episode.actions == len(stored)
    delta = max(float(np.max(np.abs(np.asarray(a["executed_act"]) - np.asarray(b["executed_act"])))) for a, b in zip(backend.rows, stored))
    assert delta <= 1e-4, delta
    print(f"live-render parity: max |delta executed_act| = {delta:.2e}")
