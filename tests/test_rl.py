"""RL layer of the retry tranche (src/so_arm101_v2/rl), 2026-10-04."""

import json
from pathlib import Path

import numpy as np
import pytest

from so_arm101_v2.contracts.bench import scene_bench_config
from so_arm101_v2.rl.ppo import episode_gae, explained_variance, gae
from so_arm101_v2.rl.reward import RewardConfig, chunk_reward, potential
from so_arm101_v2.rl.scenarios import ScenarioPool, assert_disjoint, scenario_key
from so_arm101_v2.simulation.bench import bench_suite

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml"


# ---------------------------------------------------------------- reward


def test_terminal_rewards_and_shaping_bounds():
    config = RewardConfig()
    assert chunk_reward(config, phi_start=0, phi_end=0, success=True, invalidated=False)["total"] == 1.0
    assert chunk_reward(config, phi_start=0, phi_end=0, success=False, invalidated=True)["total"] == -1.0
    top = potential(config, strict_grasp_acquired=True, pickup_completed=True, place_distance_m=0.0)
    assert top == pytest.approx(0.3)
    assert potential(config, strict_grasp_acquired=False, pickup_completed=False, place_distance_m=0.0) == 0.0
    with pytest.raises(ValueError):
        RewardConfig(grasp_bonus=0.3, pickup_bonus=0.3)


def test_shaping_telescopes_and_hover_scores_below_success():
    config = RewardConfig()
    phis = [0.0, 0.0, 0.1, 0.2 + 0.1 * 0.4, 0.3, 0.2]       # grasp, pickup, carry closer, release (carry term drops)
    parts = [chunk_reward(config, phi_start=a, phi_end=b, success=False, invalidated=False)["shaping"] for a, b in zip(phis, phis[1:])]
    assert sum(parts) == pytest.approx(phis[-1] - phis[0])
    success_return = sum(parts) + chunk_reward(config, phi_start=0, phi_end=0, success=True, invalidated=False)["total"]
    hover_return = 0.0                                          # no contact: every potential term is 0
    assert success_return >= 0.7 > hover_return


# ---------------------------------------------------------------- GAE


def test_gae_matches_hand_computation_and_never_crosses_episodes():
    adv, ret = gae([0.0, 1.0], [0.5, 0.25], gamma=0.9, lam=0.8)
    d1 = 1.0 - 0.25
    d0 = 0.0 + 0.9 * 0.25 - 0.5
    assert adv[1] == pytest.approx(d1)
    assert adv[0] == pytest.approx(d0 + 0.9 * 0.8 * d1)
    assert ret == pytest.approx(adv + np.array([0.5, 0.25]))
    # two interleaved episodes give exactly their separate GAE
    ep = [7, 3, 7, 3, 7]
    rewards = [0.1, 0.0, 0.2, -1.0, 1.0]
    values = [0.3, 0.1, 0.4, 0.0, 0.6]
    a, r = episode_gae(ep, rewards, values, gamma=0.99, lam=0.95)
    a7, _ = gae([0.1, 0.2, 1.0], [0.3, 0.4, 0.6], gamma=0.99, lam=0.95)
    a3, _ = gae([0.0, -1.0], [0.1, 0.0], gamma=0.99, lam=0.95)
    assert a[[0, 2, 4]] == pytest.approx(a7)
    assert a[[1, 3]] == pytest.approx(a3)
    assert explained_variance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_clipped_surrogate_and_flow_logprob_microbatch_invariance():
    torch = pytest.importorskip("torch")
    from so_arm101_v2.rl.ppo import clipped_surrogate
    new = torch.tensor([0.0, 0.2]); old = torch.tensor([0.0, 0.0]); adv = torch.tensor([1.0, 1.0])
    loss, stats = clipped_surrogate(new, old, adv, 0.1)
    assert stats["clip_fraction"] == 0.5
    assert float(loss) == pytest.approx(-(1.0 + 1.1) / 2)


# ---------------------------------------------------------------- ReinFlow on a toy flow model


class ToyFlow:
    """A tiny flow 'model' with the ResidualModel flow interface (prefix/velocity/out_proj/stats)."""

    flow = True
    kind = "toy"

    def __init__(self):
        import torch
        from so_arm101_v2.learning.pi05 import QuantileStats
        self.device = torch.device("cpu")
        self.action_width = 8
        self.trunk = torch.nn.Linear(8 + 7 + 1, 16)
        self.out_proj = torch.nn.Linear(16, 8)
        self.stats = QuantileStats(np.full(7, -1, np.float32), np.ones(7, np.float32), np.full(6, -.5, np.float32), np.full(6, .5, np.float32))

    def eval(self):
        pass

    def set_scope(self, scope):
        return list(self.out_proj.named_parameters())

    def prefix(self, frames, states):
        import torch
        return dict(state=torch.as_tensor(states, dtype=torch.float32), bsize=len(states))

    def velocity(self, ctx, x, t):
        import torch
        h = torch.cat([x, ctx["state"][:, None, :].expand(-1, x.shape[1], -1), t[:, None, None].expand(-1, x.shape[1], 1)], -1)
        feats = torch.tanh(self.trunk(h))
        return self.out_proj(feats), feats


def test_reinflow_logprob_reproduces_rollout_logprob_at_any_microbatch():
    torch = pytest.importorskip("torch")
    from so_arm101_v2.rl.reinflow import ReinFlowPolicy
    torch.manual_seed(0)
    policy = ReinFlowPolicy(ToyFlow(), steps=4, sigma_min=0.05, sigma_max=0.2, micro=3)
    frames = np.zeros((5, 4, 4, 3), np.uint8)
    states = np.random.default_rng(0).uniform(-1, 1, (5, 7)).astype(np.float32)
    out = policy.act(frames, states, seed=1)
    for micro in (1, 2, 5):
        policy.micro = micro
        with torch.no_grad():
            logp, _ = policy.logprob(frames, states, out["records"])
        assert np.allclose(logp.numpy(), out["logp"], atol=1e-4)
    det = policy.act(frames, states, deterministic=True, seed=1)
    assert np.all(det["logp"] == 0.0)
    # the chain's padded channels carry no noise
    chain = out["records"][0]["chain"]
    assert chain.shape == (5, 90, 8)


def test_gaussian_act_policy_logprob_matches_rollout():
    torch = pytest.importorskip("torch")
    from so_arm101_v2.rl.act_ppo import GaussianChunkPolicy

    class ToyACT:
        kind = "act"

        def __init__(self):
            from so_arm101_v2.learning.pi05 import QuantileStats
            self.device = torch.device("cpu")
            self.lin = torch.nn.Linear(7, 90 * 6)
            self.stats = QuantileStats(np.full(7, -1, np.float32), np.ones(7, np.float32), np.full(6, -.5, np.float32), np.full(6, .5, np.float32))

        def eval(self):
            pass

        def set_scope(self, scope):
            return list(self.lin.named_parameters())

        def mean(self, frames, states):
            return self.lin(torch.as_tensor(states)).view(-1, 90, 6)

    policy = GaussianChunkPolicy(ToyACT(), std_init=0.05, micro=2)
    states = np.random.default_rng(1).uniform(-1, 1, (3, 7)).astype(np.float32)
    out = policy.act(np.zeros((3, 4, 4, 3), np.uint8), states, seed=3)
    with torch.no_grad():
        logp, _ = policy.logprob(np.zeros((3, 4, 4, 3), np.uint8), states, out["records"])
    assert np.allclose(logp.numpy(), out["logp"], atol=1e-3)


# ---------------------------------------------------------------- environment


class RampPolicy:
    """90-step chunks: a slow wrist-roll ramp from the joints measured at the boundary (no image use)."""

    requires_pixels = False

    def __init__(self):
        self._buffer = []

    def reset(self, adapter=None):
        self._buffer = []

    @property
    def needs_frame(self):
        return not self._buffer

    def predict(self, image, current_act, adapter=None):
        if not self._buffer:
            self._buffer = list(ramp(current_act))
        return self._buffer.pop(0)


def ramp(current_act):
    delta = np.zeros(6, np.float32)
    delta[4] = 0.0002          # a slow wrist roll: inside every bound, so the episode runs to the contract's timeout
    return np.asarray([np.asarray(current_act, np.float32) + delta * (k + 1) for k in range(90)], np.float32)


def test_chunk_env_reproduces_the_evaluator_step_for_step(tmp_path):
    from so_arm101_v2.rl.env import PRIVILEGED_DIM, ChunkEnv
    from so_arm101_v2.simulation.rollout import evaluate_closed_loop
    config = scene_bench_config(SCENE)
    suite = bench_suite(config, [(0.0, 0.0)], label="rl_env_test", repeats=1)
    result = evaluate_closed_loop(SCENE, suite, {"ramp": RampPolicy}, tmp_path, environment_proven=True, record_video=False, workers=1)
    report = json.loads(Path(result.report_json).read_text())
    rows = json.loads(Path(report["rollouts"][0]["telemetry_path"]).read_text())["rows"]
    env = ChunkEnv(SCENE)
    try:
        obs = env.reset(suite.scenarios[0], frame=True)
        assert obs["frame"].shape == (256, 256, 3) and obs["privileged"].shape == (PRIVILEGED_DIM,)
        done = False
        while not done:
            obs, reward, done, info = env.step_chunk(ramp(obs["current_act"]), frame=False)
        qpos = env._qpos_rows
    finally:
        env.close()
    assert len(qpos) == len(rows)
    for a, b in zip(qpos, rows):
        assert np.allclose(a, b["robot_qpos"], atol=0, rtol=0)
    summary = info["summary"]
    assert summary.actions == len(rows) and not summary.success and not summary.invalidated


def test_vec_env_workers_do_not_import_torch_and_reset_deterministically():
    from so_arm101_v2.rl.vec_env import SubprocChunkVecEnv
    config = scene_bench_config(SCENE)
    scenario = bench_suite(config, [(0.003, -0.002)], label="rl_vec_test", repeats=1).scenarios[0]
    with SubprocChunkVecEnv(SCENE, 2) as vec:
        assert not any(info["torch_imported"] for info in vec.info)
        for i in range(2):
            vec.send_reset(i, scenario, frame=False)
        a, b = vec.recv(0), vec.recv(1)
        assert np.array_equal(a["privileged"], b["privileged"])
        vec.send_step(0, ramp(a["current_act"])[:10], frame=True)
        obs, reward, done, info = vec.recv(0)
        assert obs["frame"].shape == (256, 256, 3) and not done and info["steps"] == 10


# ---------------------------------------------------------------- scenarios


def test_scenario_pool_order_is_fixed_and_disjointness_is_checked():
    config = scene_bench_config(SCENE)
    suite = bench_suite(config, [(0.0, 0.0), (0.004, 0.0), (0.0, 0.004)], label="pool_test", repeats=1)
    pool_a, pool_b = ScenarioPool(list(suite.scenarios)), ScenarioPool(list(suite.scenarios) + [suite.scenarios[0]])
    assert len(pool_b) == 3 and pool_a.order_sha256 == pool_b.order_sha256
    assert [pool_a.episode(i).scenario_id for i in range(6)] == [pool_b.episode(i).scenario_id for i in range(6)]
    assert_disjoint(suite.scenarios[:1], suite.scenarios[1:])
    with pytest.raises(ValueError):
        assert_disjoint(suite.scenarios[:2], suite.scenarios[1:])
    assert scenario_key(suite.scenarios[0]) != scenario_key(suite.scenarios[1])


def test_reinflow_out_head_logprob_from_stored_features_equals_recomputation():
    torch = pytest.importorskip("torch")
    from so_arm101_v2.rl.reinflow import ReinFlowPolicy
    torch.manual_seed(0)
    model = ToyFlow()
    cached = ReinFlowPolicy(model, steps=4, sigma_min=0.05, sigma_max=0.2, scope="out_head")
    full = ReinFlowPolicy(model, steps=4, sigma_min=0.05, sigma_max=0.2, scope="heads")
    full.noise_head = cached.noise_head
    frames = np.zeros((3, 4, 4, 3), np.uint8)
    states = np.random.default_rng(2).uniform(-1, 1, (3, 7)).astype(np.float32)
    out = cached.act(frames, states, seed=5)
    assert out["records"][0]["features"].shape == (4, 90, 16)
    with torch.no_grad():
        a, _ = cached.logprob(frames, states, out["records"])
        b, _ = full.logprob(frames, states, out["records"])
    assert np.allclose(a.numpy(), out["logp"], atol=2e-2)      # features stored in float16
    assert np.allclose(a.numpy(), b.numpy(), atol=2e-2)
    # gradients reach only the output projection and the noise head
    a2, _ = cached.logprob(frames, states, out["records"])
    a2.sum().backward()
    assert model.out_proj.weight.grad is not None and model.trunk.weight.grad is None
