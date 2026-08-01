import json
import tempfile
import unittest
from pathlib import Path

import torch

import run_act_ppo_ablation_campaign as campaign
import train_act_in_sim as train


class FakeACT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.decoder = torch.nn.Linear(2, 2)
        self.model.decoder_pos_embed = torch.nn.Embedding(2, 2)
        self.model.action_head = torch.nn.Linear(2, 1)
        self.model.encoder = torch.nn.Linear(2, 2)


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act_policy = FakeACT()
        self.log_std = torch.nn.Parameter(torch.zeros(6))


class TrainerInterfaceTest(unittest.TestCase):
    def test_actor_scopes_are_exact_and_keep_log_std(self):
        expected = {
            "all": {"decoder", "decoder_pos_embed", "action_head", "encoder"},
            "decoder_head": {"decoder", "decoder_pos_embed", "action_head"},
            "action_head": {"action_head"},
        }
        for scope, modules in expected.items():
            policy = FakePolicy()
            names = train.configure_actor_train_scope(policy, scope)
            self.assertIn("log_std", names)
            selected = {
                name.split(".")[2]
                for name in names
                if name.startswith("act_policy.model.")
            }
            self.assertEqual(selected, modules)

    def test_branch_loads_weights_without_optimizer_or_counter_contract(self):
        train.torch = torch
        policy = FakePolicy()
        critic = torch.nn.Linear(2, 1)
        source = FakePolicy()
        source.log_std.data.fill_(-2.0)
        source_critic = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.pt"
            torch.save(
                {
                    "episode": 149,
                    "total_chunks": 3600,
                    "act_policy": source.act_policy.state_dict(),
                    "log_std": source.log_std.detach(),
                    "critic": source_critic.state_dict(),
                    "policy_optimizer": {"must_not": "load"},
                },
                path,
            )
            payload = train.load_branch_checkpoint(path, policy, critic, torch.device("cpu"))
        self.assertEqual(payload["episode"], 149)
        self.assertTrue(torch.equal(policy.log_std, source.log_std))
        for left, right in zip(policy.act_policy.parameters(), source.act_policy.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_restored_optimizer_lr_is_explicitly_reapplied(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=9e-4)
        train.set_optimizer_lr(optimizer, 3e-7)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-7)


class CampaignControllerTest(unittest.TestCase):
    def test_original_model_invariant_and_no_reset_bank(self):
        campaign.verify_invariants()
        command = campaign.training_command(
            Path("/tmp/run"), campaign.BASE_CONFIG, 17, 40, 1,
            "control", campaign.BRANCH_CHECKPOINT,
        )
        joined = " ".join(command)
        self.assertIn("--reset-curriculum none", joined)
        self.assertNotIn("--reset-bank", command)
        self.assertNotIn("--randomize-block-reset", command)

    def test_generation_one_is_reproducible_and_unique(self):
        signatures = [campaign.config_signature(item["config"]) for item in campaign.GENERATION_ONE]
        self.assertEqual(len(signatures), 20)
        self.assertEqual(len(set(signatures)), 20)
        self.assertEqual(campaign.GENERATION_ONE[0]["config"], campaign.BASE_CONFIG)

    def test_zero_precursor_does_not_become_champion(self):
        candidate = {"rank": [0.0] * 7 + [-1.0, -1.0, 0.0]}
        self.assertFalse(campaign.strictly_better(candidate, None))
        progressed = {"rank": [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0]}
        self.assertTrue(campaign.strictly_better(progressed, None))

    def test_pair_rank_prioritizes_worst_seed(self):
        consistent = [
            {"strict_success_rate": 0.2},
            {"strict_success_rate": 0.2},
        ]
        brittle = [
            {"strict_success_rate": 0.0},
            {"strict_success_rate": 0.8},
        ]
        self.assertGreater(campaign.pair_rank(consistent), campaign.pair_rank(brittle))

    def test_state_round_trip_preserves_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            state = campaign.load_state(output)
            state["trials"]["done"] = {"status": "complete"}
            campaign.save_state(output, state)
            restored = campaign.load_state(output)
        self.assertEqual(restored["trials"]["done"]["status"], "complete")
        self.assertEqual(restored["excluded_jobs"]["29"], "reverted simulator collision-model experiment")


if __name__ == "__main__":
    unittest.main()
