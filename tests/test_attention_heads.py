"""tools/attention_heads.py: the written-out attention must be the module it replaces, and its two interventions must do what they say."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from attention_heads import HookedMHA, region_lift  # noqa: E402


def _pair(seed: int = 0):
    torch.manual_seed(seed)
    mha = torch.nn.MultiheadAttention(64, 8, dropout=0.1).eval()
    query, key, value = torch.randn(5, 3, 64), torch.randn(7, 3, 64), torch.randn(7, 3, 64)
    return mha, HookedMHA(mha), query, key, value


def test_hooked_attention_matches_the_module():
    mha, hooked, query, key, value = _pair()
    with torch.no_grad():
        want, weights = mha(query, key, value, average_attn_weights=False)
        hooked.record = True
        got, _ = hooked(query, key, value)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.allclose(hooked.attn[0], weights, atol=1e-6)        # (B, H, L, S)
    assert hooked.count == 3 and hooked.sum_out.shape == (8, 5, 8)


def test_blocked_keys_get_no_weight_and_match_a_key_padding_mask():
    mha, hooked, query, key, value = _pair(1)
    block = torch.zeros(3, 7, dtype=torch.bool)
    block[:, 2] = True
    block[1, 5] = True
    with torch.no_grad():
        want, _ = mha(query, key, value, key_padding_mask=block)
        hooked.record, hooked.block = True, block
        got, _ = hooked(query, key, value)
    assert torch.allclose(got, want, atol=1e-5)
    assert float(hooked.attn[0][:, :, :, 2].abs().max()) == 0.0
    assert torch.allclose(hooked.attn[0].sum(-1), torch.ones(3, 8, 5), atol=1e-5)


def test_mean_ablation_removes_only_that_heads_dependence_on_the_input():
    mha, hooked, query, key, value = _pair(2)
    with torch.no_grad():
        hooked.record = True
        clean, _ = hooked(query, key, value)
        hooked.record = False
        hooked.mean_out = hooked.sum_out / hooked.count
        hooked.ablate = tuple(range(8))
        flat, _ = hooked(query, key, value)
        hooked.ablate = (3,)
        one, _ = hooked(query, key, value)
    assert torch.allclose(flat[:, 0], flat[:, 1], atol=1e-6)         # every head replaced: the same output for every sample
    assert not torch.allclose(one, clean, atol=1e-4)
    assert not torch.allclose(one[:, 0], one[:, 1], atol=1e-4)


def test_region_lift_is_one_for_uniform_attention():
    region = np.zeros((2, 64), np.float32)
    region[:, :8] = 1
    assert region_lift(np.full((2, 64), 1 / 64, np.float32), region) == pytest.approx(1.0)
    assert region_lift(region / 8, region) == pytest.approx(8.0)
