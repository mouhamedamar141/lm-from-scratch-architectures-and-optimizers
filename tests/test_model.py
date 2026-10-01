import pytest
import torch

from llm.model import LM, ModelConfig
from llm.model.positions import alibi_slopes, apply_rope, rope_inv_freq
from llm.model.attention import kv_cache_floats_per_token

VARIANTS = [
    dict(), dict(attn_type="gqa", n_kv_head=2), dict(attn_type="mqa"), dict(attn_type="mla", q_lora_rank=16),
    dict(position_mode="alibi"), dict(position_mode="learned"), dict(position_mode="sinusoidal"), dict(position_mode="sinusoidal", emb_scale=True),
    dict(position_mode="none"), dict(window=6), dict(qk_norm=True), dict(sdpa=True),
    dict(pre_norm=False, norm_type="layernorm", ffn_type="gelu", bias=True),
    dict(moe=True, n_experts=4, top_k=2, n_shared_experts=1), dict(moe=True, router="sigmoid_bias", n_experts=4),
    dict(rope_scaling="yarn", rope_factor=4.0), dict(rope_scaling="ntk", rope_factor=2.0),
    dict(attn_type="mla", position_mode="none"),
]


def make(**kw):
    return LM(ModelConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32, **kw)).eval()


@pytest.mark.parametrize("kw", VARIANTS, ids=[str(v) for v in VARIANTS])
def test_forward_shapes_and_cache_equivalence(kw):
    m = make(**kw)
    x = torch.randint(0, 50, (2, 20))
    full = m(x).logits
    assert full.shape == (2, 20, 50) and torch.isfinite(full).all()
    cache = m.new_cache()
    parts = [m(x[:, :12], cache=cache).logits] + [m(x[:, i:i + 1], cache=cache).logits for i in range(12, 20)]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-4)


@pytest.mark.parametrize("kw", VARIANTS[:8] + [dict(window=6), dict(attn_type="mla")], ids=str)
def test_causality_no_future_leak(kw):
    m = make(**kw)
    a = torch.randint(0, 50, (1, 16)); b = a.clone(); b[0, 10:] = (b[0, 10:] + 1) % 50
    la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)


def test_kv_cache_size_matches_formula():
    for kw, expected in [(dict(), 2 * 4 * 8), (dict(attn_type="gqa", n_kv_head=2), 2 * 2 * 8),
                         (dict(attn_type="mqa"), 2 * 8), (dict(attn_type="mla", kv_lora_rank=16, rope_head_dim=8), 16 + 8)]:
        m = make(**kw)
        cache = m.new_cache(); m(torch.randint(0, 50, (1, 10)), cache=cache)
        assert kv_cache_floats_per_token(m.cfg) == expected
        assert cache.nbytes() == 2 * 10 * expected * 4     # 2 couches, 10 tokens, float32


def test_sliding_window_cache_is_bounded():
    m = make(window=6)
    cache = m.new_cache(); m(torch.randint(0, 50, (1, 20)), cache=cache)
    assert cache.length() == 6


def test_gqa_has_fewer_parameters_than_mha():
    assert make(attn_type="gqa", n_kv_head=2).num_parameters() < make().num_parameters()
    assert make(attn_type="mqa").num_parameters() < make(attn_type="gqa", n_kv_head=2).num_parameters()


def test_rope_is_relative_and_norm_preserving():
    inv, _ = rope_inv_freq(16)
    q, k = torch.randn(16), torch.randn(16)
    def score(i, j):
        return (apply_rope(q[None], torch.tensor([i]), inv) * apply_rope(k[None], torch.tensor([j]), inv)).sum()
    assert torch.allclose(score(5, 2), score(105, 102), atol=1e-4)      # ne dépend que de i - j
    assert torch.allclose(apply_rope(q[None], torch.tensor([37]), inv).norm(), q.norm(), atol=1e-5)


def test_rope_scaling_variants_change_frequencies():
    base, _ = rope_inv_freq(16)
    lin, _ = rope_inv_freq(16, scaling="linear", factor=4.0)
    assert torch.allclose(lin, base / 4)
    ntk, _ = rope_inv_freq(16, scaling="ntk", factor=4.0)
    assert torch.allclose(ntk[0], base[0]) and (ntk[1:] < base[1:]).all()
    # contexte d'origine long (4096) : hautes fréquences conservées, basses fréquences interpolées
    yarn, ms = rope_inv_freq(16, scaling="yarn", factor=4.0, orig_len=4096)
    assert ms > 1.0 and yarn[0] == base[0] and torch.allclose(yarn[-1], base[-1] / 4)


def test_alibi_slopes():
    s = alibi_slopes(8)
    assert torch.allclose(s[0], torch.tensor(0.5)) and torch.allclose(s[-1], torch.tensor(2 ** -8.0))
    assert len(alibi_slopes(6)) == 6


def test_tied_weights_and_backward():
    m = make(); assert m.lm_head.weight is m.tok_emb.weight
    m.train(); out = m(torch.randint(0, 50, (2, 8)), torch.randint(0, 50, (2, 8)))
    out.loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_invalid_configs():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(attn_type="gqa", n_head=4, n_kv_head=3)
    with pytest.raises(ValueError):
        ModelConfig(moe=True, n_experts=2, top_k=3)
