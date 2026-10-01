import torch

from llm.model import LM, ModelConfig
from llm.model.moe import MoE


def cfg(**kw):
    base = dict(vocab_size=30, block_size=16, n_layer=1, n_head=2, n_embd=16, moe=True, n_experts=4, top_k=2, expert_hidden=16)
    base.update(kw)
    return ModelConfig(**base)


def test_topk_selection_and_load_sums_to_one():
    moe = MoE(cfg()).train()
    y, aux = moe(torch.randn(3, 8, 16))
    assert y.shape == (3, 8, 16)
    assert torch.isclose(moe.stats["load"].sum(), torch.tensor(1.0))
    assert aux > 0                       # perte d'équilibrage Switch


def test_balance_loss_minimum_is_coef_for_uniform_router():
    moe = MoE(cfg(balance_coef=1.0)).train()
    torch.nn.init.zeros_(moe.router.weight)   # routeur uniforme : P_e = 1/E, f_e ~ 1/E => E * sum(f P) = 1
    _, aux = moe(torch.randn(4, 16, 16))
    assert torch.isclose(aux, torch.tensor(1.0), atol=1e-4)


def test_capacity_drops_tokens():
    moe = MoE(cfg(capacity_factor=0.5)).train()
    torch.nn.init.zeros_(moe.router.weight); moe.router.bias = None
    with torch.no_grad():
        moe.router.weight[0] = 5.0            # force presque tout vers l'expert 0
    moe(torch.randn(2, 16, 16))
    assert moe.stats["dropped"] > 0


def test_no_capacity_means_no_drop():
    moe = MoE(cfg()).train(); moe(torch.randn(2, 16, 16))
    assert moe.stats["dropped"] == 0


def test_expert_choice_is_perfectly_balanced():
    moe = MoE(cfg(router="expert_choice")).train(); moe(torch.randn(2, 16, 16))
    load = moe.stats["load"]
    assert torch.allclose(load, torch.full_like(load, 0.25))


def test_sigmoid_bias_updates_towards_balance():
    moe = MoE(cfg(router="sigmoid_bias", bias_update_speed=0.1)).train()
    with torch.no_grad():
        moe.route_bias[0] = 50.0              # expert 0 systématiquement choisi => surchargé
    moe(torch.randn(4, 16, 16))
    assert moe.route_bias[0] < 50.0           # le biais de l'expert surchargé diminue
    assert (moe.route_bias[1:] > 0).all()     # ceux des experts sous-chargés augmentent


def test_shared_experts_and_zloss_increase_parameters_and_aux():
    a = MoE(cfg(z_loss_coef=0.0, balance_coef=0.0)).train(); b = MoE(cfg(n_shared_experts=1, z_loss_coef=1.0, balance_coef=0.0)).train()
    assert sum(p.numel() for p in b.parameters()) > sum(p.numel() for p in a.parameters())
    x = torch.randn(2, 8, 16)
    assert a(x)[1] == 0 and b(x)[1] > 0


def test_active_params_less_than_total():
    m = LM(cfg(n_layer=2))
    assert m.num_parameters(active_only=True) < m.num_parameters()


def test_moe_model_trains():
    torch.manual_seed(0)
    m = LM(cfg(vocab_size=8, n_layer=2)); x = torch.randint(0, 8, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=0.02); first = m(x, x).loss.item()
    for _ in range(30):
        opt.zero_grad(); o = m(x, x); (o.loss + o.aux_loss).backward(); opt.step()
    assert m(x, x).loss.item() < first
