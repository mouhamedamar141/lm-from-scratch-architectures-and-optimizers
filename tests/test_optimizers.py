import pytest
import torch

from llm.training import optimizers as O
from llm.training.schedules import lr_factor


def run(make, steps=15):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 4))
    opt = make([p])
    for _ in range(steps):
        opt.zero_grad(); (p.pow(2).sum() + (p ** 3).mean()).backward(); opt.step()
    return p.detach()


CASES = [
    (lambda p: O.SGD(p, 0.05), lambda p: torch.optim.SGD(p, 0.05)),
    (lambda p: O.SGD(p, 0.05, momentum=0.9), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9)),
    (lambda p: O.SGD(p, 0.05, momentum=0.9, nesterov=True), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9, nesterov=True)),
    (lambda p: O.SGD(p, 0.05, weight_decay=0.1), lambda p: torch.optim.SGD(p, 0.05, weight_decay=0.1)),
    (lambda p: O.Adam(p, 0.01, weight_decay=0.1), lambda p: torch.optim.Adam(p, 0.01, weight_decay=0.1)),
    (lambda p: O.AdamW(p, 0.01, weight_decay=0.1), lambda p: torch.optim.AdamW(p, 0.01, weight_decay=0.1)),
    (lambda p: O.RMSprop(p, 0.01), lambda p: torch.optim.RMSprop(p, 0.01, alpha=0.99)),
    (lambda p: O.Adagrad(p, 0.05), lambda p: torch.optim.Adagrad(p, 0.05)),
]


@pytest.mark.parametrize("mine,ref", CASES)
def test_matches_torch(mine, ref):
    assert torch.allclose(run(mine), run(ref), atol=1e-5)


def test_l2_and_decoupled_weight_decay_differ_for_adam():
    assert not torch.allclose(run(lambda p: O.Adam(p, 0.01, weight_decay=0.5)),
                              run(lambda p: O.AdamW(p, 0.01, weight_decay=0.5)), atol=1e-3)


def test_lion_update_is_bounded_by_lr():
    p = torch.nn.Parameter(torch.ones(5)); opt = O.Lion([p], lr=0.1)
    (p * torch.arange(1., 6.)).sum().backward(); opt.step()
    assert torch.allclose(p.detach(), torch.full((5,), 0.9))     # |update| = lr quelle que soit l'échelle du gradient


def test_newton_schulz_orthogonalizes():
    X = O.newton_schulz(torch.randn(8, 16), steps=10)
    s = torch.linalg.svdvals(X)
    assert (s > 0.3).all() and (s < 1.5).all()                 # valeurs singulières poussées vers ~1


def test_muon_step_on_matrix():
    p = torch.nn.Parameter(torch.randn(8, 8)); opt = O.Muon([p], lr=0.02)
    before = p.detach().clone(); p.pow(2).sum().backward(); opt.step()
    assert not torch.allclose(before, p.detach())


def test_schedules():
    assert lr_factor("constant", 10, 100) == 1
    assert lr_factor("warmup_cosine", 0, 100, warmup=10) == pytest.approx(0.1)
    assert lr_factor("warmup_cosine", 10, 100, warmup=10) == pytest.approx(1.0)
    assert lr_factor("warmup_cosine", 100, 100, warmup=10, min_ratio=0.1) == pytest.approx(0.1)
    assert lr_factor("linear", 50, 100, min_ratio=0.0) == pytest.approx(0.5)
    assert lr_factor("wsd", 50, 100, warmup=5, decay_frac=0.2) == 1.0
    assert lr_factor("wsd", 100, 100, warmup=5, decay_frac=0.2, min_ratio=0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        lr_factor("inconnu", 0, 10)
