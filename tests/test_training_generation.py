import torch

from llm.generation import generate
from llm.generation.sampling import filter_logits
from llm.model import LM, ModelConfig
from llm.training import TrainConfig, train
from llm.training.checkpoint import load_model


def toy(vocab=12):
    g = torch.Generator().manual_seed(0)
    base = torch.randint(0, vocab, (8,), generator=g)
    return base.repeat(200)           # séquence périodique : facile à mémoriser


def small(**kw):
    return LM(ModelConfig(vocab_size=12, block_size=16, n_layer=1, n_head=2, n_embd=16, **kw))


def test_training_reduces_loss_and_history_fields():
    toks = toy(); m = small()
    h = train(m, toks[:1200], toks[1200:], TrainConfig(steps=60, batch_size=8, eval_interval=30, eval_iters=3, lr=1e-2, device="cpu"))
    assert h["val_loss"][-1] < h["val_loss"][0] and len(h["step"]) == len(h["lr"]) >= 3


def test_gradient_accumulation_equals_big_batch_gradient():
    torch.manual_seed(0); toks = toy(); m = small().eval()
    x = torch.stack([toks[i:i + 16] for i in range(0, 8)]); y = torch.stack([toks[i + 1:i + 17] for i in range(0, 8)])
    m.zero_grad(); m(x, y).loss.backward(); g_full = m.tok_emb.weight.grad.clone()
    m.zero_grad()
    for half in (slice(0, 4), slice(4, 8)):
        (m(x[half], y[half]).loss / 2).backward()
    assert torch.allclose(g_full, m.tok_emb.weight.grad, atol=1e-6)


def test_all_optimizers_and_schedules_run():
    toks = toy()
    for opt in ("sgd", "sgd_momentum", "nesterov", "rmsprop", "adagrad", "adam", "adamw", "lion", "muon"):
        h = train(small(), toks[:1200], toks[1200:], TrainConfig(steps=4, batch_size=4, eval_interval=4, eval_iters=1,
                                                                   optimizer=opt, lr=1e-3, device="cpu"))
        assert torch.isfinite(torch.tensor(h["val_loss"][-1])), opt
    for sch in ("constant", "linear", "cosine", "warmup_cosine", "warmup_linear", "inv_sqrt", "wsd"):
        train(small(), toks[:1200], toks[1200:], TrainConfig(steps=4, batch_size=4, eval_interval=4, eval_iters=1, schedule=sch, device="cpu"))


def test_extra_options_and_resume(tmp_path):
    toks = toy(); ck = str(tmp_path / "m.pt")
    cfg = TrainConfig(steps=6, batch_size=4, eval_interval=3, eval_iters=1, grad_accum=2, dtype="bfloat16", z_loss=1e-4,
                      label_smoothing=0.1, ema_decay=0.9, grad_checkpointing=True, checkpoint=ck, device="cpu")
    h = train(small(), toks[:1200], toks[1200:], cfg)
    assert h["ema_model"] is not None
    model, payload = load_model(ck)
    assert payload["step"] > 0
    cfg2 = TrainConfig(steps=9, batch_size=4, eval_interval=3, eval_iters=1, checkpoint=ck, resume=True, device="cpu")
    h2 = train(small(), toks[:1200], toks[1200:], cfg2)
    assert h2["step"][-1] == 9


def test_early_stopping():
    toks = toy()
    h = train(small(), toks[:1200], toks[1200:], TrainConfig(steps=200, batch_size=4, eval_interval=2, eval_iters=1, lr=0.5,
                                                             early_stop_patience=2, device="cpu"))
    assert h["step"][-1] < 200


def test_generation_cache_matches_no_cache_greedy():
    m = small(attn_type="gqa", n_kv_head=1); prompt = torch.tensor([[1, 2, 3]])
    a = generate(m, prompt, 10, greedy=True, use_cache=True)
    b = generate(m, prompt, 10, greedy=True, use_cache=False)
    assert torch.equal(a, b)


def test_sampling_filters():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    assert torch.isfinite(filter_logits(logits, top_k=2)).sum() == 2
    p = filter_logits(logits, top_p=0.5); assert torch.isfinite(p).sum() >= 1 and torch.isfinite(p[0, 0])
    mp = filter_logits(logits, min_p=0.5); assert torch.isfinite(mp).sum() == 1 or torch.isfinite(mp).sum() == 2
    assert torch.isfinite(filter_logits(logits)).sum() == 5


def test_resume_is_identical_to_continuous_training(tmp_path):
    toks = toy(); kw = dict(batch_size=4, eval_interval=10, eval_iters=2, ema_decay=0.9, device="cpu", save_best_only=False, lr=5e-3)
    torch.manual_seed(0); a = small(); ha = train(a, toks[:1200], toks[1200:], TrainConfig(steps=20, **kw))
    ck = str(tmp_path / "r.pt")
    torch.manual_seed(0); b = small(); train(b, toks[:1200], toks[1200:], TrainConfig(steps=10, checkpoint=ck, **kw))
    torch.manual_seed(0); c = small(); hc = train(c, toks[:1200], toks[1200:], TrainConfig(steps=20, checkpoint=ck, resume=True, **kw))
    assert abs(ha["final_val_loss"] - hc["final_val_loss"]) < 1e-6
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), c.parameters()))
