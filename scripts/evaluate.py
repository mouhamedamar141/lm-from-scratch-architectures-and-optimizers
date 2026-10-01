"""Évalue un checkpoint : perte et perplexité sur train et validation (lots fixes)."""
import argparse

from _common import build_from_yaml
from llm.training import evaluate
from llm.training.checkpoint import load_model
from llm.utils.metrics import bits_per_char, perplexity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--checkpoint", default=None)
    a = p.parse_args()
    cfg, tok, _, tcfg, tr, va = build_from_yaml(a.config)
    model, _ = load_model(a.checkpoint or tcfg.checkpoint)
    import torch
    for name, toks in (("train", tr), ("val", va)):
        loss = evaluate(model, torch.as_tensor(toks), tcfg.batch_size, model.cfg.block_size, 20, "cpu")
        print(f"{name}: perte={loss:.4f}  perplexité={perplexity(loss):.3f}  bits/token={bits_per_char(loss):.3f}")


if __name__ == "__main__":
    main()
