"""Génère du texte depuis un checkpoint (options : température, top-k/p, min-p, pénalité, cache KV)."""
import argparse

import torch

from _common import build_from_yaml
from llm.generation import generate
from llm.training.checkpoint import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--prompt", default="Le langage")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args()
    cfg, tok, _, tcfg, *_ = build_from_yaml(a.config)
    model, _ = load_model(a.checkpoint or tcfg.checkpoint)
    ids = torch.tensor([tok.encode(a.prompt)])
    out = generate(model, ids, a.tokens, a.temperature, a.top_k, a.top_p, a.min_p, a.greedy,
                   a.repetition_penalty, use_cache=not a.no_cache)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
