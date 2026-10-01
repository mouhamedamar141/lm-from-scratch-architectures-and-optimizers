"""Génération autorégressive avec ou sans KV cache."""
import torch

from .sampling import apply_repetition_penalty, sample_next_token


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_p=None, greedy=False,
             repetition_penalty=1.0, use_cache=True, generator=None, eos_id=None):
    """Étend ``idx`` (B, T). Avec ``use_cache`` : préremplissage du prompt puis un token à la fois (O(T) par pas)."""
    model.eval()
    cache = model.new_cache() if use_cache else None
    cur = idx
    for _ in range(max_new_tokens):
        if use_cache:
            inp = cur if cache.n_seen == 0 else cur[:, -1:]
            logits = model(inp, cache=cache).logits[:, -1]
        else:
            logits = model(cur[:, -model.cfg.block_size:]).logits[:, -1]
        logits = apply_repetition_penalty(logits, cur, repetition_penalty)
        nxt = sample_next_token(logits, temperature, top_k, top_p, min_p, greedy, generator)
        cur = torch.cat([cur, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
    return cur
