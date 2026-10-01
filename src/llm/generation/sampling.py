"""Stratégies d'échantillonnage : greedy, température, top-k, top-p, min-p, pénalité de répétition."""
import torch


def apply_repetition_penalty(logits, generated, penalty: float):
    """Pénalise les tokens déjà générés (Keskar et al., 2019) : logit/penalty si > 0, logit*penalty sinon."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    logits = logits.clone()
    for b in range(logits.size(0)):
        ids = generated[b].unique()
        sel = logits[b, ids]
        logits[b, ids] = torch.where(sel > 0, sel / penalty, sel * penalty)
    return logits


def filter_logits(logits, top_k=None, top_p=None, min_p=None):
    """Masque (-inf) les candidats exclus par top-k, top-p (noyau) et min-p."""
    if top_k:
        k = min(int(top_k), logits.size(-1))
        logits = logits.masked_fill(logits < torch.topk(logits, k, dim=-1).values[..., -1, None], -torch.inf)
    if min_p:
        probs = logits.softmax(-1)
        logits = logits.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, -torch.inf)
    if top_p and 0 < top_p < 1:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cum = sl.softmax(-1).cumsum(-1)
        remove = cum - sl.softmax(-1) > top_p      # on garde le premier token qui dépasse p
        logits = logits.masked_fill(torch.zeros_like(remove).scatter(-1, si, remove), -torch.inf)
    return logits


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, greedy=False, generator=None):
    if greedy or temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = filter_logits(logits / temperature, top_k, top_p, min_p)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator)
