r"""Schedules de taux d'apprentissage : facteur multiplicatif f(t) tel que eta_t = eta_max * f(t)."""
import math

SCHEDULES = ("constant", "linear", "cosine", "warmup_cosine", "warmup_linear", "inv_sqrt", "wsd")


def lr_factor(name: str, step: int, total: int, warmup: int = 0, min_ratio: float = 0.1, decay_frac: float = 0.2) -> float:
    """``step`` commence à 0. Le warmup linéaire (step+1)/warmup s'applique à tous les schedules « warmup_* », inv_sqrt et wsd."""
    if name not in SCHEDULES:
        raise ValueError(f"Schedule inconnu : {name} (choix : {SCHEDULES})")
    if name == "constant":
        return 1.0
    if name == "linear":
        return 1.0 - (1.0 - min_ratio) * min(1.0, step / max(1, total))
    if name == "cosine":
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if name == "inv_sqrt":
        return math.sqrt(max(1, warmup) / (step + 1))
    if name == "wsd":  # Warmup - Stable - Decay (MiniCPM) : plateau puis décroissance finale
        decay_start = int(total * (1 - decay_frac))
        if step < decay_start:
            return 1.0
        p = min(1.0, (step - decay_start) / max(1, total - decay_start))
        return min_ratio + (1 - min_ratio) * (1 - p)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    if name == "warmup_linear":
        return 1.0 - (1.0 - min_ratio) * p
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))  # warmup_cosine
