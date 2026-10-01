"""Encodages de position : sinusoïdal, appris, RoPE (+ extensions de contexte) et ALiBi."""
import math
import torch
from torch import nn


def sinusoidal_table(n_pos: int, dim: int) -> torch.Tensor:
    r"""PE(p, 2i) = sin(p / 10000^{2i/d}),  PE(p, 2i+1) = cos(p / 10000^{2i/d})."""
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(n_pos, dim)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return table


class AbsolutePositions(nn.Module):
    """Positions absolues ajoutées aux embeddings : « learned » ou « sinusoidal »."""

    def __init__(self, kind: str, max_len: int, dim: int):
        super().__init__()
        self.kind, self.max_len = kind, max_len
        if kind == "learned":
            self.emb = nn.Embedding(max_len, dim)
        else:
            self.register_buffer("table", sinusoidal_table(max_len, dim), persistent=False)

    def forward(self, x, start_pos: int = 0):
        t = x.size(1)
        if start_pos + t > self.max_len:
            raise ValueError(f"Position {start_pos + t} > longueur maximale {self.max_len} (positions absolues)")
        pos = torch.arange(start_pos, start_pos + t, device=x.device)
        pe = self.emb(pos) if self.kind == "learned" else self.table[pos].to(x.dtype)
        return x + pe[None]


def rope_inv_freq(dim: int, theta: float = 10000.0, scaling=None, factor: float = 1.0,
                  orig_len: int = 64, alpha: float = 1.0, beta: float = 32.0):
    r"""Fréquences RoPE  theta_i = theta^{-2i/d}  avec extension de contexte optionnelle.

    - ``linear`` (Position Interpolation) : positions divisées par s.
    - ``ntk``    (NTK-aware)              : base theta' = theta * s^{d/(d-2)}.
    - ``yarn``   (YaRN)                   : interpolation par bande de fréquences + facteur de température.
    Retourne (inv_freq, mscale) ; mscale multiplie cos/sin (température d'attention de YaRN).
    """
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    mscale = 1.0
    if scaling is None or factor == 1.0:
        return inv, mscale
    if scaling == "linear":
        return inv / factor, mscale
    if scaling == "ntk":
        new_theta = theta * factor ** (dim / (dim - 2))
        return 1.0 / (new_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), mscale
    if scaling == "yarn":
        rotations = orig_len * inv / (2 * math.pi)            # nombre de tours sur la longueur d'origine
        gamma = ((rotations - alpha) / (beta - alpha)).clamp(0.0, 1.0)  # 0 = interpoler, 1 = garder
        inv = (1 - gamma) * inv / factor + gamma * inv
        return inv, 0.1 * math.log(factor) + 1.0
    raise ValueError(f"rope_scaling inconnu : {scaling}")


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, positions, inv_freq, mscale: float = 1.0):
    r"""Applique la rotation de RoPE à ``x`` de forme (..., T, d) ; ``positions`` de forme (T,).

    Convention « moitiés » (LLaMA/HF) : la paire (i, i + d/2) est tournée de l'angle p * theta_i.
    """
    angles = positions.float()[:, None] * inv_freq.to(x.device)[None, :]
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos() * mscale, emb.sin() * mscale
    return (x * cos.to(x.dtype)) + (rotate_half(x) * sin.to(x.dtype))


def alibi_slopes(n_head: int) -> torch.Tensor:
    r"""Pentes géométriques m_h = 2^{-8h/n} (h = 1..n), généralisées aux n non puissances de 2."""
    def power_of_2(n):
        start = 2 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]
    if math.log2(n_head).is_integer():
        return torch.tensor(power_of_2(n_head))
    closest = 2 ** math.floor(math.log2(n_head))
    extra = power_of_2(2 * closest)[0::2][: n_head - closest]
    return torch.tensor(power_of_2(closest) + extra)


def alibi_bias(slopes, q_pos, k_pos):
    r"""Biais additif ALiBi : -m_h * (i - j) pour j <= i. Forme (H, Tq, Tk)."""
    dist = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).float()
    return -slopes[:, None, None].to(dist.device) * dist[None]
