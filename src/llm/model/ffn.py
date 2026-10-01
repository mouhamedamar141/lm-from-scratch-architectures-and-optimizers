"""Réseaux feed-forward : ReLU/GELU/SiLU (2 matrices) et variantes à porte GLU (3 matrices)."""
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig

ACT = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu, "reglu": F.relu, "geglu": F.gelu, "swiglu": F.silu}
GATED = {"reglu", "geglu", "swiglu"}


def ffn_hidden(cfg: ModelConfig) -> int:
    """Largeur cachée. Variantes à porte : 2/3 de la largeur classique pour conserver ~ le même nombre de paramètres."""
    if cfg.ffn_type in GATED:
        h = int(2 * cfg.ffn_mult * cfg.n_embd / 3)
        m = cfg.ffn_multiple_of
        return ((h + m - 1) // m) * m
    return int(cfg.ffn_mult * cfg.n_embd)


class FeedForward(nn.Module):
    r"""Classique : W2 act(W1 x).   À porte : W2 (act(W_gate x) ⊙ W_up x)."""

    def __init__(self, cfg: ModelConfig, hidden: int = None):
        super().__init__()
        hidden = hidden or ffn_hidden(cfg)
        self.gated = cfg.ffn_type in GATED
        self.act = ACT[cfg.ffn_type]
        self.up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias) if self.gated else None
        self.down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        h = self.act(self.gate(x)) * self.up(x) if self.gated else self.act(self.up(x))
        return self.drop(self.down(h))
