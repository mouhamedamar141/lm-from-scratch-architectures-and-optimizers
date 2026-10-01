"""Bloc Transformer (Pre-LN ou Post-LN) avec FFN dense ou MoE."""
from torch import nn

from .attention import make_attention
from .config import ModelConfig
from .ffn import FeedForward
from .moe import MoE
from .norms import make_norm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.norm1 = make_norm(cfg.norm_type, cfg.n_embd, cfg.norm_eps, cfg.bias)
        self.norm2 = make_norm(cfg.norm_type, cfg.n_embd, cfg.norm_eps, cfg.bias)
        self.attn = make_attention(cfg)
        self.is_moe = cfg.moe and layer_idx >= cfg.n_dense_layers
        self.ffn = MoE(cfg) if self.is_moe else FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def _ffn(self, x):
        if self.is_moe:
            return self.ffn(x)
        return self.ffn(x), x.new_zeros(())

    def forward(self, x, cache=None, start_pos=0):
        if self.cfg.pre_norm:   # x + F(norm(x)) : chemin résiduel « propre »
            x = x + self.drop(self.attn(self.norm1(x), cache, self.layer_idx, start_pos))
            y, aux = self._ffn(self.norm2(x))
            return x + self.drop(y), aux
        x = self.norm1(x + self.drop(self.attn(x, cache, self.layer_idx, start_pos)))  # Post-LN
        y, aux = self._ffn(x)
        return self.norm2(x + self.drop(y)), aux
