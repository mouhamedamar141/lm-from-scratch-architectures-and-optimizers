"""Modèle de langage décodeur : GPT-2, LLaMA, Mistral, Mixtral, DeepSeek... selon la configuration."""
import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .block import Block
from .cache import KVCache
from .config import ModelConfig
from .norms import make_norm
from .positions import AbsolutePositions


class Output(NamedTuple):
    logits: torch.Tensor
    loss: Optional[torch.Tensor]      # entropie croisée seule (comparable entre modèles)
    aux_loss: torch.Tensor            # pertes auxiliaires MoE (équilibrage, z-loss)


class LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = AbsolutePositions(cfg.position_mode, cfg.block_size, cfg.n_embd) \
            if cfg.position_mode in ("learned", "sinusoidal") else None
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.final_norm = make_norm(cfg.norm_type, cfg.n_embd, cfg.norm_eps, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)
        for name, p in self.named_parameters():   # init GPT-2 : sorties résiduelles / sqrt(2 L)
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, cache: Optional[KVCache] = None) -> Output:
        b, t = idx.shape
        start = cache.n_seen if cache is not None else 0
        if cache is None and t > self.cfg.block_size and self.cfg.position_mode in ("learned", "sinusoidal"):
            raise ValueError("Séquence plus longue que block_size (positions absolues)")
        x = self.tok_emb(idx)
        if self.cfg.emb_scale:
            x = x * math.sqrt(self.cfg.n_embd)
        if self.pos is not None:
            x = self.pos(x, start)
        x = self.drop(x)
        aux = x.new_zeros(())
        for blk in self.blocks:
            x, a = blk(x, cache, start)
            aux = aux + a
        if cache is not None:
            cache.n_seen += t
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return Output(logits, loss, aux)

    # --- utilitaires ---
    def num_parameters(self, active_only: bool = False) -> int:
        """Paramètres uniques. ``active_only`` : compte seulement top_k experts (+ partagés) par couche MoE."""
        total = sum(p.numel() for p in self.parameters())
        if active_only and self.cfg.moe:
            for blk in self.blocks:
                if blk.is_moe:
                    per = sum(p.numel() for p in blk.ffn.experts[0].parameters())
                    total -= per * (self.cfg.n_experts - self.cfg.top_k)
        return total

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.n_layer, self.cfg.window)

    def moe_stats(self):
        return [b.ffn.stats for b in self.blocks if b.is_moe]
