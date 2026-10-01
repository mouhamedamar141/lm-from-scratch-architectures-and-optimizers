"""Mixture of Experts : routage top-k (softmax ou sigmoïde + biais), Expert Choice, experts partagés."""
import math
import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .ffn import FeedForward, ffn_hidden


class MoE(nn.Module):
    """Couche MoE remplaçant le FFN. ``forward`` renvoie (sortie, perte_auxiliaire) ;
    les statistiques de charge sont dans ``self.stats`` (détachées du graphe)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.expert_hidden or ffn_hidden(cfg)
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(cfg, hidden) for _ in range(cfg.n_experts)])
        self.shared = nn.ModuleList([FeedForward(cfg, hidden) for _ in range(cfg.n_shared_experts)])
        self.register_buffer("route_bias", torch.zeros(cfg.n_experts), persistent=True)
        self.stats = {}

    def forward(self, x):
        cfg = self.cfg
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        n_tok = flat.size(0)
        logits = self.router(flat).float()                                  # (N, E)
        aux = flat.new_zeros(())
        if cfg.router == "expert_choice":
            out, load, dropped = self._expert_choice(flat, logits)
        else:
            if cfg.router == "softmax":
                probs = logits.softmax(-1)
                top_w, top_i = probs.topk(cfg.top_k, dim=-1)
                top_w = top_w / top_w.sum(-1, keepdim=True)                 # renormalisation (Mixtral)
            else:  # sigmoid_bias : le biais sert à SÉLECTIONNER, pas à pondérer (DeepSeek-V3)
                scores = logits.sigmoid()
                top_i = (scores + self.route_bias).topk(cfg.top_k, dim=-1).indices
                top_w = scores.gather(-1, top_i)
                top_w = top_w / top_w.sum(-1, keepdim=True)
                probs = scores / scores.sum(-1, keepdim=True)
            out, load, dropped = self._dispatch(flat, top_i, top_w)
            if self.training and cfg.router == "softmax" and cfg.balance_coef > 0:
                # Switch : E * sum_e f_e * P_e   (f = part de tokens envoyés à e, P = probabilité moyenne)
                f = load / max(1, n_tok * cfg.top_k)
                aux = aux + cfg.balance_coef * cfg.n_experts * (f.detach() * probs.mean(0)).sum()
            if self.training and cfg.router == "sigmoid_bias":
                with torch.no_grad():   # sous-chargé -> biais augmente ; surchargé -> biais diminue
                    err = load.mean() - load
                    self.route_bias += cfg.bias_update_speed * torch.sign(err)
        if self.training and cfg.z_loss_coef > 0:
            aux = aux + cfg.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        for s in self.shared:
            out = out + s(flat)
        self.stats = {"load": (load / load.sum().clamp(min=1)).detach(), "dropped": float(dropped)}
        return out.view(b, t, d), aux

    def _dispatch(self, flat, top_i, top_w):
        cfg, n_tok = self.cfg, flat.size(0)
        out = torch.zeros_like(flat)
        load = torch.zeros(cfg.n_experts, device=flat.device)
        cap = None if cfg.capacity_factor is None else math.ceil(cfg.capacity_factor * n_tok * cfg.top_k / cfg.n_experts)
        n_dropped = 0
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if cap is not None and tok.numel() > cap:       # capacité dépassée : les derniers tokens sont abandonnés
                n_dropped += tok.numel() - cap
                tok, slot = tok[:cap], slot[:cap]
            load[e] = tok.numel()
            if tok.numel():
                out.index_add_(0, tok, expert(flat[tok]) * top_w[tok, slot, None].to(flat.dtype))
        return out, load, n_dropped / max(1, n_tok * self.cfg.top_k)

    def _expert_choice(self, flat, logits):
        """Chaque expert choisit ses c meilleurs tokens : charge parfaitement équilibrée,
        mais NON causal (le choix dépend des autres tokens du lot) -> inadapté au décodage autorégressif."""
        cfg, n_tok = self.cfg, flat.size(0)
        cap = max(1, int(n_tok * cfg.top_k / cfg.n_experts))
        probs = logits.softmax(-1)
        w, idx = probs.t().topk(cap, dim=-1)                # (E, c)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            out.index_add_(0, idx[e], expert(flat[idx[e]]) * w[e, :, None].to(flat.dtype))
        covered = torch.zeros(n_tok, device=flat.device)
        covered[idx.reshape(-1)] = 1.0
        load = torch.full((cfg.n_experts,), float(cap), device=flat.device)
        return out, load, 1.0 - covered.mean().item()       # « dropped » = tokens sans aucun expert
