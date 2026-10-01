"""Attention causale : MHA, MQA, GQA et MLA, avec RoPE/ALiBi, QK-Norm, fenêtre glissante et KV cache."""
import math
import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .norms import RMSNorm
from .positions import alibi_bias, alibi_slopes, apply_rope, rope_inv_freq


def build_bias(q_pos, k_pos, window=None, slopes=None, dtype=torch.float32):
    """Biais additif (Tq, Tk) ou (H, Tq, Tk) : -inf hors masque causal/fenêtre, ALiBi éventuel."""
    allowed = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        allowed &= (q_pos[:, None] - k_pos[None, :]) < window
    bias = torch.zeros(allowed.shape, dtype=dtype, device=q_pos.device).masked_fill(~allowed, float("-inf"))
    if slopes is not None:
        bias = bias[None] + alibi_bias(slopes, q_pos, k_pos).to(dtype)
    return bias


def attend(q, k, v, bias, scale, dropout_p=0.0, training=False, use_sdpa=False):
    """softmax(q k^T * scale + bias) v ; q: (B,H,Tq,dq), k: (B,H,Tk,dq), v: (B,H,Tk,dv)."""
    if use_sdpa:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=dropout_p if training else 0.0, scale=scale)
    scores = (q @ k.transpose(-2, -1)) * scale + bias
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    weights = F.dropout(weights, dropout_p, training)
    return weights @ v


class GroupedQueryAttention(nn.Module):
    """MHA (n_kv_head = n_head), GQA (1 < n_kv_head < n_head) et MQA (n_kv_head = 1) en une seule classe."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.hd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv * self.hd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv * self.hd, bias=cfg.bias)
        self.out_proj = nn.Linear(self.n_head * self.hd, cfg.n_embd, bias=cfg.bias)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.hd, cfg.norm_eps) if cfg.qk_norm else None
        self.use_rope = cfg.position_mode == "rope"
        if self.use_rope:
            orig = cfg.rope_orig_len or cfg.block_size
            inv, self.mscale = rope_inv_freq(self.hd, cfg.rope_theta, cfg.rope_scaling, cfg.rope_factor, orig)
            self.register_buffer("inv_freq", inv, persistent=False)
        if cfg.position_mode == "alibi":
            self.register_buffer("slopes", alibi_slopes(self.n_head), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None, layer=0, start_pos=0):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q_pos = torch.arange(start_pos, start_pos + t, device=x.device)
        if self.use_rope:
            q = apply_rope(q, q_pos, self.inv_freq, self.mscale)
            k = apply_rope(k, q_pos, self.inv_freq, self.mscale)  # K est mis en cache APRÈS rotation
        if cache is not None:
            full = cache.update(layer, 2, k=k, v=v)
            k, v = full["k"], full["v"]
        tk = k.size(2)
        k_pos = torch.arange(start_pos + t - tk, start_pos + t, device=x.device)
        if self.n_kv != self.n_head:  # partage des têtes K/V entre groupes de requêtes
            rep = self.n_head // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        bias = build_bias(q_pos, k_pos, self.cfg.window, getattr(self, "slopes", None), q.dtype)
        out = attend(q, k, v, bias, 1.0 / math.sqrt(self.hd), self.cfg.dropout, self.training, self.cfg.sdpa)
        return self.drop(self.out_proj(out.transpose(1, 2).reshape(b, t, -1)))


class MultiHeadLatentAttention(nn.Module):
    """MLA (DeepSeek-V2) : K et V sont reconstruits depuis un latent c_KV de dimension r ;
    seuls c_KV (r) et la clé RoPE partagée (d_r) sont mis en cache, soit r + d_r valeurs par token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.hd = cfg.n_head, cfg.head_dim
        self.r = cfg.kv_lora_rank
        self.use_rope = cfg.position_mode == "rope"
        self.dr = cfg.rope_head_dim if self.use_rope else 0
        d = cfg.n_embd
        if cfg.q_lora_rank:
            self.q_down = nn.Linear(d, cfg.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
            self.q_up = nn.Linear(cfg.q_lora_rank, self.n_head * (self.hd + self.dr), bias=False)
        else:
            self.q_up = nn.Linear(d, self.n_head * (self.hd + self.dr), bias=False)
            self.q_down = None
        self.kv_down = nn.Linear(d, self.r, bias=False)         # W^{DKV}
        self.kv_norm = RMSNorm(self.r, cfg.norm_eps)
        self.k_up = nn.Linear(self.r, self.n_head * self.hd, bias=False)  # W^{UK}
        self.v_up = nn.Linear(self.r, self.n_head * self.hd, bias=False)  # W^{UV}
        if self.use_rope:
            self.k_rope = nn.Linear(d, self.dr, bias=False)     # clé RoPE partagée entre les têtes
            inv, self.mscale = rope_inv_freq(self.dr, cfg.rope_theta, cfg.rope_scaling, cfg.rope_factor,
                                             cfg.rope_orig_len or cfg.block_size)
            self.register_buffer("inv_freq", inv, persistent=False)
        if cfg.position_mode == "alibi":
            self.register_buffer("slopes", alibi_slopes(self.n_head), persistent=False)
        self.out_proj = nn.Linear(self.n_head * self.hd, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None, layer=0, start_pos=0):
        b, t, _ = x.shape
        H, hd, dr = self.n_head, self.hd, self.dr
        q_in = self.q_norm(self.q_down(x)) if self.q_down is not None else x
        q = self.q_up(q_in).view(b, t, H, hd + dr).transpose(1, 2)
        q_nope, q_rope = q[..., :hd], q[..., hd:]
        c = self.kv_norm(self.kv_down(x))                         # (B, T, r) : ce qu'on met en cache
        kr = self.k_rope(x) if self.use_rope else None            # (B, T, dr)
        q_pos = torch.arange(start_pos, start_pos + t, device=x.device)
        if self.use_rope:
            q_rope = apply_rope(q_rope, q_pos, self.inv_freq, self.mscale)
            kr = apply_rope(kr, q_pos, self.inv_freq, self.mscale)
        if cache is not None:
            tensors = {"c": c} if kr is None else {"c": c, "kr": kr}
            full = cache.update(layer, 1, **tensors)
            c, kr = full["c"], full.get("kr")
        tk = c.size(1)
        k_pos = torch.arange(start_pos + t - tk, start_pos + t, device=x.device)
        k_nope = self.k_up(c).view(b, tk, H, hd).transpose(1, 2)  # reconstruction de K depuis le latent
        v = self.v_up(c).view(b, tk, H, hd).transpose(1, 2)
        if self.use_rope:
            k = torch.cat([k_nope, kr[:, None].expand(b, H, tk, dr)], dim=-1)
            q = torch.cat([q_nope, q_rope], dim=-1)
        else:
            k, q = k_nope, q_nope
        bias = build_bias(q_pos, k_pos, self.cfg.window, getattr(self, "slopes", None), q.dtype)
        out = attend(q, k, v, bias, 1.0 / math.sqrt(hd + dr), self.cfg.dropout, self.training, self.cfg.sdpa)
        return self.drop(self.out_proj(out.transpose(1, 2).reshape(b, t, -1)))


def make_attention(cfg: ModelConfig) -> nn.Module:
    return MultiHeadLatentAttention(cfg) if cfg.attn_type == "mla" else GroupedQueryAttention(cfg)


def kv_cache_floats_per_token(cfg: ModelConfig) -> int:
    """Nombre de valeurs mises en cache par token et par couche (formule théorique)."""
    if cfg.attn_type == "mla":
        return cfg.kv_lora_rank + (cfg.rope_head_dim if cfg.position_mode == "rope" else 0)
    return 2 * cfg.n_kv_head * cfg.head_dim
