"""Normalisations : LayerNorm et RMSNorm écrites à la main."""
import torch
from torch import nn


class LayerNorm(nn.Module):
    r"""LayerNorm : y = (x - mu) / sqrt(sigma^2 + eps) * gamma + beta."""

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps) * self.weight
        return y + self.bias if self.bias is not None else y


class RMSNorm(nn.Module):
    r"""RMSNorm : y = x / sqrt(mean(x^2) + eps) * gamma (pas de recentrage, pas de biais)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()  # statistiques en fp32 pour la stabilité numérique
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


def make_norm(kind: str, dim: int, eps: float = 1e-5, bias: bool = True) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm(dim, eps, bias)
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    raise ValueError(f"Normalisation inconnue : {kind}")
