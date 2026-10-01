r"""Optimiseurs écrits à la main (SGD, RMSprop, Adagrad, Adam, AdamW, Lion, Muon).

Notation : g_t gradient, theta_t paramètres, eta le taux d'apprentissage, lambda le weight decay.
Chaque classe documente sa règle de mise à jour ; les tests comparent SGD/Adam/AdamW à torch.optim.
"""
import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    r"""SGD avec momentum (Polyak) ou Nesterov ; le weight decay est de type L2 (ajouté au gradient).

    g <- g + lambda*theta ;  v <- mu*v + g ;  theta <- theta - eta * v
    Nesterov : theta <- theta - eta * (g + mu*v).
    """

    def __init__(self, params, lr=1e-2, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if g["weight_decay"]:
                    d = d + g["weight_decay"] * p
                if g["momentum"]:
                    st = self.state[p]
                    buf = st.setdefault("v", torch.zeros_like(p))
                    buf.mul_(g["momentum"]).add_(d)
                    d = d + g["momentum"] * buf if g["nesterov"] else buf
                p.add_(d, alpha=-g["lr"])


class RMSprop(Optimizer):
    r"""s <- alpha*s + (1-alpha) g^2 ;  theta <- theta - eta * g / (sqrt(s) + eps)."""

    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad + g["weight_decay"] * p if g["weight_decay"] else p.grad
                s = self.state[p].setdefault("s", torch.zeros_like(p))
                s.mul_(g["alpha"]).addcmul_(d, d, value=1 - g["alpha"])
                p.addcdiv_(d, s.sqrt().add_(g["eps"]), value=-g["lr"])


class Adagrad(Optimizer):
    r"""G <- G + g^2 ;  theta <- theta - eta * g / (sqrt(G) + eps)."""

    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad + g["weight_decay"] * p if g["weight_decay"] else p.grad
                G = self.state[p].setdefault("G", torch.zeros_like(p))
                G.addcmul_(d, d)
                p.addcdiv_(d, G.sqrt().add_(g["eps"]), value=-g["lr"])


class Adam(Optimizer):
    r"""Adam ; ``decoupled=False`` -> L2 classique, ``decoupled=True`` -> AdamW (weight decay découplé).

    m <- b1 m + (1-b1) g ;  v <- b2 v + (1-b2) g^2
    m^ = m/(1-b1^t) ;  v^ = v/(1-b2^t)
    Adam  : g <- g + lambda*theta AVANT les moments.
    AdamW : theta <- theta - eta*lambda*theta  À PART, puis  theta <- theta - eta * m^/(sqrt(v^)+eps).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st["t"] += 1
                d = p.grad
                if g["weight_decay"] and not g["decoupled"]:
                    d = d + g["weight_decay"] * p
                st["m"].mul_(b1).add_(d, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(d, d, value=1 - b2)
                m_hat = st["m"] / (1 - b1 ** st["t"])
                v_hat = st["v"] / (1 - b2 ** st["t"])
                if g["weight_decay"] and g["decoupled"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.addcdiv_(m_hat, v_hat.sqrt().add_(g["eps"]), value=-g["lr"])


class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay, decoupled=True)


class Lion(Optimizer):
    r"""Lion (Chen et al. 2023) : update = sign(b1 m + (1-b1) g) ;  m <- b2 m + (1-b2) g.
    theta <- theta - eta (update + lambda theta). Prendre lr ~ 3-10x plus petit qu'AdamW."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("m", torch.zeros_like(p))
                update = (b1 * m + (1 - b1) * p.grad).sign()
                p.add_(update + g["weight_decay"] * p, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def newton_schulz(G, steps=5, eps=1e-7):
    r"""Orthogonalise G (approximation de U V^T de sa SVD) par itération de Newton-Schulz quintique."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X.t() if transposed else X


class Muon(Optimizer):
    r"""Muon (Jordan 2024) pour les matrices 2D des couches cachées : momentum puis orthogonalisation.

    m <- mu m + g ;  update = NS(g + mu m) ;  theta <- theta - eta * scale * update
    scale = 0.2 sqrt(max(n, m)) (Liu et al. 2025) aligne le RMS de l'update sur celui d'AdamW.
    Les autres paramètres (embeddings, normes, vecteurs) doivent utiliser AdamW (groupe distinct).
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.0, ns_steps=5, match_adamw_rms=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      ns_steps=ns_steps, match_adamw_rms=match_adamw_rms))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon ne s'applique qu'aux matrices 2D")
                buf = self.state[p].setdefault("m", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = newton_schulz(p.grad + g["momentum"] * buf, g["ns_steps"]).to(p.dtype)
                if g["match_adamw_rms"]:
                    upd = upd * 0.2 * max(p.shape) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd, alpha=-g["lr"])


OPTIMIZERS = ("sgd", "sgd_momentum", "nesterov", "rmsprop", "adagrad", "adam", "adamw", "lion", "muon")


def split_decay(model):
    """Sépare les paramètres : matrices (decay) / biais, normes, embeddings (sans decay)."""
    decay, no_decay = [], []
    seen = set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_matrix = p.ndim >= 2 and "emb" not in n and "norm" not in n and "route_bias" not in n
        (decay if is_matrix else no_decay).append((n, p))
    return decay, no_decay


def build_optimizer(model, name="adamw", lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8,
                    momentum=0.9, muon_lr=None):
    """Fabrique un optimiseur avec groupes decay/no-decay (le weight decay ne s'applique pas aux normes/biais/embeddings)."""
    name = name.lower()
    decay, no_decay = split_decay(model)
    d, nd = [p for _, p in decay], [p for _, p in no_decay]
    if name == "muon":  # Muon sur les matrices cachées, AdamW sur le reste
        hidden = [p for n, p in decay if "lm_head" not in n]
        rest = [p for n, p in decay if "lm_head" in n] + nd
        opt = Muon(hidden, lr=muon_lr or lr * 10, weight_decay=weight_decay)
        opt2 = AdamW(rest, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        return _Combined(opt, opt2)
    groups = [{"params": d, "weight_decay": weight_decay}, {"params": nd, "weight_decay": 0.0}]
    if name in ("sgd", "sgd_momentum", "nesterov"):
        mom = 0.0 if name == "sgd" else momentum
        return SGD(groups, lr=lr, momentum=mom, nesterov=name == "nesterov")
    if name == "rmsprop":
        return RMSprop(groups, lr=lr, eps=eps)
    if name == "adagrad":
        return Adagrad(groups, lr=lr)
    if name == "adam":
        return Adam(groups, lr=lr, betas=betas, eps=eps)
    if name == "adamw":
        return AdamW(groups, lr=lr, betas=betas, eps=eps)
    if name == "lion":
        return Lion(groups, lr=lr, betas=(0.9, 0.99))
    raise ValueError(f"Optimiseur inconnu : {name} (choix : {OPTIMIZERS})")


class _Combined:
    """Enveloppe deux optimiseurs (ex. Muon + AdamW) derrière l'interface habituelle."""

    def __init__(self, *opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["opts"]):
            o.load_state_dict(s)
