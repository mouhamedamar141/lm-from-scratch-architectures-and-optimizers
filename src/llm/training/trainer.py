"""Boucle d'entraînement complète : toutes les options sont des champs de ``TrainConfig``."""
from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .optimizers import build_optimizer
from .schedules import lr_factor


@dataclass
class TrainConfig:
    # budget
    steps: int = 200
    batch_size: int = 16
    grad_accum: int = 1               # micro-lots accumulés avant chaque mise à jour
    block_size: Optional[int] = None  # défaut : model.cfg.block_size
    # optimisation
    optimizer: str = "adamw"          # sgd | sgd_momentum | nesterov | rmsprop | adagrad | adam | adamw | lion | muon
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.9
    grad_clip: Optional[float] = 1.0
    # schedule
    schedule: str = "warmup_cosine"   # constant | linear | cosine | warmup_cosine | warmup_linear | inv_sqrt | wsd
    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    decay_frac: float = 0.2           # WSD : part finale du budget consacrée à la décroissance
    # numérique / performance
    dtype: str = "float32"            # float32 | bfloat16 | float16 (autocast)
    compile: bool = False             # torch.compile (nécessite un compilateur C++)
    grad_checkpointing: bool = False  # recalcul des blocs pendant la rétropropagation (économise la mémoire)
    z_loss: float = 0.0               # PaLM : coef * mean(logsumexp(logits)^2)
    label_smoothing: float = 0.0
    ema_decay: Optional[float] = None # moyenne mobile exponentielle des poids (évaluée à la place du modèle)
    # suivi
    eval_interval: int = 50
    eval_iters: int = 10
    log_interval: int = 0             # 0 = pas de log ; sinon affiche toutes les n étapes
    early_stop_patience: Optional[int] = None   # nombre d'évaluations sans amélioration de la val
    # persistance
    checkpoint: Optional[str] = None
    save_best_only: bool = True
    resume: bool = False
    seed: int = 0
    device: str = "auto"

    def to_dict(self):
        return asdict(self)


def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device, gen: Optional[torch.Generator] = None):
    """Fenêtres aléatoires (x, y) avec y = x décalé d'un token."""
    ix = torch.randint(0, len(tokens) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([tokens[i:i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens, batch_size, block_size, iters, device, seed: int = 1234):
    """Perte de validation sur des lots FIXES (même seed => mêmes lots pour tous les modèles comparés)."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(iters):
        x, y = get_batch(tokens, batch_size, block_size, device, gen)
        losses.append(model(x, y).loss.item())
    model.train(was_training)
    return float(np.mean(losses))


class EMA:
    def __init__(self, model, decay):
        self.decay, self.shadow = decay, copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def _autocast(dtype: str, device: str):
    if dtype == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                          dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16)


def train(model, train_tokens, val_tokens, cfg: TrainConfig, callback: Optional[Callable] = None):
    """Entraîne ``model`` et renvoie un dictionnaire ``history`` (listes alignées) + résumé final."""
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(device).train()
    block = cfg.block_size or model.cfg.block_size
    train_tokens, val_tokens = torch.as_tensor(train_tokens), torch.as_tensor(val_tokens)
    opt = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps, cfg.momentum)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda") if cfg.dtype == "float16" and device.startswith("cuda") else None
    fwd = torch.compile(model) if cfg.compile else model
    if cfg.grad_checkpointing:
        from torch.utils.checkpoint import checkpoint
        for blk in model.blocks:
            blk._orig_forward = blk.forward
            blk.forward = (lambda f: (lambda x, cache=None, start_pos=0: checkpoint(f, x, cache, start_pos, use_reentrant=False)))(blk._orig_forward)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    start, best, bad_evals = 0, float("inf"), 0
    hist = {k: [] for k in ("step", "train_loss", "val_loss", "grad_norm", "lr", "tokens_per_s", "aux_loss", "elapsed")}
    if cfg.resume and cfg.checkpoint and Path(cfg.checkpoint).exists():
        ck = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start, best, hist = ck["step"], ck.get("best", best), ck.get("history", hist)
        if ema and ck.get("ema"):
            ema.shadow.load_state_dict(ck["ema"])
        if ck.get("data_rng") is not None:
            gen.set_state(ck["data_rng"].cpu())   # les lots suivants sont ceux qu'aurait vus l'entraînement continu
        print(f"Reprise à l'étape {start}")

    t0, tok_count, run_loss, run_n, run_aux = time.perf_counter(), 0, 0.0, 0, 0.0
    eval_model = ema.shadow if ema else model
    for step in range(start, cfg.steps):
        f = lr_factor(cfg.schedule, step, cfg.steps, cfg.warmup_steps, cfg.min_lr_ratio, cfg.decay_frac)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * f
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_tokens, cfg.batch_size, block, device, gen)
            with _autocast(cfg.dtype, device):
                out = fwd(x, y)
                loss = out.loss
                if cfg.label_smoothing > 0:
                    lp = out.logits.float().log_softmax(-1)
                    smooth = -lp.mean(-1).mean()
                    loss = (1 - cfg.label_smoothing) * loss + cfg.label_smoothing * smooth
                if cfg.z_loss > 0:
                    loss = loss + cfg.z_loss * torch.logsumexp(out.logits.float(), -1).pow(2).mean()
                total = (loss + out.aux_loss) / cfg.grad_accum
            (scaler.scale(total) if scaler else total).backward()
            run_loss += out.loss.item(); run_aux += float(out.aux_loss.detach()); run_n += 1
            tok_count += x.numel()
        if scaler:
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip else float("inf"))
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        if ema:
            ema.update(model)
        s = step + 1
        if cfg.log_interval and s % cfg.log_interval == 0:
            print(f"étape {s:5d} | perte {run_loss / run_n:.4f} | |g| {float(gn):.2f} | lr {opt.param_groups[0]['lr']:.2e}")
        if s % cfg.eval_interval == 0 or s == cfg.steps or s == 1:
            v = evaluate(eval_model, val_tokens, cfg.batch_size, block, cfg.eval_iters, device)
            el = time.perf_counter() - t0
            for k, val in zip(hist, (s, run_loss / run_n, v, float(gn), opt.param_groups[0]["lr"],
                                     tok_count / max(el, 1e-9), run_aux / run_n, el)):
                hist[k].append(val)
            run_loss, run_n, run_aux = 0.0, 0, 0.0
            if callback:
                callback(s, hist)
            improved = v < best
            best = min(best, v)
            bad_evals = 0 if improved else bad_evals + 1
            if cfg.checkpoint and (improved or not cfg.save_best_only):
                Path(cfg.checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict() if ema else None,
                            "optimizer": opt.state_dict(), "step": s, "best": best, "data_rng": gen.get_state(),
                            "history": hist, "config": model.cfg.to_dict(), "train_config": cfg.to_dict()}, cfg.checkpoint)
            if cfg.early_stop_patience and bad_evals >= cfg.early_stop_patience:
                print(f"Arrêt anticipé à l'étape {s}")
                break
    if cfg.grad_checkpointing:
        for blk in model.blocks:
            blk.forward = blk._orig_forward
    hist["final_val_loss"] = hist["val_loss"][-1]
    hist["best_val_loss"] = best
    hist["params"] = model.num_parameters()
    hist["active_params"] = model.num_parameters(active_only=True)
    hist["ema_model"] = ema.shadow if ema else None
    return hist
