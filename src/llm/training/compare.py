"""Comparaison contrôlée de variantes : mêmes données, mêmes seeds, même budget ; moyenne ± écart-type."""
import math
import time

import numpy as np
import torch

from .trainer import TrainConfig, train


def compare(variants, data, base_model=None, base_train=None, seeds=(0, 1), verbose=True, keep_models=False):
    """``variants`` : {nom: {"model": {...}, "train": {...}}} (ou directement des champs de ModelConfig).
    ``data`` : (train_tokens, val_tokens, vocab_size). Retourne {nom: résultats} et affiche un tableau.
    Les champs de ``variants`` écrasent ``base_model`` / ``base_train`` ; seul le facteur testé doit changer."""
    from llm.model import LM, ModelConfig  # noqa: F401  (noms disponibles dans la boîte à outils des notebooks)
    tr, va, vocab = data
    results = {}
    for name, spec in variants.items():
        spec = spec if ("model" in spec or "train" in spec) else {"model": spec}
        mkw = {**(base_model or {}), **spec.get("model", {})}
        tkw = {**(base_train or {}), **spec.get("train", {})}
        runs, models = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            model = LM(ModelConfig(vocab_size=vocab, **mkw))
            t0 = time.perf_counter()
            h = train(model, tr, va, TrainConfig(seed=seed, device="cpu", **tkw))
            h["wall"] = time.perf_counter() - t0
            h.pop("ema_model", None)
            runs.append(h); models.append(model if keep_models else None)
        vals = np.array([r["final_val_loss"] for r in runs])
        results[name] = dict(runs=runs, mean=vals.mean(), std=vals.std(ddof=1) if len(vals) > 1 else float("nan"),
                             params=runs[0]["params"], active=runs[0]["active_params"],
                             tok_s=float(np.mean([r["tokens_per_s"][-1] for r in runs])),
                             model_kw=mkw, train_kw=tkw, models=models)
    if verbose:
        table(results)
    return results


def table(results, ref=None):
    """Tableau : perte de validation (moy ± std), bits/token, écart à la référence, paramètres, débit."""
    ref = ref or next(iter(results))
    print(f"{'variante':<26}{'val (moy±std)':>18}{'bits/tok':>9}{'Δ vs réf':>10}{'params':>10}{'actifs':>10}{'tok/s':>9}")
    best = min(r["mean"] for r in results.values())
    for name, r in results.items():
        d = r["mean"] - results[ref]["mean"]
        star = " *" if r["mean"] == best else ""
        print(f"{name:<26}{r['mean']:>10.4f} ±{r['std']:.4f}{r['mean'] / math.log(2):>9.3f}{d:>+10.4f}{r['params']:>10,}{r['active']:>10,}{r['tok_s']:>9.0f}{star}")
    print("(* = meilleure moyenne ; un écart inférieur à ~2 écarts-types n'est pas concluant avec si peu de seeds)")


def plot_curves(results, key="val_loss", ax=None, title=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(6.5, 3.6))[1]
    for name, r in results.items():
        steps = r["runs"][0]["step"]
        arr = np.array([run[key] for run in r["runs"]])
        m = arr.mean(0)
        ax.plot(steps, m, marker="o", ms=3, label=name)
        if len(arr) > 1:
            ax.fill_between(steps, arr.min(0), arr.max(0), alpha=0.15)
    ax.set_xlabel("étape"); ax.set_ylabel(key); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax
