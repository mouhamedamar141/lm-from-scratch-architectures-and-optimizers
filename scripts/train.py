"""Entraîne le modèle décrit par un YAML (toutes les options de TrainConfig sont disponibles)."""
import argparse
import json
from pathlib import Path

from _common import build_from_yaml
from llm.training import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg, tok, model, tcfg, tr, va = build_from_yaml(a.config)
    tcfg.resume = a.resume or tcfg.resume
    h = train(model, tr, va, tcfg, callback=lambda s, h: print(f"étape {s:5d} | val {h['val_loss'][-1]:.4f} | {h['tokens_per_s'][-1]:.0f} tok/s", flush=True))
    out = Path("experiments/results") / (Path(tcfg.checkpoint or "run").stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in h.items() if k != "ema_model"}, indent=1))
    print(f"Paramètres : {h['params']:,} (actifs : {h['active_params']:,}) ; meilleure val : {h['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
