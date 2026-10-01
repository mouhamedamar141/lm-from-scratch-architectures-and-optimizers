"""Ablation contrôlée : varie UN champ du modèle ou de l'entraînement, mêmes seeds/budget, écrit JSON + CSV."""
import argparse
import csv
import json
from pathlib import Path

from _common import build_from_yaml
from llm.model import LM, ModelConfig
from llm.training import TrainConfig, train
from llm.utils.metrics import perplexity


def parse(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--field", required=True, help="ex. position_mode, attn_type, ffn_type, moe, optimizer, schedule, lr...")
    p.add_argument("--values", required=True, help="valeurs séparées par des virgules")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output", default="experiments/results/ablation.json")
    a = p.parse_args()
    cfg, tok, base, tcfg, tr, va = build_from_yaml(a.config)
    model_fields, train_fields = set(base.cfg.to_dict()), set(tcfg.to_dict())
    rows = []
    for value in map(parse, a.values.split(",")):
        for seed in map(int, a.seeds.split(",")):
            mc, tc = base.cfg.to_dict(), tcfg.to_dict()
            (mc if a.field in model_fields else tc)[a.field] = value
            if a.field not in model_fields | train_fields:
                raise SystemExit(f"Champ inconnu : {a.field}")
            if a.field == "attn_type":
                mc["n_kv_head"] = None   # sera recalculé par ModelConfig (mha/mqa/gqa)
            tc.update(seed=seed, checkpoint=None, **({"steps": a.steps} if a.steps else {}))
            tc["betas"] = tuple(tc["betas"])
            h = train(LM(ModelConfig.from_dict(mc)), tr, va, TrainConfig(**tc))
            row = dict(field=a.field, value=value, seed=seed, val_loss=h["final_val_loss"], perplexity=perplexity(h["final_val_loss"]),
                       params=h["params"], active_params=h["active_params"], tokens_per_s=h["tokens_per_s"][-1])
            print(row, flush=True); rows.append(row)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
