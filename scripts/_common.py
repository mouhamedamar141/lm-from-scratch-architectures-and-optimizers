"""Fonctions partagées par les scripts : construction du modèle et des données depuis un YAML."""
from llm.data.preprocessing import load_tokens
from llm.model import LM, ModelConfig
from llm.tokenizer.tokenizer import load_tokenizer
from llm.training import TrainConfig
from llm.utils.config import load_config


def build_from_yaml(path, overrides=None):
    cfg = load_config(path)
    tok = load_tokenizer(cfg["data"]["tokenizer"])
    model_cfg = ModelConfig(vocab_size=tok.vocab_size, **{**cfg["model"], **(overrides or {})})
    train_kw = dict(cfg["training"])
    if "betas" in train_kw:
        train_kw["betas"] = tuple(train_kw["betas"])
    train_cfg = TrainConfig(seed=cfg.get("seed", 0), device=cfg.get("device", "auto"), **train_kw)
    return cfg, tok, LM(model_cfg), train_cfg, load_tokens(cfg["data"]["train_tokens"]), load_tokens(cfg["data"]["val_tokens"])
