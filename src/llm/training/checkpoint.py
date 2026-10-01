"""Chargement d'un checkpoint produit par ``train`` : ``model`` = poids courants, ``ema`` = moyenne mobile (si activée)."""
import torch


def load_model(path, map_location="cpu", use_ema=False):
    from llm.model import LM, ModelConfig
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = LM(ModelConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["ema"] if use_ema and ck.get("ema") else ck["model"])
    return model, ck
