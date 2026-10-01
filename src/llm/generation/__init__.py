"""Échantillonnage et génération."""
from .generate import generate
from .sampling import filter_logits, sample_next_token

__all__ = ["generate", "filter_logits", "sample_next_token"]
