"""Composants explicites du Transformer (une seule configuration pilote toutes les variantes)."""
from .config import ModelConfig
from .model import LM, Output

__all__ = ["ModelConfig", "LM", "Output"]
