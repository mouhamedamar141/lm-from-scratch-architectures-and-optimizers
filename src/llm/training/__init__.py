"""Optimiseurs, schedules, boucle d'entraînement et comparaisons contrôlées."""
from .compare import compare, plot_curves, table
from .optimizers import OPTIMIZERS, build_optimizer
from .schedules import SCHEDULES, lr_factor
from .trainer import TrainConfig, evaluate, get_batch, train

__all__ = ["OPTIMIZERS", "build_optimizer", "SCHEDULES", "lr_factor", "TrainConfig", "train", "evaluate",
           "get_batch", "compare", "plot_curves", "table"]
