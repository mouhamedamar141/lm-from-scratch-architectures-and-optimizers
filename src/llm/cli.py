"""Points d'entrée installables pour les commandes du projet."""
import runpy
import sys
from pathlib import Path


def _run(script):
    # Les commandes reprennent les scripts documentés sans dupliquer leur logique.
    path = Path(__file__).resolve().parents[2] / "scripts" / script
    sys.path.insert(0, str(path.parent))  # permet `from _common import ...` dans les scripts
    runpy.run_path(str(path), run_name="__main__")


def prepare_main(): _run("prepare_data.py")
def train_main(): _run("train.py")
def evaluate_main(): _run("evaluate.py")
def generate_main(): _run("generate.py")
def tokenizer_main(): _run("train_tokenizer.py")
def experiments_main(): _run("run_experiments.py")
