"""Chargement de fichiers YAML de configuration."""
from pathlib import Path


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Installez PyYAML avec `pip install pyyaml` pour lire les configurations YAML") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
