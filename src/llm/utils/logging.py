"""Enregistrement structuré des métriques d'expérience."""
import csv
import json
from pathlib import Path


def save_history(history, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    if history:
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
