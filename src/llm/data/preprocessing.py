"""Chargement du corpus et tokenisation déterministe des flux train/validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def prepare_data(corpus_path, tokenizer, output_dir, val_fraction=0.1):
    text = Path(corpus_path).read_text(encoding="utf-8")
    if len(text) < 10:
        raise ValueError("Corpus trop court ; fournir au moins 10 caractères")
    ids = np.asarray(tokenizer.encode(text), dtype=np.uint32)
    split = max(1, min(len(ids) - 1, int(len(ids) * (1 - val_fraction))))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ids[:split].tofile(out / "train.bin"); ids[split:].tofile(out / "val.bin")
    meta = {"corpus": str(corpus_path), "tokens": int(len(ids)), "train_tokens": int(split),
            "val_tokens": int(len(ids) - split), "vocab_size": tokenizer.vocab_size,
            "chars_per_token": len(text) / len(ids), "val_fraction": val_fraction}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_tokens(path):
    return np.fromfile(path, dtype=np.uint32).astype(np.int64)
