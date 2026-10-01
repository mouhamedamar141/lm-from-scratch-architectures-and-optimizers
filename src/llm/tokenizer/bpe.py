"""Tokenizer BPE compact au niveau des caractères Unicode.

Cette implémentation pédagogique n'est pas un BPE octet de production. Les espaces sont
préservés. L'aller-retour est exact pour les caractères connus ; un caractère inconnu
est remplacé par le token spécial <unk>.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>")


class BPETokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]] | None = None):
        self.vocab = list(dict.fromkeys([*SPECIAL, *vocab]))
        self.merges = list(merges or [])
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = self.vocab
        self.special_ids = {token: self.token_to_id[token] for token in SPECIAL}

    @classmethod
    def train(cls, texts: str | Iterable[str], vocab_size: int = 512) -> "BPETokenizer":
        """Apprend les fusions de caractères adjacents les plus fréquentes."""
        if isinstance(texts, str):
            corpus = texts
        else:
            corpus = "\n".join(texts)
        symbols = list(dict.fromkeys(corpus))
        if vocab_size < len(SPECIAL) + len(symbols):
            raise ValueError(f"vocab_size doit être supérieur ou égal à {len(SPECIAL) + len(symbols)} pour ce corpus")
        sequences = [list(line) for line in corpus.splitlines(keepends=True) if line]
        merges: list[tuple[str, str]] = []
        while len(SPECIAL) + len(symbols) + len(merges) < vocab_size:
            counts: Counter[tuple[str, str]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            if count < 2:
                break
            merged = pair[0] + pair[1]
            merges.append(pair)
            symbols.append(merged)
            for i, seq in enumerate(sequences):
                out, j = [], 0
                while j < len(seq):
                    if j + 1 < len(seq) and (seq[j], seq[j + 1]) == pair:
                        out.append(merged)
                        j += 2
                    else:
                        out.append(seq[j])
                        j += 1
                sequences[i] = out
        return cls(symbols, merges)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        seq = list(text)
        for left, right in self.merges:
            merged = left + right
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        ids = [self.token_to_id.get(piece, self.special_ids["<unk>"]) for piece in seq]
        if add_bos:
            ids.insert(0, self.special_ids["<bos>"])
        if add_eos:
            ids.append(self.special_ids["<eos>"])
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if skip_special_tokens and token in SPECIAL:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "bpe-char", "vocab": self.vocab,
                                   "merges": self.merges}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], [tuple(pair) for pair in data["merges"]])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
