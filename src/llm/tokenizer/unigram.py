"""Tokenizer Unigram transparent avec segmentation par l'algorithme de Viterbi.

Cette approximation pédagogique de SentencePiece Unigram note les pièces candidates
à partir de leurs fréquences normalisées ; EM et le repli octet sont omis.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from .bpe import SPECIAL


class UnigramTokenizer:
    def __init__(self, scores: dict[str, float]):
        self.scores = dict(scores)
        self.vocab = list(dict.fromkeys([*SPECIAL, *scores.keys()]))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = self.vocab
        self.special_ids = {t: self.token_to_id[t] for t in SPECIAL}
        self.max_piece = max((len(t) for t in scores), default=1)

    @classmethod
    def train(cls, texts: str | Iterable[str], vocab_size: int = 512, max_piece_len: int = 12):
        text = texts if isinstance(texts, str) else "\n".join(texts)
        counts: Counter[str] = Counter()
        for i, char in enumerate(text):
            counts[char] += 1
            for n in range(2, min(max_piece_len, len(text) - i) + 1):
                counts[text[i:i+n]] += 1
        chars = sorted(set(text))
        capacity = max(0, vocab_size - len(SPECIAL) - len(chars))
        multi = [(p, c) for p, c in counts.items() if len(p) > 1]
        multi.sort(key=lambda x: (x[1] / len(text), x[1], x[0]), reverse=True)
        chosen = {c: counts[c] for c in chars}
        chosen.update(dict(multi[:capacity]))
        total = sum(chosen.values()) or 1
        # Les probabilités logarithmiques donnent un score additif à Viterbi.
        return cls({p: math.log(freq / total) for p, freq in chosen.items()})

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        n = len(text)
        best = [-math.inf] * (n + 1)
        prev: list[tuple[int, str] | None] = [None] * (n + 1)
        best[0] = 0.0
        for end in range(1, n + 1):
            lo = max(0, end - self.max_piece)
            for start in range(lo, end):
                piece = text[start:end]
                if piece in self.scores and best[start] > -math.inf:
                    score = best[start] + self.scores[piece]
                    if score > best[end]:
                        best[end], prev[end] = score, (start, piece)
            if prev[end] is None:
                # Conserver un caractère Unicode inconnu sous forme de token si possible.
                char = text[end - 1]
                if char in self.token_to_id:
                    best[end], prev[end] = best[end - 1] - 100.0, (end - 1, char)
                else:
                    best[end], prev[end] = best[end - 1] - 100.0, (end - 1, "<unk>")
        pieces, pos = [], n
        while pos:
            item = prev[pos]
            if item is None:
                pieces.append("<unk>")
                pos -= 1
            else:
                pos, piece = item
                pieces.append(piece)
        ids = [self.token_to_id.get(p, self.special_ids["<unk>"]) for p in reversed(pieces)]
        if add_bos: ids.insert(0, self.special_ids["<bos>"])
        if add_eos: ids.append(self.special_ids["<eos>"])
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        out = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if skip_special_tokens and token in SPECIAL: continue
            out.append(token)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "unigram-char", "scores": self.scores}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UnigramTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["scores"])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
