"""WordPiece (style BERT) : fusions choisies par score de vraisemblance, encodage glouton par plus long préfixe."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>")


class WordPieceTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = list(dict.fromkeys([*SPECIAL, *vocab]))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.special_ids = {s: self.token_to_id[s] for s in SPECIAL}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int = 200) -> "WordPieceTokenizer":
        r"""Score de fusion : freq(ab) / (freq(a) * freq(b)) (contrairement à BPE qui utilise freq(ab))."""
        words = Counter(WORD.findall(text))
        seqs = {w: [w[0]] + ["##" + c for c in w[1:]] for w in words}
        vocab = {s for seq in seqs.values() for s in seq}
        while len(SPECIAL) + len(vocab) < vocab_size:
            uni, duo = Counter(), Counter()
            for w, seq in seqs.items():
                f = words[w]
                for s in seq:
                    uni[s] += f
                for p in zip(seq, seq[1:]):
                    duo[p] += f
            if not duo:
                break
            (a, b), _ = max(duo.items(), key=lambda kv: (kv[1] / (uni[kv[0][0]] * uni[kv[0][1]]), kv[0]))
            new = a + b[2:]
            vocab.add(new)
            for w, seq in seqs.items():
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        out.append(new); i += 2
                    else:
                        out.append(seq[i]); i += 1
                seqs[w] = out
        return cls(sorted(vocab))

    def _encode_word(self, w: str) -> list[str]:
        pieces, start = [], 0
        while start < len(w):
            end, cur = len(w), None
            while start < end:
                sub = w[start:end] if start == 0 else "##" + w[start:end]
                if sub in self.token_to_id:
                    cur = sub; break
                end -= 1
            if cur is None:
                return ["<unk>"]
            pieces.append(cur); start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        return [p for w in WORD.findall(text) for p in self._encode_word(w)]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id[p] for p in self.tokenize(text)]
        return ([self.special_ids["<bos>"]] if add_bos else []) + ids + ([self.special_ids["<eos>"]] if add_eos else [])

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = ""
        for i in ids:
            t = self.vocab[int(i)]
            if skip_special_tokens and t in SPECIAL:
                continue
            out += t[2:] if t.startswith("##") else (" " + t)
        return out.strip()

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "wordpiece", "vocab": self.vocab}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "WordPieceTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["vocab"])
