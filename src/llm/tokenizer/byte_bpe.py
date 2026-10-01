"""BPE au niveau des octets (style GPT-2) : couvre tout Unicode, aucun <unk> possible."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Pré-tokenisation inspirée de GPT-2 (module `re` standard : pas de \p{L}, on utilise [^\W\d_]).
PRETOKEN = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
SPECIAL = ("<pad>", "<bos>", "<eos>")


class ByteBPETokenizer:
    """Vocabulaire = 3 tokens spéciaux + 256 octets + fusions apprises."""

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.n_special = len(SPECIAL)
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.bytes_of = {self.n_special + b: bytes([b]) for b in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.bytes_of[self.n_special + 256 + i] = self.bytes_of[a] + self.bytes_of[b]
        self.special_ids = {s: i for i, s in enumerate(SPECIAL)}

    @property
    def vocab_size(self) -> int:
        return self.n_special + 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = 300) -> "ByteBPETokenizer":
        n_merges = vocab_size - len(SPECIAL) - 256
        if n_merges < 0:
            raise ValueError(f"vocab_size doit être >= {len(SPECIAL) + 256}")
        words = Counter(PRETOKEN.findall(text))
        seqs = {w: [len(SPECIAL) + b for b in w.encode("utf-8")] for w in words}
        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            pairs: Counter = Counter()
            for w, seq in seqs.items():
                for pair in zip(seq, seq[1:]):
                    pairs[pair] += words[w]
            if not pairs:
                break
            best, count = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            if count < 2:
                break
            new_id = len(SPECIAL) + 256 + len(merges)
            merges.append(best)
            for w, seq in seqs.items():
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                        out.append(new_id); i += 2
                    else:
                        out.append(seq[i]); i += 1
                seqs[w] = out
        return cls(merges)

    def _encode_word(self, word: str) -> list[int]:
        seq = [self.n_special + b for b in word.encode("utf-8")]
        while len(seq) > 1:  # on applique à chaque tour la fusion de plus petit rang
            ranked = [(self.rank.get(p, 1 << 30), i) for i, p in enumerate(zip(seq, seq[1:]))]
            r, i = min(ranked)
            if r == 1 << 30:
                break
            seq[i:i + 2] = [self.n_special + 256 + r]
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.special_ids["<bos>"]] if add_bos else []
        for w in PRETOKEN.findall(text):
            ids += self._encode_word(w)
        return ids + ([self.special_ids["<eos>"]] if add_eos else [])

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        raw = b"".join(self.bytes_of[int(i)] for i in ids if int(i) >= self.n_special or not skip_special_tokens
                       and int(i) in self.bytes_of)
        return raw.decode("utf-8", errors="replace")

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "bpe-byte", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ByteBPETokenizer":
        return cls([tuple(m) for m in json.loads(Path(path).read_text(encoding="utf-8"))["merges"]])
