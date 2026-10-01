"""Charge un tokenizer sauvegardé et expose une interface uniforme."""
import json
from pathlib import Path

from .bpe import BPETokenizer
from .byte_bpe import ByteBPETokenizer
from .unigram import UnigramTokenizer
from .wordpiece import WordPieceTokenizer

LOADERS = {"bpe-char": BPETokenizer, "bpe-byte": ByteBPETokenizer,
           "unigram-char": UnigramTokenizer, "wordpiece": WordPieceTokenizer}


def load_tokenizer(path):
    kind = json.loads(Path(path).read_text(encoding="utf-8")).get("type")
    if kind not in LOADERS:
        raise ValueError(f"Type de tokenizer non pris en charge dans {path} : {kind}")
    return LOADERS[kind].load(path)
