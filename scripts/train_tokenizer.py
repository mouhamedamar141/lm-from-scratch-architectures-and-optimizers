"""Entraîne un tokenizer (bpe | byte-bpe | wordpiece | unigram) sur un fichier texte."""
import argparse
from pathlib import Path

from llm.tokenizer import BPETokenizer, ByteBPETokenizer, UnigramTokenizer, WordPieceTokenizer

KINDS = {"bpe": BPETokenizer, "byte-bpe": ByteBPETokenizer, "wordpiece": WordPieceTokenizer, "unigram": UnigramTokenizer}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/raw/corpus.txt")
    p.add_argument("--output", default="data/tokenizer/tokenizer.json")
    p.add_argument("--type", choices=list(KINDS), default="bpe")
    p.add_argument("--vocab-size", type=int, default=256)
    a = p.parse_args()
    tok = KINDS[a.type].train(Path(a.corpus).read_text(encoding="utf-8"), a.vocab_size)
    tok.save(a.output)
    print(f"Tokenizer {a.type} : {tok.vocab_size} unités -> {a.output}")


if __name__ == "__main__":
    main()
