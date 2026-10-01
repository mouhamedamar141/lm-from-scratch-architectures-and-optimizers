"""Tokenise le corpus et écrit les flux binaires train/validation. Option : générer un corpus synthétique."""
import argparse
from pathlib import Path

from llm.data.preprocessing import prepare_data
from llm.data.synthetic import make_corpus
from llm.tokenizer.tokenizer import load_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/raw/corpus.txt")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--synthetic-chars", type=int, default=0, help="si > 0 : écrit un corpus synthétique dans --corpus")
    a = p.parse_args()
    if a.synthetic_chars:
        Path(a.corpus).parent.mkdir(parents=True, exist_ok=True)
        Path(a.corpus).write_text(make_corpus(a.synthetic_chars), encoding="utf-8")
    print(prepare_data(a.corpus, load_tokenizer(a.tokenizer), a.output_dir, a.val_fraction))


if __name__ == "__main__":
    main()
