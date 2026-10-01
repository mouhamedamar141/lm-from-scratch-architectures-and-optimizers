from llm.tokenizer import BPETokenizer, UnigramTokenizer


def test_bpe_reversible_for_known_characters_and_unknown_unicode():
    tok = BPETokenizer.train("bonjour, monde! café λ\nbonjour", vocab_size=40)
    sample = "bonjour, monde! café λ"
    assert tok.decode(tok.encode(sample)) == sample
    unknown = tok.encode("雪")
    assert unknown == [tok.special_ids["<unk>"]]
    assert "<unk>" in tok.decode(unknown, skip_special_tokens=False)
    assert tok.vocab_size <= 40


def test_bpe_special_tokens():
    tok = BPETokenizer.train("abababab", vocab_size=8)
    ids = tok.encode("ab", add_bos=True, add_eos=True)
    assert tok.decode(ids, skip_special_tokens=False).startswith("<bos>")
    assert tok.decode(ids) == "ab"


def test_unigram_round_trip_on_seen_text_and_save_load(tmp_path):
    text = "bonjour monde bonjour"
    tok = UnigramTokenizer.train(text, vocab_size=20)
    assert tok.decode(tok.encode(text)) == text
    path = tmp_path / "tok.json"; tok.save(path)
    restored = UnigramTokenizer.load(path)
    assert restored.decode(restored.encode(text)) == text


def test_byte_bpe_roundtrip_any_unicode():
    from llm.tokenizer import ByteBPETokenizer
    tok = ByteBPETokenizer.train("bonjour le monde bonjour le monde " * 5, vocab_size=270)
    for s in ["bonjour le monde", "café 雪 😊", "  espaces   multiples\n"]:
        assert tok.decode(tok.encode(s)) == s      # aucun <unk> possible
    assert tok.vocab_size <= 270
    assert len(tok.encode("bonjour le monde")) < len("bonjour le monde")


def test_wordpiece_encodes_known_words():
    from llm.tokenizer import WordPieceTokenizer
    tok = WordPieceTokenizer.train("le chat mange le chat dort " * 5, vocab_size=40)
    assert tok.decode(tok.encode("le chat")) == "le chat"
