"""Vocabulaire déterministe avec quatre tokens spéciaux."""
from .bpe import SPECIAL


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = list(dict.fromkeys([*SPECIAL, *tokens]))
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = self.tokens
    def __len__(self): return len(self.tokens)
    def encode(self, tokens): return [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in tokens]
    def decode(self, ids): return [self.id_to_token[int(i)] for i in ids]
