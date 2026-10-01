"""Cache clé/valeur pour la génération incrémentale."""
import torch


class KVCache:
    """Cache par couche. Chaque couche stocke un dictionnaire de tenseurs concaténés le long du temps.

    - attention MHA/MQA/GQA : ``k`` et ``v`` de forme (B, H_kv, T, d), ``dim=2``.
    - attention MLA         : ``c`` (latent, B, T, r) et ``kr`` (clé RoPE, B, T, d_r), ``dim=1``.
    ``n_seen`` compte les tokens déjà traités : c'est la position absolue du prochain token.
    """

    def __init__(self, n_layer: int, window=None):
        self.layers = [None] * n_layer
        self.n_seen = 0
        self.window = window

    def update(self, layer: int, dim: int, **tensors):
        old = self.layers[layer]
        full = {k: (v if old is None else torch.cat([old[k], v], dim=dim)) for k, v in tensors.items()}
        stored = full
        if self.window is not None:  # cache roulant : on ne conserve que les `window` derniers tokens
            stored = {k: v.narrow(dim, max(0, v.size(dim) - self.window), min(v.size(dim), self.window))
                      for k, v in full.items()}
        self.layers[layer] = stored
        return full  # la requête courante voit tout ce qui vient d'être concaténé

    def length(self, layer: int = 0) -> int:
        entry = self.layers[layer]
        if entry is None:
            return 0
        first = next(iter(entry.values()))
        return first.size(2 if first.dim() == 4 else 1)

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for e in self.layers if e for v in e.values())
