from .lib import md, code, refs_block

FILE = "01-tokenisation-bases.ipynb"
TITLE = "Tokenisation : caractère, mot, sous-mot"

TEXTE = '''CORPUS = """le chat dort sur le canapé. le chat mange la souris. le chien dort dans le jardin.
les chats et les chiens dorment ensemble. un modèle de langage prédit le mot suivant.
the cat sleeps on the sofa. the dog eats the mouse. language models predict the next token.
la tokenisation transforme le texte en nombres. tokenization turns text into numbers."""
'''


def cells():
    return [
md("""# 01 · Tokenisation : caractère, mot, sous-mot

**À retenir**
- Un *tokenizer* convertit du texte en identifiants entiers ; c'est le premier choix de conception d'un LLM.
- **Caractère** : aucun mot inconnu, mais séquences longues. **Mot** : séquences courtes, mais vocabulaire énorme et mots inconnus (OOV).
- **Sous-mot** (BPE, WordPiece, Unigram) : compromis utilisé par tous les LLM modernes.
- Compromis mesurable : longueur de séquence $L$ ↔ taille du vocabulaire $V$ (coût $V\\times d$ paramètres d'embedding).

**Hypothèse testée** : les sous-mots offrent un meilleur compromis (séquence courte *et* pas d'OOV) que caractères ou mots."""),
md("## 1. Corpus et trois tokenizers minimaux"),
code(TEXTE + '''
TEST = "les chats dorment. the cats sleep. tokenisation multilingue ✓"   # contient des mots et un symbole inconnus

class CharTokenizer:
    def __init__(self, corpus):
        self.itos = ["<unk>"] + sorted(set(corpus)); self.stoi = {c: i for i, c in enumerate(self.itos)}
    def encode(self, s): return [self.stoi.get(c, 0) for c in s]
    def decode(self, ids): return "".join(self.itos[i] for i in ids)

class WordTokenizer:
    def __init__(self, corpus):
        words = sorted(set(corpus.split()))
        self.itos = ["<unk>"] + words; self.stoi = {w: i for i, w in enumerate(self.itos)}
    def encode(self, s): return [self.stoi.get(w, 0) for w in s.split()]
    def decode(self, ids): return " ".join(self.itos[i] for i in ids)

char_tok, word_tok = CharTokenizer(CORPUS), WordTokenizer(CORPUS)
for name, tok in [("caractère", char_tok), ("mot", word_tok)]:
    ids = tok.encode(TEST)
    print(f"{name:10s} V={len(tok.itos):3d}  L={len(ids):3d}  inconnus={ids.count(0)}  -> {tok.decode(ids)!r}")'''),
md("""Le tokenizer **mot** perd les mots absents du corpus (`<unk>`) ; le **caractère** n'en perd (presque) aucun mais produit ~5× plus de tokens.

## 2. Le compromis, mesuré
Métriques : *taux de compression* $\\;c=\\dfrac{\\#\\text{caractères}}{\\#\\text{tokens}}$ ; *fertilité* = tokens par mot ; *taux d'OOV* sur un texte de test."""),
code('''def metriques(tok, texte):
    ids = tok.encode(texte); n_mots = len(texte.split())
    return dict(V=len(tok.itos), tokens=len(ids), compression=len(texte)/len(ids), fertilite=len(ids)/n_mots, oov=ids.count(0)/len(ids))

TEXTE_TEST = "le chien mange la souris. the dog sleeps. un chercheur compare des modèles."
print(f"{'tokenizer':10s}{'V':>5}{'tokens':>8}{'car/token':>11}{'tok/mot':>9}{'OOV %':>7}")
for name, tok in [("caractère", char_tok), ("mot", word_tok)]:
    m = metriques(tok, TEXTE_TEST)
    print(f"{name:10s}{m['V']:>5}{m['tokens']:>8}{m['compression']:>11.2f}{m['fertilite']:>9.2f}{100*m['oov']:>7.1f}")'''),
md("""**Lecture.** Le mot compresse fort mais avec beaucoup d'OOV (ici le test contient des mots absents) ; le caractère n'a pas d'OOV mais un rapport de 1 caractère/token.
Les tokenizers **sous-mot** (notebooks 02 et 03) visent l'entre-deux : vocabulaire fixe de 10³–10⁵ unités, aucun OOV grâce au repli sur des unités plus petites.

## 3. Pourquoi le vocabulaire coûte cher
Embedding d'entrée + tête de sortie : $2\\,V\\,d$ paramètres (ou $V\\,d$ si les poids sont partagés [2]). Pour $d=4096$ :"""),
code('''for V in (32_000, 128_000, 256_000):
    d = 4096
    print(f"V={V:>7,}  embeddings : {V*d/1e6:7.0f} M params ({V*d*2/1e9:.2f} Go en bf16) ; avec tête non partagée : {2*V*d/1e6:.0f} M")'''),
md("""**Compromis** : un grand $V$ raccourcit les séquences (attention en $O(L^2)$, cache KV en $O(L)$) mais alourdit les embeddings et rend les tokens rares mal entraînés.

## Bilan
| | Caractère | Mot | Sous-mot |
|---|---|---|---|
| Vocabulaire $V$ | très petit | énorme | moyen (fixé) |
| Longueur $L$ | très longue | courte | courte à moyenne |
| OOV | non | oui | non |

**Limites** : corpus jouet (quelques dizaines de mots) ; les valeurs numériques illustrent la méthode, pas des tendances générales.

""" + refs_block("sennrich2016", "press2017", "radford2019")),
]
