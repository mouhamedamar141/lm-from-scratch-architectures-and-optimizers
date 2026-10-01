from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "04-embeddings.ipynb"
TITLE = "Embeddings de tokens"


def cells():
    return [
md("""# 04 · Embeddings de tokens

**À retenir**
- Un embedding est une **table de paramètres** $E\\in\\mathbb R^{V\\times d}$ : le token $i$ devient la ligne $E_i$. Équivalent à $\\text{onehot}(i)\\,E$, mais sans construire le one-hot.
- Le gradient ne touche **que les lignes des tokens présents** dans le lot (mise à jour creuse).
- **Weight tying** [2] : réutiliser $E$ comme matrice de sortie ($\\text{logits}=h\\,E^\\top$) économise $V\\,d$ paramètres.
- Le sens vient du **contexte** : des mots aux contextes proches obtiennent des vecteurs proches [1].

Hypothèses testées : (H1) lookup ≡ one-hot·E ; (H2) le gradient est creux ; (H3) des vecteurs issus des co-occurrences regroupent les mots par usage ; (H4) le weight tying réduit les paramètres pour une perte comparable."""),
code(PRELUDE_IMPORTS + "\nimport torch.nn as nn\ntorch.manual_seed(0)"),
md("## 1. Lookup = produit avec un one-hot ; gradient creux"),
code('''V, d = 6, 4
emb = nn.Embedding(V, d)
ids = torch.tensor([1, 3, 3])
lookup = emb(ids)
onehot = torch.nn.functional.one_hot(ids, V).float() @ emb.weight
print("H1 lookup == one-hot @ E :", torch.allclose(lookup, onehot))

lookup.sum().backward()
print("\\nnorme du gradient par ligne de E (lignes 1 et 3 seulement) :")
print(emb.weight.grad.norm(dim=1))'''),
md("""La ligne 3 reçoit un gradient double (le token apparaît deux fois). Pour un vocabulaire de 10⁵ tokens, seule une petite fraction des lignes est mise à jour à chaque pas : c'est pourquoi les tokens rares sont mal appris.

## 2. Le sens vient du contexte : co-occurrences → SVD
Sans réseau : on construit la matrice de co-occurrence de mots du corpus synthétique, on applique le PPMI puis une SVD tronquée (approche classique équivalente en esprit à word2vec [1])."""),
tool("data/synthetic.py"),
code('''texte = make_corpus(60_000, seed=1)
mots = texte.replace(".", " .").replace("(", "( ").replace(")", " )").replace(",", " ,").split()
vocab = sorted(set(mots)); idx = {w: i for i, w in enumerate(vocab)}
C = np.zeros((len(vocab), len(vocab)))
for i, w in enumerate(mots):
    for j in range(max(0, i - 2), min(len(mots), i + 3)):
        if i != j: C[idx[w], idx[mots[j]]] += 1
tot = C.sum(); pw = C.sum(1, keepdims=True) / tot
ppmi = np.maximum(np.log((C / tot + 1e-12) / (pw * pw.T + 1e-12)), 0)
U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
vecs = U[:, :8] * S[:8]; vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

def voisins(mot, k=4):
    sim = vecs @ vecs[idx[mot]]; return [vocab[i] for i in np.argsort(-sim)[1:k + 1]]
for m in ["chats", "chien", "mange", "cherchent", "pomme"]:
    print(f"{m:10s} -> {voisins(m)}")'''),
md("""**Lecture** : les voisins doivent regrouper les mots interchangeables (pluriels avec pluriels, verbes avec verbes, objets avec objets), simplement à partir de leurs contextes. À vérifier dans la sortie ci-dessus ; avec un corpus généré par grammaire, la structure est nette, ce qui **ne prouve pas** le même résultat sur du texte naturel.

## 3. Weight tying : paramètres et perte
Comparaison contrôlée (mêmes données, seeds, budget) : `tie_weights=True` vs `False`, tokens = caractères ($V$ petit ⇒ l'économie relative est faible ; on calcule aussi l'effet pour un $V$ réaliste)."""),
tool(),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
BASE_T = dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10)
res = compare({"tying (E partagée)": dict(tie_weights=True), "sans tying": dict(tie_weights=False)}, DATA, BASE_M, BASE_T, seeds=(0, 1))
for V_, d_ in [(32_000, 512), (128_000, 4096)]:
    print(f"Économie pour V={V_:,}, d={d_} : {V_*d_/1e6:.0f} M paramètres")'''),
md("""**Conclusion (H4), d'après la sortie ci-dessus** : le tying retire bien $V\\,d$ paramètres, mais **à cette échelle et ce budget il dégrade la perte** (écart de l'ordre de plusieurs écarts-types) : avec $V=31$ et $d=64$, l'économie est négligeable (2 k paramètres) et la contrainte $E_{\\text{in}}=E_{\\text{out}}$ coûte de la capacité. Le tying se justifie quand $V\\,d$ est une grande part du modèle (petits modèles à grand vocabulaire) ; il faut donc le tester plutôt que l'adopter par défaut.

**Limites** : 2 seeds, modèle de 100 k paramètres, corpus synthétique ; le sens de « voisin » n'est pas évalué quantitativement (pas de tâche de similarité).

""" + refs_block("mikolov2013", "press2017", "vaswani2017")),
]
