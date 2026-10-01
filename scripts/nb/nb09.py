from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "09-ffn-et-activations.ipynb"
TITLE = "Réseaux feed-forward : ReLU, GELU, SiLU et variantes à porte (GLU)"


def cells():
    return [
md("""# 09 · Feed-forward et activations

**À retenir**
- Le FFN traite chaque position indépendamment ; il concentre ~2/3 des paramètres d'un bloc.
- Classique : $\\text{FFN}(x)=W_2\\,\\sigma(W_1x)$ avec $W_1\\in\\mathbb R^{d_{ff}\\times d}$, $d_{ff}=4d$.
- **À porte (GLU)** [1, 2] : $\\text{FFN}(x)=W_2\\big(\\sigma(W_g x)\\odot W_u x\\big)$. Avec $\\sigma=$ SiLU → **SwiGLU** (LLaMA, Mistral, Qwen, DeepSeek), GELU → GeGLU (Gemma), ReLU → ReGLU.
- 3 matrices au lieu de 2 ⇒ pour **conserver le nombre de paramètres**, on prend $d_{ff}=\\tfrac23\\cdot4d=\\tfrac83d$ (arrondi).

$$\\mathrm{ReLU}(x)=\\max(0,x),\\quad \\mathrm{GELU}(x)=x\\,\\Phi(x),\\quad \\mathrm{SiLU}(x)=x\\,\\sigma(x)=\\frac{x}{1+e^{-x}}$$"""),
code(PRELUDE_IMPORTS),
tool(),
code('''x = torch.linspace(-4, 4, 400)
plt.figure(figsize=(5.6, 3.3))
for name, f in [("ReLU", torch.relu), ("GELU", torch.nn.functional.gelu), ("SiLU/Swish", torch.nn.functional.silu)]: plt.plot(x, f(x), label=name)
plt.legend(); plt.title("Activations"); plt.xlabel("x"); plt.show()'''),
md("""## 1. Comparaison à nombre de paramètres égal
`ffn_hidden` applique la règle $\\tfrac23$ aux variantes à porte. On vérifie que les totaux de paramètres sont proches."""),
code('''cfg0 = dict(vocab_size=31, block_size=32, n_layer=2, n_head=4, n_embd=64)
print(f"{'FFN':8}{'largeur cachée':>16}{'params FFN/couche':>20}")
for t in ["relu", "gelu", "silu", "reglu", "geglu", "swiglu"]:
    c = ModelConfig(**cfg0, ffn_type=t); f = FeedForward(c)
    print(f"{t:8}{ffn_hidden(c):>16}{sum(p.numel() for p in f.parameters()):>20,}")'''),
md("## 2. Expérience : 6 FFN, mêmes données, seeds et budget"),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
BASE_T = dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10)
res = compare({t: dict(ffn_type=t) for t in ["relu", "gelu", "silu", "reglu", "geglu", "swiglu"]}, DATA, BASE_M, BASE_T, seeds=(0, 1, 2))
plot_curves(res, title="perte de validation"); plt.show()'''),
md("""## Bilan
**Constats sur l'exécution fournie** (3 seeds) : les trois variantes à porte (ReGLU, GeGLU, SwiGLU) sont **meilleures que les FFN classiques** et indiscernables entre elles ; parmi les FFN classiques, **ReLU devance GELU et SiLU** sur ce budget très court (250 pas). Ce dernier point **n'est pas** celui de la littérature à grande échelle : il reflète vraisemblablement une vitesse d'optimisation initiale et non la qualité asymptotique — ne pas l'extrapoler.
- Les variantes à porte sont le standard des LLM modernes [2, 5] ; le gain de [2] est rapporté à grande échelle.
- Ne pas comparer des FFN à porte et classiques **sans** ajuster $d_{ff}$ : ici la règle $\\tfrac23$ laisse un écart résiduel de ~2 % de paramètres au profit des variantes à porte (arrondi de la largeur), écart petit mais à garder en tête.

**Limites** : 3 seeds, petit modèle, 250 pas (régime très court), corpus synthétique.

""" + refs_block("dauphin2017", "shazeer2020", "hendrycks2016", "ramachandran2017", "touvron2023")),
]
