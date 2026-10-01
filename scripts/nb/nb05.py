from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "05-encodages-de-position.ipynb"
TITLE = "Encodages de position : sinusoïdal, appris, RoPE, ALiBi, NoPE"


def cells():
    return [
md("""# 05 · Encodages de position

**À retenir**
- L'auto-attention est **équivariante aux permutations** : sans information de position, « le chat chasse le chien » ≡ « le chien chasse le chat ».
- **Absolus** (ajoutés à l'embedding) : sinusoïdal [1], appris (GPT-2). Longueur maximale fixée.
- **Relatifs dans l'attention** : **RoPE** [2] (rotation de $q,k$), **ALiBi** [3] (biais linéaire $-m_h\\,(i-j)$). Extrapolent mieux.
- **NoPE** [4] : aucune position ; le masque causal suffit à en déduire l'ordre dans un décodeur.
- **Extension de contexte** de RoPE : Position Interpolation [5], NTK-aware, YaRN [6].

| Méthode | Où | Formule clé |
|---|---|---|
| Sinusoïdal | entrée | $PE_{p,2i}=\\sin(p/10000^{2i/d}),\\ PE_{p,2i+1}=\\cos(\\cdot)$ |
| Appris | entrée | $x_p\\leftarrow x_p+P_p$, $P\\in\\mathbb R^{L\\times d}$ |
| **RoPE** | $q,k$ | $\\tilde q_p=R_{p}\\,q$, $\\ \\tilde q_i^\\top\\tilde k_j=q_i^\\top R_{j-i}k_j$ |
| ALiBi | scores | $s_{ij}=q_i^\\top k_j/\\sqrt{d}-m_h(i-j)$ |"""),
code(PRELUDE_IMPORTS),
tool(),
md("""## 1. Rotary Embedding : propriété relative vérifiée numériquement
Pour chaque paire de coordonnées $(x_{2i},x_{2i+1})$, RoPE applique la rotation d'angle $p\\,\\theta_i$ avec $\\theta_i=\\theta_{\\text{base}}^{-2i/d}$. Le produit scalaire ne dépend alors que du décalage $j-i$."""),
code('''torch.manual_seed(0)
d = 16; inv, _ = rope_inv_freq(d)
q, k = torch.randn(d), torch.randn(d)
score = lambda i, j: (apply_rope(q[None], torch.tensor([i]), inv) * apply_rope(k[None], torch.tensor([j]), inv)).sum().item()
print("score(i=5,  j=2)   =", round(score(5, 2), 5))
print("score(i=105,j=102) =", round(score(105, 102), 5), "  <- même décalage (3) => même score")
print("score(i=5,  j=3)   =", round(score(5, 3), 5), "  <- décalage 2 => score différent")
print("norme conservée    :", torch.allclose(apply_rope(q[None], torch.tensor([37]), inv).norm(), q.norm(), atol=1e-5))'''),
md("""## 2. Visualisation : sinusoïdal, fréquences RoPE, pentes ALiBi"""),
code('''fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
pe = sinusoidal_table(64, 32); ax[0].imshow(pe.numpy(), aspect="auto", cmap="RdBu"); ax[0].set_title("sinusoïdal (position × dimension)"); ax[0].set_xlabel("dim"); ax[0].set_ylabel("position"); ax[0].grid(False)
for th in (10_000, 500_000):
    inv, _ = rope_inv_freq(64, theta=th); ax[1].semilogy(inv.numpy(), "o-", ms=3, label=f"θ={th:,}")
ax[1].set_title("RoPE : fréquences θ_i"); ax[1].set_xlabel("paire i"); ax[1].legend()
n = 8; dist = torch.arange(0, 33).float(); sl = alibi_slopes(n)
for h in range(n): ax[2].plot(dist, -sl[h] * dist, label=f"m={sl[h]:.3g}" if h in (0, n - 1) else None)
ax[2].set_title("ALiBi : biais −m·(i−j) par tête"); ax[2].set_xlabel("distance i−j"); ax[2].legend(); plt.tight_layout(); plt.show()'''),
md("""**Lecture** : les premières paires RoPE tournent vite (information locale), les dernières très lentement (information à longue portée). Augmenter la base $\\theta$ ralentit toutes les rotations : c'est le levier utilisé par les modèles à long contexte (par ex. LLaMA-3 utilise une base $5\\times10^5$ [7]).

## 3. Extension de contexte : comment chaque méthode modifie les fréquences
Pour un facteur $s=L_{\\text{cible}}/L_{\\text{train}}$ :
- **Interpolation linéaire (PI)** : $\\theta_i\\leftarrow\\theta_i/s$.
- **NTK-aware** : base $\\theta\\leftarrow\\theta\\,s^{d/(d-2)}$ (les hautes fréquences bougent peu, les basses beaucoup).
- **YaRN** : interpolation *par bande* selon le nombre de tours $r_i=L_{\\text{train}}\\theta_i/2\\pi$ (rampe entre $\\alpha$ et $\\beta$) + facteur de température $\\sqrt{1/t}=0.1\\ln s+1$ sur $\\cos/\\sin$."""),
code('''d, s, L0 = 32, 4.0, 2048
plt.figure(figsize=(6, 3.4))
base, _ = rope_inv_freq(d)
plt.semilogy(base, "k-", lw=2, label="original")
for name in ["linear", "ntk", "yarn"]:
    f, ms = rope_inv_freq(d, scaling=name, factor=s, orig_len=L0)
    plt.semilogy(f, "o--", ms=3, label=f"{name} (mscale={ms:.3f})")
plt.xlabel("paire i"); plt.ylabel("fréquence θ_i"); plt.legend(); plt.title("Fréquences RoPE après extension (s=4)"); plt.show()'''),
md("""## 4. Expérience : 5 modes de position, entraînés à longueur 32, évalués jusqu'à 128
Protocole : mêmes données (corpus synthétique), même modèle (2 couches, $d=64$), 250 pas, 2 seeds. Évaluation à longueur 32 (vue) puis 64 et 128 (**jamais vues**). Les positions absolues ne peuvent pas dépasser leur table (NaN).

Attention : les phrases du corpus font < 60 caractères, donc l'évaluation à 128 mélange plusieurs phrases ; c'est un test de robustesse à la longueur, pas de dépendance à très longue portée."""),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
BASE_T = dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10)
VARIANTES = {"none": dict(position_mode="none"), "learned": dict(position_mode="learned"),
             "sinusoidal": dict(position_mode="sinusoidal"),                       # embeddings d'écart-type 0.02 : la position (amplitude ~1) les écrase
             "sinusoidal ×√d": dict(position_mode="sinusoidal", emb_scale=True),   # convention du Transformer original
             "rope": dict(position_mode="rope"), "alibi": dict(position_mode="alibi")}
MODES = list(VARIANTES)
res = compare(VARIANTES, DATA, BASE_M, BASE_T, seeds=(0, 1), keep_models=True)'''),
code('''@torch.no_grad()
def eval_derniers(model, L, n_cible=32, iters=12, seed=1234):
    """Perte sur les n_cible DERNIERS tokens de fenêtres de longueur L : même nombre de cibles quel que soit L,
    seul change le contexte disponible (L - n_cible tokens de plus). Évite le biais « début de séquence »."""
    if model.cfg.position_mode in ("learned", "sinusoidal") and L > model.cfg.block_size: return float("nan")
    model.eval(); g = torch.Generator().manual_seed(seed); tot = []
    for _ in range(iters):
        x, y = get_batch(VAL, 16, L, "cpu", g)
        lg = model(x).logits[:, -n_cible:]
        tot.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y[:, -n_cible:].reshape(-1)).item())
    return float(np.mean(tot))

LONGUEURS = [32, 64, 128]
tab = {m: [np.mean([eval_derniers(mod, L) for mod in res[m]["models"]]) for L in LONGUEURS] for m in MODES}
print("Perte sur les 32 derniers tokens, avec un contexte total de L tokens")
print(f"{'position':16}" + "".join(f"{'L='+str(L):>10}" for L in LONGUEURS))
for m, v in tab.items(): print(f"{m:16}" + "".join(f"{x:>10.3f}" for x in v))
plt.figure(figsize=(5.8, 3.4))
for m, v in tab.items(): plt.plot(LONGUEURS, v, "o-", label=m)
plt.xscale("log", base=2); plt.xlabel("longueur d'évaluation (entraînement : 32)"); plt.ylabel("perte de validation"); plt.legend(); plt.show()'''),
md("""**Lecture** : la colonne $L=32$ mesure la qualité à la longueur d'entraînement ; les colonnes $L=64,128$ mesurent la robustesse à un contexte plus long *sans changer le nombre de cibles*. Un chiffre qui **monte** avec $L$ signale une dégradation hors distribution ; un chiffre stable ou qui baisse, un modèle qui gère (ou exploite) le contexte supplémentaire. Ce sont deux questions distinctes de la qualité à $L=32$. Les modes absolus (NaN) sont limités par construction.

## 5. Étendre RoPE *sans réentraîner* (linéaire, NTK, YaRN)
On prend les modèles RoPE entraînés à 32 et on change uniquement les fréquences à l'inférence, pour une évaluation à 64 et 128 (facteur $s=L/32$), avec la même métrique (32 derniers tokens)."""),
code('''def set_rope(model, scaling, factor):
    for blk in model.blocks:
        a = blk.attn; c = model.cfg
        inv, ms = rope_inv_freq(a.hd, c.rope_theta, scaling, factor, orig_len=c.block_size)
        a.inv_freq.copy_(inv); a.mscale = ms

out = {}
for scaling in [None, "linear", "ntk", "yarn"]:
    row = []
    for L in (32, 64, 128):
        vals = []
        for mod in res["rope"]["models"]:
            set_rope(mod, scaling, max(1.0, L / 32)); vals.append(eval_derniers(mod, L))
        row.append(np.mean(vals))
    out[str(scaling)] = row
for mod in res["rope"]["models"]: set_rope(mod, None, 1.0)
print(f"{'RoPE + ':10}" + "".join(f"{'L='+str(L):>10}" for L in (32, 64, 128)))
for k, v in out.items(): print(f"{k:10}" + "".join(f"{x:>10.3f}" for x in v))'''),
md("""**Constats sur l'exécution fournie** (à refaire avec d'autres seeds avant d'en tirer une règle) :
- **NTK-aware** réduit nettement la dégradation à $2\\times$ la longueur d'entraînement, mais plus à $4\\times$.
- **PI (linéaire)** et **YaRN** *dégradent* ici. Deux raisons : (i) ces méthodes sont conçues pour être suivies d'un **court fine-tuning** [5, 6], absent ici ; (ii) les paramètres $\\alpha=1,\\beta=32$ de YaRN supposent un contexte d'origine de plusieurs milliers de tokens ; avec $L_{\\text{train}}=32$ le nombre de tours $r_i$ est petit pour *toutes* les dimensions, donc YaRN se réduit à de l'interpolation pure (+ température).
- **ALiBi** est le seul à rester stable (et même à profiter du contexte supplémentaire) sans aucune modification.

## Bilan
| Méthode | + | − |
|---|---|---|
| Appris | simple, flexible | longueur fixe, pas d'extrapolation |
| Sinusoïdal | sans paramètre | longueur fixe en pratique |
| **RoPE** | relative, extensible (PI/NTK/YaRN), standard (LLaMA, Qwen, Mistral) | extrapolation brute limitée |
| ALiBi | extrapole nativement | biais de récence fort, peu utilisé dans les modèles récents |
| NoPE | aucun mécanisme | à éprouver à grande échelle |

**Limites** : 2 seeds ; un seul corpus synthétique ; modèle minuscule. Pour un résultat publiable : ≥ 5 seeds, vrai corpus, plusieurs tailles.

""" + refs_block("vaswani2017", "su2021", "press2022", "kazemnejad2023", "chen2023pi", "peng2023", "dubey2024")),
]
