from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "16-ablations-finales.ipynb"
TITLE = "Ablations finales : un facteur à la fois autour d'une base LLaMA-style"


def cells():
    return [
md(r"""# 16 · Ablations finales

**Question** — autour d'une base « LLaMA-style » réduite, **quel est l'effet de chaque choix de conception** (position, normalisation, FFN, attention, MoE, optimiseur, schedule) sur la perte de validation ?

**Méthode** — *un facteur à la fois* ; mêmes données, mêmes 3 seeds, même budget ; effet mesuré par la **différence appariée** $\Delta_s=\mathcal L^{(s)}_{\text{variante}}-\mathcal L^{(s)}_{\text{base}}$ (par seed $s$) ; on rapporte $\bar\Delta\pm\text{SE}$ avec $\text{SE}=\text{std}(\Delta_s)/\sqrt{n}$. Verdict prudent : « mieux » / « pire » seulement si $|\bar\Delta|>2\,\text{SE}$, sinon « équivalent (non concluant) ». Avec $n=3$ seeds, ce seuil est **indicatif**, pas un test statistique rigoureux (les seeds partagent les mêmes données ; l'écart-type est estimé sur 3 valeurs)."""),
code(PRELUDE_IMPORTS),
tool(),
code(DATA_CELL),
code('''h = ffn_hidden(ModelConfig(vocab_size=31, block_size=32, n_layer=2, n_head=8, n_embd=64, ffn_type="swiglu"))
BASE_M = dict(block_size=32, n_layer=2, n_head=8, n_embd=64, position_mode="rope", norm_type="rmsnorm", ffn_type="swiglu", attn_type="gqa", n_kv_head=4)
BASE_T = dict(steps=250, batch_size=16, lr=3e-3, weight_decay=0.1, optimizer="adamw", schedule="warmup_cosine", warmup_steps=20, grad_clip=1.0, eval_interval=125, eval_iters=10)
M = lambda **k: dict(model=k, train={})
T = lambda **k: dict(model={}, train=k)
ABLATIONS = {
 "BASE (LLaMA-style)": M(),
 # position
 "position: none": M(position_mode="none"), "position: alibi": M(position_mode="alibi"), "position: learned": M(position_mode="learned"),
 "position: sinusoidal ×√d": M(position_mode="sinusoidal", emb_scale=True),
 # normalisation
 "norme: LayerNorm": M(norm_type="layernorm", bias=True), "norme: Post-LN": M(pre_norm=False), "norme: + QK-Norm": M(qk_norm=True),
 # FFN
 "ffn: GELU": M(ffn_type="gelu"), "ffn: ReLU": M(ffn_type="relu"), "ffn: GeGLU": M(ffn_type="geglu"),
 # attention
 "attn: MHA": M(attn_type="mha"), "attn: MQA": M(attn_type="mqa"), "attn: fenêtre 16": M(window=16),
 # MoE (FFN actif ≈ base)
 "moe: 8e top-2": M(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, balance_coef=0.01),
 "moe: 16e top-3 + 1 partagé, sans perte aux.": M(moe=True, n_experts=16, top_k=3, n_shared_experts=1, expert_hidden=h // 4, router="sigmoid_bias"),
 # optimisation
 "opt: Adam (L2)": T(optimizer="adam"), "opt: Lion (lr/3)": T(optimizer="lion", lr=1e-3), "opt: SGD momentum (lr 0.1)": T(optimizer="sgd_momentum", lr=0.1, weight_decay=0.0),
 "sched: constant": T(schedule="constant"), "sched: WSD": T(schedule="wsd"), "clip: aucun": T(grad_clip=None),
}
SEEDS = (0, 1, 2)
res = compare(ABLATIONS, DATA, BASE_M, BASE_T, seeds=SEEDS)'''),
md("## Effets appariés vs la base"),
code('''base = np.array([r["final_val_loss"] for r in res["BASE (LLaMA-style)"]["runs"]])
lignes = []
for nom, r in res.items():
    if nom.startswith("BASE"): continue
    d = np.array([x["final_val_loss"] for x in r["runs"]]) - base
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    verdict = "pire" if m > 2 * se else ("mieux" if m < -2 * se else "équivalent (non concluant)")
    lignes.append((nom, m, se, verdict, r["params"], r["active"]))
print(f"base : {base.mean():.4f} ± {base.std(ddof=1):.4f}\\n")
print(f"{'variante':46}{'Δ moyen':>9}{'± SE':>8}  {'verdict':28}{'params':>9}{'actifs':>9}")
for nom, m, se, v, p, a in lignes: print(f"{nom:46}{m:>+9.4f}{se:>8.4f}  {v:28}{p:>9,}{a:>9,}")'''),
code('''noms = [l[0] for l in lignes]; ms = np.array([l[1] for l in lignes]); ses = np.array([l[2] for l in lignes])
ordre = np.argsort(ms); plt.figure(figsize=(8, 6.5))
plt.barh([noms[i] for i in ordre], ms[ordre], xerr=2 * ses[ordre], color=["tab:green" if ms[i] < -2 * ses[i] else "tab:red" if ms[i] > 2 * ses[i] else "tab:gray" for i in ordre], capsize=2)
plt.axvline(0, color="k", lw=1); plt.xlabel("Δ perte de validation vs base (barres = ±2 SE) ; < 0 = mieux"); plt.title("Ablations : un facteur à la fois"); plt.tight_layout(); plt.show()'''),
md(r"""## Constats sur l'exécution fournie
**Effets grands (≫ bruit ; cohérents avec les notebooks précédents)**
- Position : *none*, *learned*, *sinusoïdal ×√d* sont bien **pires** que RoPE (+0,27 à +0,78) sur ce budget de 250 pas ; ALiBi est un peu **meilleur** (−0,055).
- **Post-LN : +0,42** (cf. notebook 08).
- Optimisation : **Adam avec $L_2$ à $\lambda=0{,}1$ (+1,8)**, SGD momentum (+0,85) et Lion à lr non réglé (+0,23) sont bien pires qu'AdamW. Attention : pour Adam-$L_2$ c'est précisément l'effet décrit au notebook 11 (le terme $\lambda\theta$ domine les petits gradients) ; SGD et Lion n'ont **pas** de lr réglé ici, donc ce n'est **pas** un verdict sur ces optimiseurs.
- GELU : +0,09 (cohérent avec le notebook 09).

**Effets petits (|Δ| ≲ 0,05) : à considérer fragiles**
- QK-Norm (−0,057) : cohérent avec les notebooks 06 et 08, donc plutôt crédible.
- MHA « mieux » que GQA (−0,045) : à relativiser, MHA a **plus de paramètres** (102,7 k contre 94,5 k) ; LayerNorm « mieux » (−0,031) a aussi plus de paramètres (biais).
- Fenêtre 16 « mieux » (−0,027) : propriété du corpus synthétique local (cf. notebook 06).
- **MoE 16e + partagé « mieux » (−0,034)** : atteint le seuil de 2 SE ici, mais le notebook 10 (autre base) donne un écart nul pour le même type de MoE, et le routeur sans perte auxiliaire y était même légèrement *moins bon* : **résultat non robuste**, et 3× plus de paramètres totaux.
- **Schedule constant « mieux » que warmup-cosinus (−0,026)** ici, à l'inverse du notebook 12 (lr 6e-3, 300 pas : le constant y est nettement pire). Le sens de l'effet dépend du lr et du budget ; ne pas en tirer de règle. WSD (−0,073) est le plus robuste des schedules dans ces deux notebooks.
- Écrêtage : sans effet distinguable à ce lr (le notebook 12 le montre utile à lr élevé).

## Lecture et garde-fous
- **Vert / rouge** : effet dont l'amplitude dépasse 2 SE dans cette exécution ; **gris** : indistinguable du bruit avec 3 seeds.
- Un effet « pire » pour SGD ou Lion peut refléter un **lr mal réglé** (le notebook 11 règle le lr par optimiseur ; ici on garde un lr par défaut pour isoler le facteur). Idem pour les MoE : à si petite échelle et 250 pas, les experts n'ont pas le temps de se spécialiser.
- Les **interactions** entre facteurs ne sont pas mesurées (un facteur à la fois).
- Pour un manuscrit : ≥ 5 seeds, corpus réel, budget de tokens proche du régime compute-optimal [2] (≈ 20 tokens par paramètre), sélection des hyperparamètres sur validation et rapport sur test.

## Protocole reproductible
Chaque ligne est un dictionnaire `{model: {...}, train: {...}}` : ajouter une ligne à `ABLATIONS` suffit pour tester un nouveau facteur ; le même protocole est disponible en ligne de commande (`scripts/run_experiments.py --field ... --values ...`).

**Limites** : petite échelle ($\sim10^5$ paramètres, 250 pas), corpus synthétique, 3 seeds, un seul facteur à la fois, hyperparamètres non réglés par variante, **aucune correction pour comparaisons multiples** (21 variantes ⇒ quelques faux positifs attendus).

""" + refs_block("kaplan2020", "hoffmann2022", "loshchilov2019", "shazeer2020", "su2021", "zhang2019", "ainslie2023", "fedus2022")),
]
