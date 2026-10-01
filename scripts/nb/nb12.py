from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "12-schedules-clipping-precision.ipynb"
TITLE = "Schedules de lr, écrêtage, accumulation de gradient, précision mixte"


def cells():
    return [
md(r"""# 12 · Schedules, écrêtage du gradient, accumulation, précision mixte

**À retenir**
- **Warmup** [2] : monter linéairement le lr sur $T_w$ pas évite les premiers pas instables (estimations de $v$ d'Adam mal calibrées).
- Schedules usuels ($p=\frac{t-T_w}{T-T_w}\in[0,1]$, plancher $\rho=\eta_{\min}/\eta_{\max}$) :
$$\text{cosinus [1]: }\ \eta_t=\eta_{\max}\Big[\rho+\tfrac{1-\rho}{2}\big(1+\cos\pi p\big)\Big]\qquad \text{linéaire: }\ \eta_t=\eta_{\max}\big[1-(1-\rho)p\big]$$
$$\text{inv-sqrt: }\ \eta_t=\eta_{\max}\sqrt{T_w/t}\qquad \textbf{WSD [5]: plateau }\eta_{\max}\text{ puis décroissance linéaire sur les }\delta T\text{ derniers pas}$$
- **Écrêtage** [3] : $g\leftarrow g\cdot\min\!\big(1,\ c/\|g\|_2\big)$ — protège contre les pics de gradient.
- **Accumulation** : $K$ micro-lots de taille $b$ ≡ un lot de taille $Kb$ (en divisant chaque perte par $K$).
- **Précision mixte** [4] : calculs en bf16/fp16, poids maître et statistiques en fp32 ; fp16 nécessite un *loss scaling*, pas bf16 (même plage dynamique que fp32, mantisse plus courte).
- **WSD** permet de prolonger l'entraînement sans connaître le budget à l'avance (le plateau peut être repris)."""),
code(PRELUDE_IMPORTS),
tool(),
md("## 1. Les schedules"),
code('''T, Tw = 300, 20
plt.figure(figsize=(7, 3.6))
for nom in ["constant", "linear", "cosine", "warmup_cosine", "warmup_linear", "inv_sqrt", "wsd"]:
    plt.plot([lr_factor(nom, t, T, Tw, 0.1, 0.2) for t in range(T)], label=nom)
plt.xlabel("pas t"); plt.ylabel("η_t / η_max"); plt.legend(fontsize=7); plt.title("Schedules de taux d'apprentissage"); plt.show()'''),
md("## 2. Expérience : effet du schedule (mêmes pic de lr, données, seeds)"),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
BASE_T = dict(steps=300, batch_size=16, lr=6e-3, warmup_steps=20, eval_interval=100, eval_iters=10)
res = compare({s: dict(model={}, train=dict(schedule=s)) for s in ["constant", "linear", "cosine", "warmup_cosine", "warmup_linear", "inv_sqrt", "wsd"]}, DATA, BASE_M, BASE_T, seeds=(0, 1))'''),
md(r"""**Constats (exécution fournie)** : tous les schedules qui **décroissent** vers un plancher (linéaire, cosinus, WSD, avec ou sans warmup) battent nettement le lr constant (≈ −0,07 à −0,08) et sont proches entre eux (écarts ≲ 0,01, du même ordre que le bruit) ; **inv-sqrt** est intermédiaire. WSD est compétitif tout en autorisant un plateau extensible [5]. À ce budget de 300 pas, ne pas classer finement les schedules décroissants entre eux.

## 3. Effet du warmup à lr élevé
Pic $\eta=1.5\times10^{-2}$ (agressif) avec 0, 10, 40 pas de warmup."""),
code('''res_w = compare({f"warmup {w}": dict(model={}, train=dict(warmup_steps=w, lr=1.5e-2)) for w in (0, 10, 40)}, DATA, BASE_M, dict(steps=200, batch_size=16, eval_interval=100, eval_iters=10), seeds=(0, 1))'''),
md(r"""## 4. Écrêtage du gradient
Norme du gradient au cours de l'entraînement à lr élevé, sans écrêtage puis avec $c=1$ et $c=0.3$."""),
code('''res_c = compare({"sans écrêtage": dict(model={}, train=dict(grad_clip=None)), "c = 1.0": dict(model={}, train=dict(grad_clip=1.0)), "c = 0.3": dict(model={}, train=dict(grad_clip=0.3))},
                DATA, BASE_M, dict(steps=200, batch_size=16, lr=1.5e-2, warmup_steps=10, eval_interval=20, eval_iters=5), seeds=(0, 1))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
plot_curves(res_c, key="grad_norm", ax=ax[0], title="norme du gradient (avant écrêtage)"); plot_curves(res_c, key="val_loss", ax=ax[1], title="perte de validation"); plt.tight_layout(); plt.show()'''),
md(r"""*La norme tracée est celle **avant** écrêtage (valeur renvoyée par `clip_grad_norm_`)* : elle montre les pics que l'écrêtage neutralise.

## 5. Accumulation de gradient : équivalence exacte
Sur un même lot de 8 séquences : gradient du lot complet vs somme de 2 micro-lots de 4 (chaque perte divisée par 2)."""),
code('''torch.manual_seed(0)
m = LM(ModelConfig(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, n_embd=64)).eval()
g = torch.Generator().manual_seed(0); x, y = get_batch(TRAIN, 8, 32, "cpu", g)
m.zero_grad(); m(x, y).loss.backward(); g_full = torch.cat([p.grad.flatten() for p in m.parameters()])
m.zero_grad()
for sl in (slice(0, 4), slice(4, 8)): (m(x[sl], y[sl]).loss / 2).backward()
g_acc = torch.cat([p.grad.flatten() for p in m.parameters()])
print("écart max des gradients :", (g_full - g_acc).abs().max().item(), "| cosinus :", torch.nn.functional.cosine_similarity(g_full, g_acc, dim=0).item())'''),
md(r"""Égalité à l'arrondi flottant près : l'accumulation n'est pas une approximation **sauf** pour les couches qui dépendent de statistiques *du lot* (BatchNorm, MoE avec capacité par lot, Expert Choice). Les LayerNorm/RMSNorm des Transformers sont par token, donc exactes.

## 6. Précision mixte
### 6.1 Représentation
bf16 : 8 bits d'exposant, **7 bits de mantisse** (précision relative $\approx 2^{-8}$) ; fp16 : 5 bits d'exposant, 10 bits de mantisse (plage $\approx 6\times10^{4}$ max, $6\times10^{-8}$ min sub-normal)."""),
code('''for v in [1 + 2**-8, 1 + 2**-10, 70000.0, 1e-8]:
    t = torch.tensor(v)
    print(f"{v!r:22} fp32={t.item():<14.10g} bf16={t.to(torch.bfloat16).item():<14.10g} fp16={t.to(torch.float16).item():<14.10g}")'''),
md("### 6.2 Effet sur l'entraînement (autocast CPU bf16 vs fp32)"),
code('''res_p = compare({"fp32": dict(model={}, train=dict(dtype="float32")), "bf16 (autocast)": dict(model={}, train=dict(dtype="bfloat16"))},
                DATA, BASE_M, dict(steps=200, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=100, eval_iters=10), seeds=(0, 1))'''),
md(r"""**Lecture (exécution fournie)** : la qualité est préservée (écart de perte ≪ écart-type). Le débit dépend fortement du matériel : sur le CPU d'exécution fourni le bf16 est plus rapide (≈ 1,5×, processeur avec instructions bf16), mais sur d'autres CPU il peut être **plus lent** que fp32 ; le gain de vitesse et de mémoire est avant tout un phénomène GPU/TPU. Ne pas généraliser cette mesure. fp16 sur GPU exige un `GradScaler` (intégré dans `TrainConfig(dtype="float16")`, non testable sur CPU ici).

## Bilan
| Levier | Recommandation | Point d'attention |
|---|---|---|
| Warmup | toujours (≈ 1–5 % des pas) | plus important à lr élevé / Post-LN |
| Décroissance | cosinus ou WSD jusqu'à 10 % de $\eta_{\max}$ | WSD si budget incertain [5] |
| Écrêtage | $c=1$ | surveiller la fréquence d'écrêtage |
| Accumulation | pour émuler de grands lots | exacte sauf statistiques de lot |
| bf16 | par défaut sur GPU récents | garder fp32 pour softmax/normes/perte |

**Limites** : 300 pas seulement, petit modèle, 2 seeds ; effets de schedule à grande échelle (dizaines de milliers de pas) non couverts ; mesures de vitesse CPU non représentatives des GPU.

""" + refs_block("loshchilov2017", "goyal2017", "pascanu2013", "micikevicius2018", "hu2024", "hoffmann2022")),
]
