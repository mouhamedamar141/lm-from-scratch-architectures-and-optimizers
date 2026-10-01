from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "10-mixture-of-experts.ipynb"
TITLE = "Mixture of Experts : routage, équilibrage, variantes"


def cells():
    return [
md("""# 10 · Mixture of Experts (MoE)

**À retenir**
- Un MoE remplace le FFN par $E$ experts $\\{f_i\\}$ et un **routeur** ; chaque token n'active que $k\\ll E$ experts :
$$y=\\sum_{i\\in\\text{TopK}(s)} g_i\\,f_i(x)\\ \\ \\Big(+\\sum_{j\\in\\text{partagés}} f_j(x)\\Big),\\qquad s=\\mathrm{softmax}(W_r x)$$
- Paramètres **totaux** ↑ (capacité), calcul **actif** par token ≈ constant. Coût réel : mémoire, communication, **équilibrage de charge**.
- **Perte d'équilibrage** (Switch [3]) : $\\;\\mathcal L_{\\text{bal}}=\\alpha\\,E\\sum_{i=1}^{E} f_i\\,P_i$, avec $f_i$ = fraction de tokens envoyés à $i$, $P_i$ = probabilité moyenne du routeur pour $i$. Minimum $=\\alpha$ si la charge est uniforme.
- **z-loss** (ST-MoE [4]) : $\\mathcal L_z=\\beta\\,\\mathbb E\\big[(\\log\\sum_i e^{z_i})^2\\big]$ : stabilise les logits du routeur.
- **Sans perte auxiliaire** (DeepSeek-V3 [8, 9]) : sélection par $s_i+b_i$, poids de mélange $g_i$ issus de $s_i$ seul ; $b_i\\leftarrow b_i+\\gamma\\,\\text{sign}(\\bar c-c_i)$ (charge $c_i$ vs moyenne $\\bar c$).
- **Expert Choice** [5] : chaque expert choisit ses $c=kN/E$ tokens ⇒ charge parfaite, mais **non causal** (inutilisable tel quel en décodage autorégressif).
- **Experts fins + partagés** (DeepSeekMoE [7]) : plus d'experts plus petits, et quelques experts toujours actifs pour le savoir commun.
- **Capacité** [2, 3] : au plus $C=\\lceil \\text{cf}\\cdot kT/E\\rceil$ tokens par expert ; le surplus est *abandonné* (résiduel seul)."""),
code(PRELUDE_IMPORTS),
tool(),
md("""## 1. Vérification : le MoE de la boîte à outils = définition mathématique
On compare à une référence **naïve** (boucle sur les tokens, sans regroupement) avec les mêmes poids."""),
code('''torch.manual_seed(0)
cfg = ModelConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2, n_embd=16, moe=True, n_experts=4, top_k=2, expert_hidden=16)
moe = MoE(cfg).eval(); x = torch.randn(2, 5, 16)

def moe_naif(moe, x, k):
    flat = x.reshape(-1, x.size(-1)); out = torch.zeros_like(flat)
    for t in range(flat.size(0)):
        s = moe.router(flat[t]).softmax(-1); w, idx = s.topk(k); w = w / w.sum()
        for wi, i in zip(w, idx): out[t] += wi * moe.experts[i](flat[t])
    return out.view_as(x)

y, _ = moe(x)
print("MoE optimisé == référence naïve :", torch.allclose(y, moe_naif(moe, x, 2), atol=1e-5))
print("charge par expert (somme = 1)   :", moe.stats["load"].numpy().round(3))'''),
md("""## 2. Perte d'équilibrage : valeur minimale théorique
Avec un routeur uniforme, $f_i=P_i=1/E$ donc $E\\sum f_iP_i=1$. Un routeur qui concentre la charge sur un expert donne une valeur strictement plus grande (démonstration ci-dessous : entrées de moyenne positive, un expert dominant)."""),
code('''moe = MoE(ModelConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2, n_embd=16, moe=True, n_experts=4, top_k=2, expert_hidden=16, balance_coef=1.0)).train()
x = torch.randn(4, 32, 16)
torch.nn.init.zeros_(moe.router.weight); print("routeur uniforme    : L_bal =", round(moe(x)[1].item(), 4), "(minimum théorique 1.0)")
with torch.no_grad(): moe.router.weight.zero_(); moe.router.weight[0] = 1.0        # l'expert 0 domine pour des entrées de moyenne positive
print("routeur déséquilibré: L_bal =", round(moe(x + 1.0)[1].item(), 4), "| charge :", moe.stats["load"].numpy().round(2), "(le maximum possible pour un expert est 1/k = 0.5)")'''),
md("""## 3. Expérience : à **paramètres actifs égaux**, quelles variantes de MoE ?
Baseline dense SwiGLU (largeur $h=176$). Chaque variante MoE garde **le même calcul FFN actif par token** ($k\\times$ largeur d'expert $=h$) mais un nombre total de paramètres plus élevé. Mesures : perte de validation, paramètres totaux/actifs, **déséquilibre de charge** (`max(charge) × E`, 1 = parfait) et **fraction de tokens abandonnés**.

Configuration : $d=64$, 2 couches, 250 pas, 2 seeds."""),
code(DATA_CELL),
code('''h = ffn_hidden(ModelConfig(vocab_size=31, block_size=32, n_layer=2, n_head=4, n_embd=64, ffn_type="swiglu"))
print("largeur FFN dense h =", h)
BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64, ffn_type="swiglu")
BASE_T = dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10)
VARS = {
 "dense":                        dict(),
 "MoE 8e top2 (bal 0.01)":       dict(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, balance_coef=0.01),
 "MoE 8e top2 (sans bal)":       dict(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, balance_coef=0.0),
 "MoE 8e top2 + z-loss":         dict(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, balance_coef=0.01, z_loss_coef=1e-3),
 "MoE 8e top2 sans perte aux.":  dict(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, router="sigmoid_bias"),
 "MoE 8e top2 capacité 1.0":     dict(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, balance_coef=0.01, capacity_factor=1.0),
 "MoE expert-choice":            dict(moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, router="expert_choice"),
 "MoE fin 16e top4":             dict(moe=True, n_experts=16, top_k=4, expert_hidden=h // 4, balance_coef=0.01),
 "MoE fin 16e top3 + 1 partagé": dict(moe=True, n_experts=16, top_k=3, n_shared_experts=1, expert_hidden=h // 4, balance_coef=0.01),
}
res = compare(VARS, DATA, BASE_M, BASE_T, seeds=(0, 1), keep_models=True)'''),
code('''@torch.no_grad()
def diagnostics(model):
    model.eval(); g = torch.Generator().manual_seed(7); x, y = get_batch(VAL, 32, 32, "cpu", g); model(x, y)
    st = model.moe_stats()
    if not st: return None
    E = model.cfg.n_experts
    return dict(desequilibre=float(np.mean([s["load"].max().item() * E for s in st])), abandon=float(np.mean([s["dropped"] for s in st])),
                charges=[s["load"].numpy() for s in st])

print(f"{'variante':30}{'déséquilibre':>14}{'tokens abandonnés':>20}")
diag = {}
for n, r in res.items():
    d = [diagnostics(m) for m in r["models"]]; diag[n] = d
    if d[0]: print(f"{n:30}{np.mean([x['desequilibre'] for x in d]):>14.2f}{100*np.mean([x['abandon'] for x in d]):>19.1f}%")'''),
code('''noms = ["MoE 8e top2 (sans bal)", "MoE 8e top2 (bal 0.01)", "MoE 8e top2 sans perte aux.", "MoE expert-choice"]
fig, ax = plt.subplots(1, 4, figsize=(13, 2.8), sharey=True)
for a, n in zip(ax, noms):
    c = diag[n][0]["charges"][-1]; a.bar(range(len(c)), c); a.axhline(1 / len(c), color="k", ls="--", lw=1); a.set_title(n, fontsize=8); a.set_xlabel("expert")
ax[0].set_ylabel("part des tokens (dernière couche)"); plt.tight_layout(); plt.show()'''),
md("""## 4. Lecture des résultats (exécution fournie) et garde-fous
**Constats** (2 seeds ; écart-type élevé pour plusieurs variantes, relire le tableau après toute ré-exécution) :
1. **Dense vs MoE à calcul actif égal** : le MoE top-2 avec perte d'équilibrage est *indiscernable* du dense (écart ≈ 0,002, très inférieur au bruit). Aucune variante de MoE n'apporte un gain net ici : à cette échelle (10⁵ paramètres, 250 pas, corpus simple), les experts n'ont ni le temps ni le besoin de se spécialiser. La littérature observe les gains du MoE à **grande échelle et grand nombre de tokens** [2, 3, 6, 10] : ce résultat n'est ni une confirmation ni une réfutation.
2. **L'équilibrage fonctionne comme prévu** : sans perte d'équilibrage, la charge est nettement déséquilibrée (indicateur `max(charge)·E` ≈ 2 pour 1 = parfait) ; la perte Switch la ramène vers ≈ 1,4 (la variante z-loss l'inclut aussi, d'où le même niveau) ; le routeur à biais (sans perte auxiliaire) vers ≈ 1,1 ; Expert Choice vaut 1 par construction. Mais **cet équilibrage ne se traduit pas ici par un gain de perte**.
3. **Capacité 1,0 : coûteuse** : ≈ 9 % des tokens abandonnés et une perte nettement plus élevée que sans capacité (écart ≫ bruit). Compromis vitesse/qualité à ne pas négliger.
4. **Expert Choice** : perte plus élevée ; de plus **non causal** (le choix des tokens dépend du lot), donc inutilisable en génération sans adaptation.
5. **Experts fins + 1 partagé** : la meilleure moyenne du tableau, mais l'écart avec le dense (≈ −0,01) est de l'ordre de l'écart-type : **non concluant**.
6. **Débit** : les MoE sont ~30–50 % plus lents en tokens/s à cause de la boucle Python sur les experts (implémentation non groupée) ; ne pas l'interpréter comme le coût sur GPU.

**Limites** : 2 seeds ; modèle à 2 couches et $d=64$ ; corpus synthétique dont la structure (quelques types de phrases) est plus simple que celle du langage, donc la « spécialisation » des experts n'y est pas représentative ; boucle sur les experts non optimisée (le débit `tok/s` **n'est pas** représentatif de noyaux GPU groupés).

## Bilan
| Variante | Idée | Modèles |
|---|---|---|
| Top-2 + perte Switch | équilibre par pénalité | Mixtral [6], Switch [3] |
| z-loss | stabilité du routeur | ST-MoE [4] |
| Sans perte auxiliaire | biais de sélection ajusté | DeepSeek-V3 [8, 9] |
| Experts fins + partagés | spécialisation + savoir commun | DeepSeekMoE [7], OLMoE [10] |
| Expert Choice | charge parfaite | [5] (non causal) |

""" + refs_block("shazeer2017", "lepikhin2021", "fedus2022", "zoph2022", "zhou2022", "jiang2024", "dai2024", "deepseekv3", "wang2024", "muennighoff2024")),
]
