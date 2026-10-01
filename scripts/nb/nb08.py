from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "08-normalisation-et-residus.ipynb"
TITLE = "Normalisations (LayerNorm, RMSNorm), Pre-LN vs Post-LN, QK-Norm"


def cells():
    return [
md("""# 08 · Normalisation et connexions résiduelles

**À retenir**
$$\\text{LayerNorm}(x)=\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\varepsilon}}\\odot\\gamma+\\beta\\qquad
\\text{RMSNorm}(x)=\\frac{x}{\\sqrt{\\tfrac1d\\sum_i x_i^2+\\varepsilon}}\\odot\\gamma$$
avec $\\mu=\\tfrac1d\\sum x_i$, $\\sigma^2=\\tfrac1d\\sum (x_i-\\mu)^2$.
- **RMSNorm** [2] supprime le recentrage et le biais : moins de calcul, qualité comparable ; standard dans LLaMA, Mistral, Qwen, DeepSeek.
- **Post-LN** (Transformer original [4]) : $x_{l+1}=\\text{Norm}(x_l+F(x_l))$. **Pre-LN** : $x_{l+1}=x_l+F(\\text{Norm}(x_l))$ — laisse un chemin résiduel *identité* jusqu'à la sortie, donc entraînement plus stable et moins dépendant du warmup [3].
- **QK-Norm** [5, 6] : normaliser $q,k$ borne les logits d'attention.
- Évolutions récentes du chemin résiduel : Hyper-Connections [7] et mHC [8] (résiduel élargi avec contraintes de stabilité) — hors périmètre expérimental ici."""),
code(PRELUDE_IMPORTS),
tool(),
md("""## 1. Implémentations vérifiées
Les classes de la boîte à outils sont comparées à `torch.nn.LayerNorm` ; on vérifie aussi la propriété clé de RMSNorm (invariance d'échelle) et la différence avec LayerNorm (invariance de translation)."""),
code('''torch.manual_seed(0)
x = torch.randn(4, 10, 32) * 3 + 1.5
ln, ref = LayerNorm(32, bias=True), torch.nn.LayerNorm(32, eps=1e-5)
print("LayerNorm == torch.nn.LayerNorm :", torch.allclose(ln(x), ref(x), atol=1e-5))
rms = RMSNorm(32)
print("RMSNorm invariante à l'échelle   :", torch.allclose(rms(x), rms(10 * x), atol=1e-4))
print("LayerNorm invariante à la translation :", torch.allclose(ln(x), ln(x + 5.0), atol=1e-4), "| RMSNorm :", torch.allclose(rms(x), rms(x + 5.0), atol=1e-2))
print("paramètres : LayerNorm", sum(p.numel() for p in ln.parameters()), "| RMSNorm", sum(p.numel() for p in rms.parameters()))
import time
big = torch.randn(64, 256, 512)
def chrono(mod, reps=15):
    mod(big)                                              # échauffement (allocations, caches)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); mod(big); t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))
for _ in range(2):                                        # deux passes alternées : l'ordre ne doit pas influencer
    print(" | ".join(f"{n}: {chrono(m):.1f} ms" for n, m in [("LayerNorm (main)", LayerNorm(512)), ("RMSNorm (main)", RMSNorm(512)), ("nn.LayerNorm (fusionné)", torch.nn.LayerNorm(512))]))'''),
md("""## 2. Pre-LN vs Post-LN : normes de gradient par couche à l'initialisation
Modèle de 8 couches, sans entraînement ; norme du gradient de **tous** les paramètres de chaque bloc, moyenne sur 3 seeds. L'analyse de [3] prédit des gradients bien plus homogènes en Pre-LN qu'en Post-LN (à l'initialisation)."""),
code('''def grad_par_couche(pre_norm, n_layer=8, seed=0):
    torch.manual_seed(seed)
    m = LM(ModelConfig(vocab_size=31, block_size=32, n_layer=n_layer, n_head=4, n_embd=64, norm_type="layernorm", bias=True, pre_norm=pre_norm))
    x = torch.randint(0, 31, (16, 32)); m(x, x).loss.backward()
    return [np.sqrt(sum((p.grad ** 2).sum().item() for p in b.parameters())) for b in m.blocks]

plt.figure(figsize=(5.8, 3.3))
for pre in (True, False):
    g = np.mean([grad_par_couche(pre, seed=s) for s in range(3)], axis=0)
    plt.plot(range(1, 9), g, "o-", label="Pre-LN" if pre else "Post-LN")
    print(f"{'Pre-LN' if pre else 'Post-LN'} : gradient couche 8 / couche 1 = {g[-1] / g[0]:.2f}")
plt.xlabel("couche (1 = entrée)"); plt.ylabel("norme du gradient"); plt.yscale("log"); plt.legend(); plt.title("Gradient par couche à l'initialisation (moy. 3 seeds)"); plt.show()'''),
md("""**Lecture (exécution fournie)** : le rapport « gradient de la couche 8 / gradient de la couche 1 » vaut ≈ 0,56 en Pre-LN et ≈ 0,04 en Post-LN : les gradients sont **beaucoup plus homogènes en Pre-LN**, ce qui va dans le sens de [3]. En revanche, ici les gradients Post-LN sont *plus faibles* près de la sortie ; ce n'est pas une reproduction du résultat détaillé de [3] (qui porte sur les paramètres de la dernière couche et un schéma d'initialisation précis) : notre mesure agrège tous les paramètres du bloc avec l'initialisation de ce dépôt (sorties résiduelles réduites par $1/\\sqrt{2L}$). Ne pas citer cette figure comme confirmation quantitative du théorème.

## 3. Expérience d'entraînement : stabilité selon la normalisation
Deux régimes : **lr modéré avec warmup** (usuel) et **lr élevé sans warmup** (stress-test de stabilité). Modèle de 4 couches, mêmes données et seeds."""),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=4, n_head=4, n_embd=64)
VARS = {"Post-LN + LayerNorm": dict(norm_type="layernorm", bias=True, pre_norm=False),
        "Pre-LN + LayerNorm": dict(norm_type="layernorm", bias=True, pre_norm=True),
        "Pre-LN + RMSNorm": dict(norm_type="rmsnorm", pre_norm=True),
        "Pre-LN + RMSNorm + QK-Norm": dict(norm_type="rmsnorm", pre_norm=True, qk_norm=True)}
print(">>> Régime A : lr=3e-3, warmup 20 pas")
resA = compare(VARS, DATA, BASE_M, dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10), seeds=(0, 1), keep_models=True)
print("\\n>>> Régime B (stress) : lr=1.5e-2, SANS warmup")
resB = compare(VARS, DATA, BASE_M, dict(steps=250, batch_size=16, lr=1.5e-2, warmup_steps=0, eval_interval=125, eval_iters=10), seeds=(0, 1))'''),
code('''fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
plot_curves(resA, ax=ax[0], title="A : lr 3e-3 + warmup"); plot_curves(resB, ax=ax[1], title="B : lr 1.5e-2, sans warmup"); plt.tight_layout(); plt.show()'''),
md("""**Constats sur l'exécution fournie** (2 seeds : à confirmer) :
- **Post-LN est nettement moins bon** que Pre-LN dès le régime A, et se dégrade fortement en régime B (lr élevé, sans warmup), là où les Pre-LN restent utilisables.
- **RMSNorm ≈ LayerNorm** en régime A (écart dans le bruit). En B les moyennes sont proches mais la dispersion entre seeds diffère (voir colonne ±) ; avec 2 seeds, ne pas en déduire une supériorité.
- **QK-Norm** est la meilleure variante dans les deux régimes, avec un écart supérieur au bruit.
Le classement peut dépendre du régime : c'est pourquoi il faut tester la stabilité et pas seulement la performance à hyperparamètres « favorables ».

## 4. Norme du flux résiduel *après entraînement* (Pre-LN + RMSNorm)
En Pre-LN chaque couche **ajoute** sa contribution au résiduel. À l'initialisation les contributions sont minuscules (écart-type des poids 0,02, sorties résiduelles réduites par $1/\\sqrt{2L}$) ; c'est après entraînement que l'on observe l'évolution de la norme avec la profondeur, ce qui justifie la norme finale avant la tête de sortie."""),
code('''m = resA["Pre-LN + RMSNorm"]["models"][0].eval()
x = m.tok_emb(torch.randint(0, VOCAB, (16, 32))); normes = [x.norm(dim=-1).mean().item()]
with torch.no_grad():
    for b in m.blocks: x, _ = b(x); normes.append(x.norm(dim=-1).mean().item())
print("norme moyenne du résiduel (embedding, puis après chaque couche) :", [round(n, 2) for n in normes])'''),
md("""## Bilan
| Choix | Recommandation courante | Raison |
|---|---|---|
| Position de la norme | **Pre-LN** | gradients plus uniformes, moins de dépendance au warmup [3] |
| Type | **RMSNorm** | moins de calcul, pas de perte de qualité notable [2] ; dans la mesure du §1, environ 2× plus rapide que notre LayerNorm écrit à la main, mais plus lent que `nn.LayerNorm` fusionné : le gain réel dépend des noyaux disponibles |
| QK-Norm | à grande échelle / lr élevé | stabilise les logits d'attention [5, 6] |
| Norme finale | obligatoire en Pre-LN | le résiduel s'accumule couche après couche (cf. §4) |

**Limites** : 4 couches seulement (les effets de profondeur décrits dans [3] se manifestent surtout pour beaucoup plus de couches) ; 2 seeds ; une seule tâche synthétique.

""" + refs_block("ba2016", "zhang2019", "xiong2020", "vaswani2017", "henry2020", "dehghani2023", "zhu2024hc", "xie2025mhc", "he2016")),
]
