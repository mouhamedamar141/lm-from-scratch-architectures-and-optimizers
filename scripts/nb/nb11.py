from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "11-optimiseurs-sgd-adam-adamw.ipynb"
TITLE = "Optimiseurs : SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, Lion, Muon"


def cells():
    return [
md(r"""# 11 · Optimiseurs : SGD, Adam, AdamW (et variantes)

**À retenir**
- **SGD** : un pas unique $\eta$ pour tous les paramètres ; sensible au conditionnement.
- **Momentum / Nesterov** : accumulent une vitesse pour traverser les vallées étroites.
- **Adagrad / RMSprop / Adam** : pas **adaptatif par coordonnée** (division par une estimation de $\sqrt{\mathbb E[g^2]}$).
- **AdamW** [7] : Adam avec **weight decay découplé** — le standard de l'entraînement des LLM.
- **Lion** [8] (signe du momentum), **Muon** [9, 10] (orthogonalisation des matrices) : optimiseurs récents.
- Une comparaison honnête exige de **régler le taux d'apprentissage de chaque optimiseur** (sinon on compare des réglages, pas des algorithmes).

Notations : $g_t=\nabla_\theta\mathcal L_{B_t}(\theta_t)$ (gradient sur le mini-lot), $\eta$ pas, $\lambda$ weight decay, $\varepsilon$ petit terme de stabilité, $t\ge1$ indice du pas, opérations vectorielles *terme à terme*."""),
md(r"""## 1. Formules

**SGD**
$$\theta_{t+1}=\theta_t-\eta\,g_t$$

**SGD avec momentum** (Polyak [1, 2] ; convention de PyTorch)
$$v_t=\mu\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t$$

**Nesterov** [2, 3] (forme PyTorch, gradient « anticipé »)
$$v_t=\mu\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,(g_t+\mu\,v_t)$$

**Adagrad** [4]
$$G_t=G_{t-1}+g_t^{2},\qquad \theta_{t+1}=\theta_t-\eta\,\frac{g_t}{\sqrt{G_t}+\varepsilon}$$

**RMSprop** [5]
$$s_t=\alpha\,s_{t-1}+(1-\alpha)\,g_t^{2},\qquad \theta_{t+1}=\theta_t-\eta\,\frac{g_t}{\sqrt{s_t}+\varepsilon}$$

**Adam** [6]
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^{2}$$
$$\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}},\qquad \theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$$
*Correction de biais* : $m_0=v_0=0$ donne $\mathbb E[m_t]=(1-\beta_1^t)\,\mathbb E[g]$ ; diviser par $1-\beta_1^t$ rend l'estimation non biaisée. Sans elle, les premiers pas seraient trop petits (surtout pour $v$ avec $\beta_2=0.999$).

**Adam avec régularisation $L_2$** : on remplace $g_t\leftarrow g_t+\lambda\theta_t$ **avant** de calculer $m_t,v_t$.

**AdamW** [7] (weight decay **découplé**)
$$\theta_{t+1}=\theta_t-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}+\lambda\,\theta_t\right)$$

*Pourquoi cela change tout* : en Adam-$L_2$, le terme $\lambda\theta_t$ est divisé par $\sqrt{\hat v_t}$ : un paramètre à gros gradient est **moins régularisé** et un paramètre à petit gradient **plus**. En AdamW la décroissance est $\eta\lambda\theta_t$ pour tous, indépendamment de l'échelle du gradient (démonstration numérique §4).

**Lion** [8]
$$u_t=\mathrm{sign}\big(\beta_1 m_{t-1}+(1-\beta_1)g_t\big),\quad \theta_{t+1}=\theta_t-\eta\,(u_t+\lambda\theta_t),\quad m_t=\beta_2 m_{t-1}+(1-\beta_2)g_t$$
Chaque coordonnée bouge de $\pm\eta$ : utiliser un $\eta$ typiquement 3 à 10× plus petit qu'AdamW.

**Muon** [9, 10] (matrices 2D $W\in\mathbb R^{n\times m}$ des couches cachées)
$$M_t=\mu M_{t-1}+G_t,\qquad O_t=\mathrm{NS}\big(G_t+\mu M_t\big)\approx U V^\top\ (\text{SVD de }G_t+\mu M_t),\qquad W_{t+1}=W_t-\eta\,c\,O_t$$
où $\mathrm{NS}$ est l'itération de Newton–Schulz ($X\leftarrow aX+b\,XX^\top X+c\,(XX^\top)^2X$) et $c=0.2\sqrt{\max(n,m)}$ aligne l'échelle de mise à jour sur celle d'AdamW [10]. Les embeddings, normes et la tête restent en AdamW.

**Récapitulatif**

| Optimiseur | État par paramètre | Idée |
|---|---|---|
| SGD | 0 | pas fixe |
| Momentum / Nesterov | 1 ($v$) | inertie |
| Adagrad / RMSprop | 1 ($G$ / $s$) | échelle par coordonnée |
| Adam / AdamW | 2 ($m,v$) | inertie + échelle |
| Lion | 1 ($m$) | signe |
| Muon | 1 ($M$) | orthogonalisation |"""),
code(PRELUDE_IMPORTS),
tool(),
md(r"""## 2. Vérification : nos implémentations = `torch.optim`
Même trajectoire sur 15 pas d'un problème test (écart maximal des paramètres)."""),
code('''def trajectoire(make, steps=15):
    torch.manual_seed(0); p = torch.nn.Parameter(torch.randn(6, 4)); opt = make([p])
    for _ in range(steps):
        opt.zero_grad(); (p.pow(2).sum() + (p ** 3).mean()).backward(); opt.step()
    return p.detach()

cas = {"SGD": (lambda p: SGD(p, 0.05), lambda p: torch.optim.SGD(p, 0.05)),
       "Momentum": (lambda p: SGD(p, 0.05, momentum=0.9), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9)),
       "Nesterov": (lambda p: SGD(p, 0.05, momentum=0.9, nesterov=True), lambda p: torch.optim.SGD(p, 0.05, momentum=0.9, nesterov=True)),
       "Adagrad": (lambda p: Adagrad(p, 0.05), lambda p: torch.optim.Adagrad(p, 0.05)),
       "RMSprop": (lambda p: RMSprop(p, 0.01), lambda p: torch.optim.RMSprop(p, 0.01, alpha=0.99)),
       "Adam (L2)": (lambda p: Adam(p, 0.01, weight_decay=0.1), lambda p: torch.optim.Adam(p, 0.01, weight_decay=0.1)),
       "AdamW": (lambda p: AdamW(p, 0.01, weight_decay=0.1), lambda p: torch.optim.AdamW(p, 0.01, weight_decay=0.1))}
for nom, (a, b) in cas.items(): print(f"{nom:10s} écart max vs torch.optim : {(trajectoire(a) - trajectoire(b)).abs().max().item():.2e}")'''),
md(r"""## 3. Terrain de jeu : vallée étroite (quadratique mal conditionnée)
$f(x,y)=\tfrac12\,(x^2+50\,y^2)$, nombre de conditionnement $\kappa=50$, départ $(-4,1.2)$. Pour chaque optimiseur, le $\eta$ est **choisi par balayage** (meilleure valeur finale après 60 pas)."""),
code('''def f(z): return 0.5 * (z[0] ** 2 + 50 * z[1] ** 2)
def suivre(make, lr, steps=60):
    z = torch.nn.Parameter(torch.tensor([-4.0, 1.2])); opt = make([z], lr); traj = [z.detach().clone()]; pertes = [f(z).item()]
    for _ in range(steps):
        opt.zero_grad(); f(z).backward(); opt.step(); traj.append(z.detach().clone()); pertes.append(f(z).item())
    return torch.stack(traj).numpy(), np.array(pertes)

algos = {"SGD": lambda p, lr: SGD(p, lr), "Momentum": lambda p, lr: SGD(p, lr, momentum=0.9), "Nesterov": lambda p, lr: SGD(p, lr, momentum=0.9, nesterov=True),
         "RMSprop": lambda p, lr: RMSprop(p, lr), "Adam": lambda p, lr: Adam(p, lr), "Lion": lambda p, lr: Lion(p, lr)}
grille = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
meilleurs = {}
for nom, mk in algos.items():
    scores = {lr: suivre(mk, lr)[1][-1] for lr in grille}; scores = {k: (v if np.isfinite(v) else 1e18) for k, v in scores.items()}
    meilleurs[nom] = min(scores, key=scores.get)
print("η retenus :", meilleurs)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
X, Y = np.meshgrid(np.linspace(-5, 2, 200), np.linspace(-1.5, 1.5, 200)); ax[0].contour(X, Y, 0.5 * (X ** 2 + 50 * Y ** 2), levels=np.geomspace(0.05, 40, 14), cmap="Greys", alpha=0.6)
for nom, mk in algos.items():
    traj, pertes = suivre(mk, meilleurs[nom]); ax[0].plot(traj[:, 0], traj[:, 1], ".-", ms=3, lw=1, label=f"{nom} (η={meilleurs[nom]})"); ax[1].semilogy(np.maximum(pertes, 1e-12), label=nom)
ax[0].set_xlim(-5, 2); ax[0].set_ylim(-1.5, 1.5); ax[0].set_title("trajectoires (cadre fixe : une excursion peut sortir de la fenêtre)"); ax[0].legend(fontsize=7); ax[1].set_title("f(θ_t)"); ax[1].set_xlabel("pas"); ax[1].legend(fontsize=7); plt.tight_layout(); plt.show()'''),
md(r"""**Lecture (exécution fournie)** :
- **SGD** est limité par la direction raide : son pas ne peut pas dépasser $\\approx 2/\\lambda_{\\max}=2/50=0{,}04$ sans diverger ; il avance donc lentement le long de la direction plate ($x$) — la courbe $f(\\theta_t)$ décroît lentement.
- **Momentum** oscille (dépassements visibles) ; **Nesterov** est la plus régulière et atteint la plus petite valeur parmi les méthodes à pas fixe à la fin des 60 pas.
- **RMSprop** avec le $\\eta=1$ retenu fait un premier pas énorme (sa trajectoire sort du cadre) puis converge très vite. **Le critère « meilleure valeur finale » favorise ce réglage agressif** : c'est une propriété de la sélection sur une grille grossière, pas une supériorité de RMSprop.
- **Lion** oscille avec une amplitude constante (chaque coordonnée bouge de $\\pm\\eta$ quel que soit le gradient) : il faut décroître $\\eta$ (schedule) pour converger — visible sur la courbe.
- **Adam** ($\\eta=0{,}1$) descend de façon régulière mais lente. **Ce problème convexe 2D n'est pas représentatif des réseaux profonds** : il illustre la géométrie, pas le classement.

## 4. Weight decay : $L_2$ dans Adam vs découplé (AdamW)
**Mesure isolée de l'effet de la décroissance.** Un paramètre unique $\theta_0=1$ reçoit 200 pas d'un gradient de bruit $g_t=\sigma\,\xi_t$ ($\xi_t\sim\mathcal N(0,1)$, mêmes tirages avec et sans $\lambda$). L'effet du weight decay est $\Delta=\theta_{\lambda=0}-\theta_{\lambda}$ (même bruit ⇒ seule la décroissance diffère). On fait varier l'**échelle du gradient** $\sigma$.

Prédiction théorique : en Adam-$L_2$, le terme $\lambda\theta$ est divisé par $\sqrt{\hat v}\approx\sigma$ donc $\Delta\propto 1/\sigma$ (saturant quand le pas $\eta$ limite le déplacement) ; en AdamW, $\Delta\approx 1-(1-\eta\lambda)^{200}$ **indépendant de $\sigma$**.
"""),
code('''def effet_decroissance(make, sigma, steps=200, lr=0.01, wd=0.1):
    finals = []
    for lam in (0.0, wd):
        g = torch.Generator().manual_seed(0); p = torch.nn.Parameter(torch.ones(1)); opt = make([p], lr, lam)
        for _ in range(steps):
            opt.zero_grad(); p.grad = sigma * torch.randn(1, generator=g); opt.step()
        finals.append(p.item())
    return finals[0] - finals[1]

print(f"{'σ (échelle du gradient)':26}{'Adam + L2':>14}{'AdamW':>12}")
for sigma in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
    a = effet_decroissance(lambda p, lr, wd: Adam(p, lr, weight_decay=wd), sigma); w = effet_decroissance(lambda p, lr, wd: AdamW(p, lr, weight_decay=wd), sigma)
    print(f"{sigma:<26g}{a:>14.4f}{w:>12.4f}")
print(f"\\nAdamW théorique : 1-(1-ηλ)^200 = {1-(1-0.01*0.1)**200:.4f}")'''),
md(r"""**Lecture** : pour Adam-$L_2$, l'effet de la décroissance **dépend fortement de l'échelle du gradient** (fort pour les petits gradients, quasi nul pour les grands) ; pour AdamW il est **constant** et proche de la valeur théorique (0,19 mesuré pour 0,18 : l'écart vient du terme de bruit, lui aussi multiplié par le facteur de décroissance). Les deux coïncident quand $\\sigma\\approx1$, car alors $\\sqrt{\\hat v}\\approx1$ ; en dessous, Adam-$L_2$ décroît **beaucoup plus** (jusqu'à saturer), au-dessus **beaucoup moins**, avec la proportionnalité $1/\\sigma$ attendue ($0{,}186\\to0{,}020\\to0{,}002$ pour $\\sigma=1,10,100$). C'est la raison pour laquelle $\lambda$ est interprétable et transférable en AdamW, pas en Adam-$L_2$ [7].

## 5. Comparaison sur un modèle de langage, taux d'apprentissage réglé par optimiseur
Protocole : (1) **balayage** de $\eta$ (1 seed, 150 pas) pour chaque optimiseur ; (2) **entraînement final** au meilleur $\eta$ (2 seeds, 300 pas). Modèle : 2 couches, $d=64$, mêmes données. Écrêtage du gradient à 1.0 pour tous ; schedule warmup-cosinus.

*Biais de sélection* : le $\eta$ est choisi sur la perte de validation, puis on rapporte cette même perte : légèrement optimiste, de façon comparable pour tous. Pour un article : sélectionner sur un jeu de validation, rapporter sur un jeu de test distinct."""),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
GRILLES = {"sgd": [0.1, 0.3, 1.0, 3.0], "sgd_momentum": [0.1, 0.3, 1.0, 3.0], "nesterov": [0.1, 0.3, 1.0, 3.0],
           "rmsprop": [3e-4, 1e-3, 3e-3, 1e-2], "adagrad": [0.003, 0.01, 0.03, 0.1], "adam": [1e-3, 3e-3, 1e-2, 3e-2],
           "adamw": [1e-3, 3e-3, 1e-2, 3e-2], "lion": [3e-4, 1e-3, 3e-3, 1e-2], "muon": [1e-3, 3e-3, 1e-2]}
# Weight decay : 0.1 (découplé) pour AdamW, Lion et Muon ; 0 pour les autres. Adam est comparé avec λ=0 pour isoler l'ALGORITHME :
# avec un L2 de 0.1, Adam s'effondre pour la raison expliquée au §4 (voir §6, qui traite ce point à part).
WD = {o: (0.1 if o in ("adamw", "lion", "muon") else 0.0) for o in GRILLES}
balayage = {}
for opt, grille in GRILLES.items():
    balayage[opt] = {}
    for lr in grille:
        r = compare({opt: dict(model={}, train=dict(optimizer=opt, lr=lr))}, DATA, BASE_M, dict(steps=150, batch_size=16, warmup_steps=10, eval_interval=150, eval_iters=10, weight_decay=WD[opt]), seeds=(100,), verbose=False)
        v = r[opt]["mean"]; balayage[opt][lr] = v if np.isfinite(v) else 1e9
    print(f"{opt:13s}", {lr: round(v, 3) for lr, v in balayage[opt].items()})
meilleurs = {o: min(b, key=b.get) for o, b in balayage.items()}
print("\\nη retenus :", meilleurs)'''),
code('''plt.figure(figsize=(7, 3.8))
for o, b in balayage.items(): plt.plot(list(b), [min(v, 6) for v in b.values()], "o-", label=o)
plt.xscale("log"); plt.xlabel("taux d'apprentissage η"); plt.ylabel("perte de validation (plafonnée à 6)"); plt.title("Sensibilité au taux d'apprentissage"); plt.legend(fontsize=7, ncol=2); plt.show()'''),
code('''TRAIN_KW = dict(steps=300, batch_size=16, warmup_steps=20, eval_interval=100, eval_iters=10)
variantes = {o: dict(model={}, train=dict(optimizer=o, lr=lr, weight_decay=WD[o])) for o, lr in meilleurs.items()}
res = compare(variantes, DATA, BASE_M, TRAIN_KW, seeds=(0, 1))
plot_curves(res, title="perte de validation au meilleur η"); plt.show()'''),
md(r"""## 6. Adam ($L_2$) vs AdamW sur le modèle de langage, selon le weight decay
Même $\eta$ (celui d'AdamW), $\lambda\in\{0,\,0.1,\,1\}$ : le décrochage éventuel montre comment chaque formulation interagit avec l'échelle de $\lambda$."""),
code('''lr_w = meilleurs["adamw"]
var = {}
for wd in (0.0, 0.1, 1.0):
    var[f"Adam L2 λ={wd}"] = dict(model={}, train=dict(optimizer="adam", lr=lr_w, weight_decay=wd))
    var[f"AdamW  λ={wd}"] = dict(model={}, train=dict(optimizer="adamw", lr=lr_w, weight_decay=wd))
res_wd = compare(var, DATA, BASE_M, TRAIN_KW, seeds=(0, 1))'''),
md(r"""## Constats sur l'exécution fournie (taux d'apprentissage réglé, 2 seeds, 300 pas)
- **Adam, AdamW, Lion et Muon forment le groupe de tête** (val ≈ 0,350–0,360) ; leurs écarts (≲ 0,01) sont du même ordre que le bruit : **pas de classement concluant** entre eux. Muon a la meilleure moyenne mais une dispersion plus grande (±0,0065) et un débit plus faible (≈ 9,6 k tok/s contre ≈ 13 k) : l'itération de Newton–Schulz n'est pas optimisée ici.
- **Nesterov et RMSprop** (≈ 0,50), **Adagrad** (≈ 0,56) et **SGD avec momentum** (≈ 0,58) suivent, avec un écart-type élevé (jusqu'à 0,04) : à 2 seeds, leur ordre interne n'est pas établi. **SGD sans momentum** est très en retrait (1,58).
- **Sensibilité au taux d'apprentissage** (courbe §5) : les optimiseurs adaptatifs ont un optimum net mais étroit (un facteur 3 de part et d'autre dégrade beaucoup, sur un budget de 150 pas) ; tous les optimiseurs retenus sont à l'intérieur de la grille, pas sur son bord.
- **§6 — Adam ($L_2$) vs AdamW** : à $\lambda=0$ les deux sont **identiques** (0,3601, sanity check de l'implémentation). À $\lambda=0{,}1$, **Adam-$L_2$ s'effondre** (val ≈ 2,0, très instable) alors qu'AdamW reste au niveau de $\lambda=0$ (0,356) ; à $\lambda=1$ Adam-$L_2$ est encore pire (2,47) et AdamW se dégrade légèrement (0,368). C'est la conséquence pratique de la dépendance à l'échelle du gradient démontrée au §4.
- Le groupe de tête dépend du **budget** (300 pas) : les ordres peuvent changer à long terme.

## Bilan
| Optimiseur | Mémoire d'état | Réglage | Usage |
|---|---|---|---|
| SGD(+momentum) | 0–1× | $\eta$ très sensible | vision (ResNets) ; rarement pour les Transformers |
| Adagrad / RMSprop | 1× | — | historiques |
| **Adam / AdamW** | 2× | $\beta_2\in[0.95,0.999]$, $\varepsilon$, $\lambda$ | **standard LLM** ($\beta=(0.9,0.95)$, $\lambda=0.1$) |
| Lion | 1× | $\eta$ plus petit, $\lambda$ plus grand | alternative économe en mémoire [8] |
| Muon | 1× | $\eta$ plus grand, matrices 2D | alternative récente à grande échelle [9, 10] |

Compléments : Adam peut diverger sur certains problèmes convexes (contre-exemple et variante AMSGrad dans [11]) ; Adafactor [12] réduit l'état mémoire en factorisant le second moment.

**Ce que les figures et tableaux ci-dessus permettent de dire** : la sensibilité à $\eta$ (courbes de la §5), le classement à $\eta$ réglé, et l'effet de $\lambda$ sur Adam vs AdamW (§6). Ils ne permettent **pas** de conclure sur des modèles de plusieurs milliards de paramètres.

**Limites** : petit modèle et budget très court (300 pas, régime où l'optimisation est dominée par la transitoire initiale) ; 2 seeds ; grille de $\eta$ grossière (4 valeurs, pas de $\beta$/$\varepsilon$ réglés) ; Muon appliqué aux matrices cachées avec une échelle $\eta_{\text{Muon}}=10\,\eta$ non optimisée ; sélection sur le jeu de validation.

""" + refs_block("polyak1964", "nesterov1983", "sutskever2013", "duchi2011", "tieleman2012", "kingma2015", "loshchilov2019", "chen2023lion", "jordan2024", "liu2025", "reddi2018", "shazeer2018")),
]
