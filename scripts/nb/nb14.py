from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "14-generation-et-decodage.ipynb"
TITLE = "Génération : échantillonnage et décodage spéculatif"


def cells():
    return [
md(r"""# 14 · Génération : stratégies d'échantillonnage et décodage spéculatif

**À retenir** — à chaque pas, le modèle donne des logits $z\in\mathbb R^V$ ; le choix du token est une *décision de décodage* indépendante de l'entraînement.

| Méthode | Règle | Effet |
|---|---|---|
| Glouton | $\arg\max_i z_i$ | déterministe, tend à répéter |
| Température $T$ | $p_i=\mathrm{softmax}(z_i/T)$ | $T<1$ plus net, $T>1$ plus aléatoire |
| Top-$k$ | garde les $k$ plus probables | coupe la longue traîne |
| Top-$p$ (noyau) [1] | plus petit ensemble de masse cumulée $\ge p$ | taille adaptative |
| Min-$p$ [2] | garde $p_i\ge p_{\min}\cdot\max_j p_j$ | seuil relatif au meilleur token |
| Pénalité de répétition [3] | $z_i\leftarrow z_i/\theta$ si $z_i>0$, sinon $z_i\theta$, pour les tokens déjà générés | limite les boucles |

**Décodage spéculatif** [4] : un petit modèle *brouillon* propose $\gamma$ tokens, le grand modèle les **vérifie en une seule passe** ; on garde le plus long préfixe validé + 1 token du grand modèle. En glouton, la sortie est **identique** à celle du grand modèle seul.
Nombre moyen de tokens par appel du grand modèle, avec taux d'acceptation i.i.d. $\alpha$ : $\ \dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$."""),
code(PRELUDE_IMPORTS),
tool(),
md("## 1. Effet de chaque filtre sur une distribution jouet"),
code('''logits = torch.tensor([[4.0, 3.2, 2.5, 1.0, 0.5, 0.0, -0.5, -1.0]])
reglages = {"T=1": dict(), "T=0.5": dict(T=0.5), "T=1.5": dict(T=1.5), "top-k=3": dict(top_k=3), "top-p=0.8": dict(top_p=0.8), "min-p=0.2": dict(min_p=0.2)}
fig, ax = plt.subplots(1, len(reglages), figsize=(13, 2.6), sharey=True)
for a, (nom, kw) in zip(ax, reglages.items()):
    z = logits / kw.get("T", 1.0); p = filter_logits(z, kw.get("top_k"), kw.get("top_p"), kw.get("min_p")).softmax(-1)[0]
    a.bar(range(8), p.numpy()); a.set_title(nom, fontsize=9)
ax[0].set_ylabel("probabilité"); plt.tight_layout(); plt.show()'''),
md("""## 2. Un modèle entraîné et un **vérificateur de grammaire**
Le corpus synthétique étant engendré par une grammaire, on peut **mesurer objectivement** la qualité d'une phrase générée : est-elle grammaticalement valide (accords, parenthèses, motif si/alors) ? Cela donne une métrique de qualité sans juge humain, à comparer à la **diversité** (part de phrases distinctes)."""),
code(DATA_CELL),
code('''import re
# Modèle cible plus grand et entraîné plus longtemps (~5 min sur 1 cœur CPU) : un modèle sous-entraîné produit des non-mots et fausse toute évaluation de génération.
BASE_M = dict(block_size=64, n_layer=3, n_head=4, n_embd=96)
BASE_T = dict(steps=1500, batch_size=32, lr=3e-3, warmup_steps=50, eval_interval=750, eval_iters=10)
res = compare({"LM": dict()}, DATA, BASE_M, BASE_T, seeds=(0,), keep_models=True)
lm = res["LM"]["models"][0].eval()

def alt(xs): return "(?:" + "|".join(re.escape(x) for x in xs) + ")"
patrons = []
for nb in "sp":
    sujs = alt([s for s, n in SUJ if n == nb]); v = alt(VERBE[nb])
    patrons += [f"{sujs} {v} {alt(OBJ)} {alt(ADV)}\\\\.", f"{sujs} {re.escape(REL[nb])} {v} {alt(OBJ)}\\\\."]
    for nb2 in "sp":
        patrons.append(f"{sujs} {v} \\\\({alt([s for s, n in SUJ if n == nb2])} {alt(VERBE[nb2])}\\\\) {alt(ADV)}\\\\.")
patrons.append(f"si {alt(COND)}, alors {alt(CONS)}\\\\.")
VALIDE = re.compile("|".join(patrons))
verif = [VALIDE.fullmatch(s) is not None for s in TEXTE.split("\\n")[:2000]]
print(f"contrôle du vérificateur : {100*np.mean(verif):.1f}% des phrases du corpus d'entraînement sont valides (attendu : 100%)")'''),
code('''def evaluer(reglage, n_seq=24, n_tok=300, seed=0):
    g = torch.Generator().manual_seed(seed); prompt = torch.full((n_seq, 1), STOI["\\n"])
    out = generate(lm, prompt, n_tok, generator=g, **reglage)
    phrases = []
    for seq in out.tolist():
        txt = "".join(CHARS[i] for i in seq).split("\\n")[1:-1]      # on retire la première/dernière phrase (tronquées)
        phrases += [s for s in txt if s]
    ok = [VALIDE.fullmatch(s) is not None for s in phrases]
    valides = [s for s, o in zip(phrases, ok) if o]
    invalides = [s for s, o in zip(phrases, ok) if not o]
    return dict(n=len(phrases), valide=100 * np.mean(ok), distinct=100 * len(set(phrases)) / max(1, len(phrases)),
                distinct_valides=100 * len(set(valides)) / max(1, len(valides)), invalides=invalides)

REGLAGES = {"glouton": dict(greedy=True), "T=0.7": dict(temperature=0.7), "T=1.0": dict(temperature=1.0), "T=1.3": dict(temperature=1.3),
            "T=1.0 top-k=5": dict(temperature=1.0, top_k=5), "T=1.0 top-p=0.9": dict(temperature=1.0, top_p=0.9), "T=1.0 min-p=0.05": dict(temperature=1.0, min_p=0.05),
            "T=1.3 top-p=0.9": dict(temperature=1.3, top_p=0.9), "T=1.3 min-p=0.1": dict(temperature=1.3, min_p=0.1), "T=1.0 pénalité 1.3": dict(temperature=1.0, repetition_penalty=1.3)}
print(f"{'réglage':22}{'phrases':>9}{'valides %':>11}{'distinctes %':>14}{'distinctes/valides %':>22}")
resultats = {}
for nom, kw in REGLAGES.items():
    r = evaluer(kw); resultats[nom] = r; print(f"{nom:22}{r['n']:>9}{r['valide']:>11.1f}{r['distinct']:>14.1f}{r['distinct_valides']:>22.1f}")'''),
code('''print("Exemples de phrases INVALIDES (T=1.0) :"); [print("  ", repr(s)) for s in resultats["T=1.0"]["invalides"][:6]]
plt.figure(figsize=(5.5, 4))
for nom, r in resultats.items(): plt.scatter(r["distinct_valides"], r["valide"]); plt.annotate(nom, (r["distinct_valides"], r["valide"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.xlabel("diversité : % de phrases distinctes parmi les valides"); plt.ylabel("qualité : % de phrases valides"); plt.title("Compromis qualité / diversité"); plt.show()'''),
md("""**Lecture (exécution fournie)** : même le glouton ne produit qu'environ la moitié de phrases valides : le modèle (335 k paramètres, 1 500 pas) **n'a pas totalement appris la grammaire** ; les exemples de phrases invalides ci-dessus montrent le type d'erreur. La validité **baisse** quand la température augmente ($T=1{,}3$ : ≈ 35 %) ; top-$p$/min-$p$/top-$k$ limitent peu cette baisse ici (écarts de quelques points, avec ~100 phrases par réglage : erreur d'échantillonnage de l'ordre de ±5 points, donc **non concluants**). Le glouton a une diversité minimale (peu de phrases distinctes). La colonne *distinctes* seule est trompeuse (les phrases invalides sont presque toutes uniques) ; d'où la colonne *distinctes parmi les valides*.

## 3. Décodage spéculatif : correction et gain
On entraîne un petit modèle brouillon (1 couche, $d=48$) et on vérifie que la sortie du décodage spéculatif glouton **égale** celle du modèle cible ; on mesure le taux d'acceptation et le nombre de tokens par appel de la cible."""),
code('''resb = compare({"brouillon": dict(n_layer=1, n_embd=48)}, DATA, BASE_M, BASE_T, seeds=(0,), keep_models=True, verbose=False)
draft = resb["brouillon"]["models"][0].eval()
print("paramètres cible / brouillon :", lm.num_parameters(), "/", draft.num_parameters(), "| val brouillon :", round(resb["brouillon"]["mean"], 3))

@torch.no_grad()
def spec_greedy(target, draft, prompt, n_new, gamma):
    cur = prompt.clone(); calls = 0; acc = 0; prop = 0
    while cur.size(1) - prompt.size(1) < n_new:
        d = cur.clone(); props = []
        for _ in range(gamma):
            nxt = draft(d).logits[:, -1].argmax(-1, keepdim=True); props.append(nxt); d = torch.cat([d, nxt], 1)
        props = torch.cat(props, 1)
        tgt = target(d).logits[:, cur.size(1) - 1:].argmax(-1); calls += 1     # UNE passe de la cible pour gamma+1 positions
        n_ok = 0
        while n_ok < gamma and props[0, n_ok] == tgt[0, n_ok]: n_ok += 1
        acc += n_ok; prop += gamma
        cur = torch.cat([cur, props[:, :n_ok], tgt[:, n_ok:n_ok + 1]], 1)
    return cur[:, :prompt.size(1) + n_new], calls, acc / prop

prompt = torch.tensor([[STOI["\\n"]]]); N_NEW = 120
# Référence à CONTEXTE COMPLET : avec use_cache=False, generate() rogne le contexte aux block_size derniers tokens, ce qui divergerait au-delà de 64 tokens.
ref = generate(lm, prompt, N_NEW, greedy=True, use_cache=True)
print(f"{'γ':>3}{'sortie identique':>34}{'α mesuré':>10}{'tokens/appel cible':>20}{'théorie (α iid)':>17}")
for gamma in (1, 2, 4, 8):
    out, calls, alpha = spec_greedy(lm, draft, prompt, N_NEW, gamma)
    theo = (1 - alpha ** (gamma + 1)) / (1 - alpha) if alpha < 1 else gamma + 1
    commun = int((out[0] != ref[0]).nonzero()[0]) if not torch.equal(out, ref) else N_NEW + 1
    print(f"{gamma:>3}{str(torch.equal(out, ref)) + ' (préfixe commun ' + str(commun) + ')':>34}{alpha:>10.2f}{N_NEW / calls:>20.2f}{theo:>17.2f}")'''),
md("""**Lecture** : la colonne « sortie identique » valide l'implémentation *si* elle vaut `True` (le décodage spéculatif glouton ne change pas le texte). Si elle vaut `False`, le préfixe commun indique où l'écart apparaît ; il peut venir d'une différence numérique entre passes de longueurs différentes qui bascule un argmax quasi-égal (le modèle n'est pas parfaitement entraîné). Le nombre de tokens par appel de la cible dépasse 1 tant que le brouillon est souvent d'accord ; le gain réel en temps dépend du rapport de coût brouillon/cible (non mesuré : la vérification est implémentée sans cache KV). **Dans l'exécution fournie, la sortie est identique pour tous les γ** (validation de l'implémentation) et le nombre de tokens par appel mesuré **dépasse** la formule i.i.d. dès γ=4 (3,64 vs 2,57 ; 5,0 vs 2,0 pour γ=8) : l'acceptation est fortement *corrélée* entre positions (quand le brouillon est dans une phrase bien formée, il a presque toujours raison), ce que la formule i.i.d. ignore.

## Bilan
| Objectif | Réglage typique |
|---|---|
| Sortie factuelle/déterministe | glouton ou $T\\le0.3$ |
| Génération créative | $T\\approx0.8$–$1$ + top-$p\\approx0.9$ ou min-$p\\approx0.05$–$0.1$ |
| Éviter les boucles | pénalité de répétition modérée (1.1–1.3), à utiliser avec prudence |
| Latence | décodage spéculatif [4], cache KV (notebook 07), prédiction multi-tokens [5] |

**Limites** : corpus synthétique à grammaire *finie* (validité binaire, diversité bornée) ; un seul modèle et un seul jeu de graines de génération ; le décodage spéculatif *échantillonné* (avec rejet probabiliste) n'est pas implémenté, seulement la variante gloutonne.

""" + refs_block("holtzman2020", "nguyen2024", "keskar2019", "leviathan2023", "gloeckle2024")),
]
