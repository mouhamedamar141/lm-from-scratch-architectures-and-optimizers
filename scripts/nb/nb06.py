from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "06-attention-mha-gqa-mla.ipynb"
TITLE = "Variantes d'attention : MHA, MQA, GQA, MLA, fenêtre glissante, QK-Norm"


def cells():
    return [
md("""# 06 · Variantes d'attention

**À retenir**
$$\\mathrm{Attn}(Q,K,V)=\\mathrm{softmax}\\!\\Big(\\tfrac{QK^\\top}{\\sqrt{d_h}}+M\\Big)V,\\qquad M_{ij}=\\begin{cases}0& j\\le i\\\\-\\infty&\\text{sinon}\\end{cases}$$

| Variante | Têtes $Q$ / $K,V$ | Cache KV par token et par couche | Idée |
|---|---|---|---|
| **MHA** [1] | $h$ / $h$ | $2\\,h\\,d_h$ | référence |
| **MQA** [2] | $h$ / $1$ | $2\\,d_h$ | 1 seule paire K,V partagée |
| **GQA** [3] | $h$ / $g$ | $2\\,g\\,d_h$ | $g$ groupes ($1<g<h$) : compromis MHA↔MQA |
| **MLA** [4] | $h$ / latent | $d_c+d_h^R$ | K,V reconstruits depuis un latent $c=W^{DKV}x\\in\\mathbb R^{d_c}$ |
| **Fenêtre glissante** [5, 6] | — | $\\le 2\\,g\\,d_h\\,W$ | chaque token ne voit que les $W$ derniers |
| **QK-Norm** [7, 8] | — | — | RMSNorm sur $q,k$ : évite l'explosion des logits d'attention |

Enjeu principal : à l'inférence, la **mémoire du cache KV** limite la longueur de contexte et le débit (voir notebook 07)."""),
code(PRELUDE_IMPORTS),
tool(),
md("""## 1. Attention causale « à la main » vérifiée
15 lignes suffisent ; on vérifie l'égalité avec `F.scaled_dot_product_attention` et avec la classe de la boîte à outils (MHA)."""),
code('''import torch.nn.functional as F
torch.manual_seed(0)
B, H, T, dh = 2, 4, 10, 8
q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))

def attention_causale(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))            # (B,H,T,T)
    masque = torch.ones(T, T, dtype=torch.bool).tril()
    scores = scores.masked_fill(~masque, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v

mine = attention_causale(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("écart max vs SDPA :", (mine - ref).abs().max().item())

# la classe de la boîte à outils en mode MHA doit donner un résultat de même forme et être causale
cfg = ModelConfig(vocab_size=10, n_embd=32, n_head=4, block_size=16)
att = GroupedQueryAttention(cfg).eval(); x = torch.randn(1, 12, 32); x2 = x.clone(); x2[:, 8:] += 1.0
print("causalité (le futur ne change pas le passé) :", torch.allclose(att(x)[:, :8], att(x2)[:, :8], atol=1e-6))'''),
md("""## 2. Visualiser les masques : causal, fenêtre glissante, ALiBi"""),
code('''pos = torch.arange(12)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for a, (titre, b) in zip(ax, [("causal", build_bias(pos, pos)), ("fenêtre W=4", build_bias(pos, pos, window=4)),
                              ("causal + ALiBi (tête 0)", build_bias(pos, pos, slopes=alibi_slopes(4))[0])]):
    a.imshow(torch.where(torch.isinf(b), torch.tensor(-9.0), b).numpy(), cmap="viridis"); a.set_title(titre); a.set_xlabel("clé j"); a.set_ylabel("requête i"); a.grid(False)
plt.tight_layout(); plt.show()'''),
md("""## 3. Coût mémoire et paramètres, par formule et par mesure
Configuration : $d=64,\\ h=8,\\ d_h=8$. MLA : $d_c=16$, $d_h^R=8$."""),
code('''base = dict(vocab_size=31, block_size=32, n_layer=2, n_head=8, n_embd=64)
configs = {"MHA": dict(attn_type="mha"), "GQA g=4": dict(attn_type="gqa", n_kv_head=4), "GQA g=2": dict(attn_type="gqa", n_kv_head=2),
           "MQA": dict(attn_type="mqa"), "MLA": dict(attn_type="mla", kv_lora_rank=16, rope_head_dim=8)}
print(f"{'variante':10}{'paramètres attention/couche':>28}{'cache (valeurs/token/couche)':>30}{'vs MHA':>9}")
ref = None
for name, kw in configs.items():
    cfg = ModelConfig(**base, **kw); att = make_attention(cfg)
    n = sum(p.numel() for p in att.parameters()); c = kv_cache_floats_per_token(cfg); ref = ref or c
    print(f"{name:10}{n:>28,}{c:>30}{c/ref:>9.2f}x")'''),
md("""Le cache est divisé par $h/g$ (GQA/MQA) ou par $2hd_h/(d_c+d_h^R)$ (MLA). Le compromis : MQA/GQA retirent des paramètres K,V ; MLA en *ajoute* (projections de compression/reconstruction) mais compresse le cache.

## 4. Expérience de qualité : mêmes données, seeds et budget
Question : à budget fixe, quelle perte de qualité coûte la réduction du cache ? (petit modèle : ne pas extrapoler aux grands)."""),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=8, n_embd=64)
BASE_T = dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10)
VARS = {"MHA": dict(attn_type="mha"), "GQA g=4": dict(attn_type="gqa", n_kv_head=4), "GQA g=2": dict(attn_type="gqa", n_kv_head=2),
        "MQA": dict(attn_type="mqa"), "MLA": dict(attn_type="mla", kv_lora_rank=16, rope_head_dim=8),
        "MHA + QK-Norm": dict(attn_type="mha", qk_norm=True), "MHA + fenêtre 8": dict(attn_type="mha", window=8)}
res = compare(VARS, DATA, BASE_M, BASE_T, seeds=(0, 1))
plot_curves(res, title="perte de validation"); plt.show()'''),
md("""## 5. Vitesse : implémentation explicite vs `scaled_dot_product_attention`
Sur CPU, avec de petites tailles, l'écart reflète surtout l'overhead ; les gains de FlashAttention [9, 10] concernent la **mémoire GPU** ($O(T)$ au lieu de $O(T^2)$ pour la matrice de scores) et les grands $T$."""),
code('''import time
def chrono(sdpa, T=256, reps=5):
    cfg = ModelConfig(vocab_size=31, block_size=T, n_layer=2, n_head=8, n_embd=64, sdpa=sdpa)
    m = LM(cfg).eval(); x = torch.randint(0, 31, (8, T))
    with torch.no_grad():
        m(x); t0 = time.perf_counter()
        for _ in range(reps): m(x)
    return (time.perf_counter() - t0) / reps * 1000
print(f"explicite : {chrono(False):.1f} ms | SDPA : {chrono(True):.1f} ms  (T=256, batch 8, CPU)")'''),
md("""**Constats sur l'exécution fournie (§4)**
- Réduire les têtes K,V **coûte** de la qualité : de l'ordre de +0,05 (GQA g=4) à +0,09 (GQA g=2) nats par rapport à MHA ; MQA est intermédiaire. Le gain de cache correspondant est $2\\times$ à $8\\times$ (§3).
- **MLA** est ici le moins bon *et* le plus instable d'un seed à l'autre (écart-type ~0,06) : avec $d_h=8$ le latent ($d_c=16$) n'a pas de marge de compression et le modèle est très petit. Ne pas conclure sur MLA depuis cette échelle.
- **QK-Norm** améliore légèrement (écart comparable à 2 écarts-types : à confirmer).
- **Fenêtre 8** est *meilleure* que MHA complète : le corpus synthétique est essentiellement local, donc restreindre la portée agit comme un biais inductif utile. C'est une propriété du corpus, pas de la fenêtre.

## Bilan
Règles générales issues de la littérature (à re-tester à votre échelle) :
- **GQA** est le choix standard des LLM ouverts récents (LLaMA-2 70B, LLaMA-3, Mistral, Qwen) [3, 11, 12].
- **MLA** obtient un cache bien plus petit que GQA tout en visant la qualité de MHA [4, 13], au prix d'une implémentation plus complexe (RoPE découplé, absorption des matrices en inférence).
- **Fenêtre glissante** borne la mémoire mais limite la portée directe (les couches empilées étendent le champ réceptif) [5, 6].
- **QK-Norm** stabilise l'entraînement à grande échelle [7, 8].

**Limites** : 2 seeds ; corpus synthétique aux dépendances courtes (la fenêtre 8 est peu pénalisée *parce que* les phrases sont courtes) ; à $d_h=8$ MLA n'a presque aucune marge de compression. Ces résultats démontrent le protocole, pas une hiérarchie universelle.

""" + refs_block("vaswani2017", "shazeer2019", "ainslie2023", "deepseekv2", "beltagy2020", "jiang2023", "henry2020", "dehghani2023", "dao2022", "dao2023", "touvron2023b", "dubey2024", "deepseekv3")),
]
