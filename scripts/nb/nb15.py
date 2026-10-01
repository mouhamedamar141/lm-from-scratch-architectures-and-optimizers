from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "15-architectures-llm-modernes.ipynb"
TITLE = "Panorama des architectures de LLM modernes et comparaison à échelle réduite"


def cells():
    return [
md(r"""# 15 · Architectures de LLM modernes

**À retenir** — les LLM ouverts récents convergent vers une « recette » : décodeur, **Pre-norm RMSNorm**, **RoPE**, **FFN SwiGLU**, sans biais, **GQA** ; les différences se jouent sur (i) la réduction du cache KV (GQA → MLA), (ii) la parcimonie (MoE), (iii) l'attention locale/globale et le contexte long, (iv) la stabilité (QK-Norm).

## 1. Panorama (choix documentés dans les rapports techniques)

| Modèle | Position | Norme | FFN | Attention | Parcimonie / particularités |
|---|---|---|---|---|---|
| **GPT-2** [1] | positions apprises | LayerNorm (Pre-LN), biais | GELU | MHA | vocabulaire BPE octets, poids liés |
| **LLaMA** [2] | RoPE | RMSNorm (Pre) | SwiGLU | MHA | sans biais |
| **Llama 2** [3] | RoPE | RMSNorm | SwiGLU | MHA ; **GQA** pour 34B/70B | contexte 4 096 |
| **Llama 3** [4] | RoPE (base 500 000) | RMSNorm | SwiGLU | **GQA** (8 têtes KV) | vocabulaire 128 K, contexte étendu à 128 K |
| **Mistral 7B** [5] | RoPE | RMSNorm | SwiGLU | GQA + **fenêtre glissante** (cache roulant) | — |
| **Mixtral 8x7B** [6] | RoPE | RMSNorm | SwiGLU (experts) | GQA | **MoE** 8 experts, top-2 |
| **Qwen2** [7] | RoPE | RMSNorm | SwiGLU | GQA (biais sur QKV) | variantes MoE (experts fins + partagés) |
| **Qwen3** [8] | RoPE | RMSNorm | SwiGLU | GQA + **QK-Norm** | variantes MoE ; modes « raisonnement » |
| **Gemma 2** [9] | RoPE | RMSNorm (pré et post) | GeGLU | GQA, couches locales/globales alternées | *soft-capping* des logits |
| **Gemma 3** [10] | RoPE | RMSNorm | GeGLU | GQA, ratio local:global élevé, **QK-Norm** | contexte long |
| **DeepSeek-V2** [11] | RoPE découplé + YaRN | RMSNorm | SwiGLU | **MLA** | **DeepSeekMoE** (experts fins + partagés) |
| **DeepSeek-V3** [12] | RoPE découplé | RMSNorm | SwiGLU | **MLA** | DeepSeekMoE, **équilibrage sans perte auxiliaire**, prédiction multi-tokens |

*Ce tableau est une synthèse de lecture ; **vérifiez chaque cellule dans la source citée** avant de la reprendre dans un manuscrit (les détails varient selon la taille du modèle et la version).*

Évolutions récentes du chemin résiduel : Hyper-Connections [13] et mHC [14]."""),
code(PRELUDE_IMPORTS),
tool(),
md(r"""## 2. Ces « styles » à échelle réduite
On **imite les choix architecturaux** (pas les tailles) avec la même configuration pour tous : $d=64$, 8 têtes, 3 couches. « Style » = mêmes mécanismes, pas une reproduction du modèle. Comparaison à **calcul FFN actif comparable** pour les MoE."""),
code('''h = ffn_hidden(ModelConfig(vocab_size=31, block_size=32, n_layer=3, n_head=8, n_embd=64, ffn_type="swiglu"))
BASE_M = dict(block_size=32, n_layer=3, n_head=8, n_embd=64)
STYLES = {
 "GPT-2":         dict(position_mode="learned", norm_type="layernorm", bias=True, ffn_type="gelu", attn_type="mha"),
 "LLaMA-1":       dict(position_mode="rope", norm_type="rmsnorm", ffn_type="swiglu", attn_type="mha"),
 "Llama-3":       dict(position_mode="rope", rope_theta=500_000.0, norm_type="rmsnorm", ffn_type="swiglu", attn_type="gqa", n_kv_head=2),
 "Mistral":       dict(position_mode="rope", norm_type="rmsnorm", ffn_type="swiglu", attn_type="gqa", n_kv_head=2, window=16),
 "Qwen3 (dense)": dict(position_mode="rope", norm_type="rmsnorm", ffn_type="swiglu", attn_type="gqa", n_kv_head=2, qk_norm=True),
 "Mixtral":       dict(position_mode="rope", norm_type="rmsnorm", ffn_type="swiglu", attn_type="gqa", n_kv_head=2, moe=True, n_experts=8, top_k=2, expert_hidden=h // 2, router="softmax"),
 "DeepSeek-V3":   dict(position_mode="rope", norm_type="rmsnorm", ffn_type="swiglu", attn_type="mla", kv_lora_rank=16, rope_head_dim=8, q_lora_rank=16,
                       moe=True, n_experts=16, top_k=3, n_shared_experts=1, expert_hidden=h // 4, router="sigmoid_bias", n_dense_layers=1),
}
print(f"largeur FFN dense h={h} ; experts : Mixtral h/2 (top-2), DeepSeek h/4 (top-3 + 1 partagé) => FFN actif ≈ h")'''),
md("### Coûts structurels (sans entraînement) : paramètres, cache KV"),
code('''print(f"{'style':15}{'params':>10}{'actifs':>10}{'cache KV (val./token/couche)':>31}{'cache 4096 tok, 24 couches, fp16 (Mo)':>40}")
for nom, kw in STYLES.items():
    cfg = ModelConfig(vocab_size=31, **BASE_M, **kw); m = LM(cfg); kv = kv_cache_floats_per_token(cfg)
    print(f"{nom:15}{m.num_parameters():>10,}{m.num_parameters(True):>10,}{kv:>31}{4096*24*kv*2/2**20:>40.1f}")'''),
md("### Qualité : entraînement comparatif (2 seeds)"),
code(DATA_CELL),
code('''BASE_T = dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=125, eval_iters=10)
res = compare(STYLES, DATA, BASE_M, BASE_T, seeds=(0, 1))
plot_curves(res, title="perte de validation"); plt.show()'''),
code('''fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
for nom, r in res.items():
    kv = kv_cache_floats_per_token(ModelConfig(vocab_size=31, **BASE_M, **STYLES[nom]))
    ax[0].errorbar(kv, r["mean"], yerr=r["std"], fmt="o", capsize=3); ax[0].annotate(nom, (kv, r["mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].errorbar(r["active"], r["mean"], yerr=r["std"], fmt="o", capsize=3); ax[1].annotate(nom, (r["active"], r["mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[0].set_xlabel("cache KV : valeurs / token / couche"); ax[0].set_ylabel("perte de validation"); ax[0].set_title("qualité vs mémoire d'inférence")
ax[1].set_xlabel("paramètres actifs"); ax[1].set_title("qualité vs calcul actif"); plt.tight_layout(); plt.show()'''),
md(r"""## 3. Constats sur l'exécution fournie, puis comment lire
Constats (2 seeds ; écarts ≲ 0,03 non concluants) :
- **GPT-2-style est très en retrait** (≈ 0,90 contre 0,39–0,54 pour les autres) : positions apprises + LayerNorm + GELU, sur un budget de 250 pas. Le notebook 05 montre déjà que les positions apprises convergent lentement ; ce n'est pas un jugement sur GPT-2 à pleine échelle.
- **Qwen3-style (GQA + QK-Norm) a la meilleure moyenne** (0,386), cohérent avec l'effet positif de QK-Norm mesuré aux notebooks 06 et 08 ; **LLaMA-1-style (MHA)** est proche (0,407) et **Mistral-style** (GQA + fenêtre 16) aussi (0,415).
- **Llama-3-style est moins bon que LLaMA-1-style** (0,486 vs 0,407). Deux changements coïncident (GQA 4:1 et base RoPE 500 000) ; **cette expérience ne permet pas de les séparer** (le notebook 06 chiffre déjà un coût de GQA). Une base RoPE très grande est conçue pour des contextes très longs, pas pour des séquences de 32 tokens.
- **Les deux styles MoE ne battent pas le dense** (Mixtral 0,467 ± 0,036 ; DeepSeek-V3-style 0,538 ± 0,023), avec 3× plus de paramètres totaux : cohérent avec le notebook 10 (pas de gain à cette échelle). DeepSeek-V3-style cumule MLA (latent $d_c=16$, peu de marge à $d_h=8$), 16 experts très petits et une couche dense ; on ne peut pas attribuer la perte à un seul de ces choix.
- **Cache KV** : de 128 valeurs/token/couche (MHA) à 32 (GQA) et 24 (MLA) : la réduction est structurelle et exacte, indépendante de la qualité.

## 4. Comment lire ces résultats
- **Deux frontières** : la qualité en fonction du cache KV (coût d'inférence en mémoire) et en fonction des paramètres actifs (coût de calcul). Une architecture est intéressante si elle est sur la frontière de Pareto de l'un ou l'autre.
- Les **MoE** ont plus de paramètres *totaux* (mémoire de poids) pour un calcul actif comparable ; ce coût n'apparaît pas sur les graphiques ci-dessus.
- **Ce n'est pas un classement des modèles réels** : le classement de modèles à l'échelle de 100 k paramètres sur un corpus synthétique n'a aucune valeur prédictive pour des modèles de milliards de paramètres. La valeur de ce notebook est le **protocole** (mêmes données, seeds, budget ; mesures structurelles + qualité) et l'implémentation vérifiée de chaque mécanisme (tests unitaires : causalité, équivalence cache/sans cache, équilibrage).

## 5. Guide de sélection (littérature)
| Contrainte dominante | Choix usuel |
|---|---|
| Mémoire d'inférence / long contexte | GQA → **MLA** ; fenêtre locale/globale ; quantification du cache |
| Qualité par FLOP d'entraînement | **MoE** fin + experts partagés, équilibrage sans perte auxiliaire |
| Stabilité à grande échelle | Pre-norm RMSNorm, **QK-Norm**, z-loss, écrêtage, WSD |
| Simplicité / reproductibilité | LLaMA-style dense + GQA |

**Limites** : 7 architectures, 2 seeds, modèles de ~10⁵ paramètres, 250 pas, corpus synthétique local (les mécanismes de contexte long, de MLA et de MoE n'y sont pas sollicités comme dans un vrai corpus). À refaire sur un corpus réel et ≥ 5 seeds pour toute affirmation empirique dans un manuscrit.

""" + refs_block("radford2019", "touvron2023", "touvron2023b", "dubey2024", "jiang2023", "jiang2024", "qwen2024", "qwen3", "gemma2", "gemma3", "deepseekv2", "deepseekv3", "zhu2024hc", "xie2025mhc")),
]
