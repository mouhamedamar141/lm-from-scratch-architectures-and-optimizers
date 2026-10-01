from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "07-kv-cache.ipynb"
TITLE = "KV cache : principe, mémoire, vitesse, fenêtre, quantification"


def cells():
    return [
md("""# 07 · KV cache

**À retenir**
- En génération, chaque nouveau token n'a besoin que de **sa** requête $q_t$ ; ses clés/valeurs passées $K_{<t},V_{<t}$ peuvent être **mémorisées** au lieu d'être recalculées.
- Sans cache : coût par pas $O(t)$ passes de projection $\\Rightarrow$ total $O(T^2)$ en projections. Avec cache : $O(1)$ projections par pas (l'attention reste $O(t)$).
- Prix : mémoire. Pour un lot $B$, $T$ tokens :
$$\\text{Mém}_{\\text{KV}}=B\\cdot T\\cdot n_{\\text{couches}}\\cdot \\underbrace{2\\,n_{kv}\\,d_h}_{\\text{valeurs/token/couche}}\\cdot \\text{octets}$$
- Leviers : **GQA/MQA** (moins de têtes KV), **MLA** (latent), **fenêtre** (borne $T$), **quantification** (moins d'octets), **paging** (moins de fragmentation) [1–5].
- **Important** : RoPE est appliqué à $k$ *avant* mise en cache (sa rotation dépend de sa position absolue, fixe)."""),
code(PRELUDE_IMPORTS),
tool(),
md("""## 1. Correction : le cache ne change pas le résultat
Préremplissage du prompt (12 tokens) puis un token à la fois, comparé au calcul complet. Écart attendu : erreurs d'arrondi flottant uniquement."""),
code('''torch.manual_seed(0)
VARIANTES = {"MHA": dict(), "GQA": dict(attn_type="gqa", n_kv_head=2), "MQA": dict(attn_type="mqa"), "MLA": dict(attn_type="mla"),
             "fenêtre 6": dict(window=6), "ALiBi": dict(position_mode="alibi"), "MoE": dict(moe=True, n_experts=4, top_k=2)}
print(f"{'variante':10}{'écart max logits (cache vs complet)':>40}")
for nom, kw in VARIANTES.items():
    m = LM(ModelConfig(vocab_size=50, block_size=64, n_layer=2, n_head=4, n_embd=32, **kw)).eval()
    x = torch.randint(0, 50, (2, 24)); full = m(x).logits; cache = m.new_cache()
    inc = torch.cat([m(x[:, :12], cache=cache).logits] + [m(x[:, i:i+1], cache=cache).logits for i in range(12, 24)], 1)
    print(f"{nom:10}{(full - inc).abs().max().item():>40.2e}")'''),
md("""## 2. Vitesse : temps par token généré, avec et sans cache
On mesure le temps du $t$-ème pas de génération (greedy)."""),
code('''def temps_par_pas(use_cache, T=200, prompt=8):
    torch.manual_seed(0)
    m = LM(ModelConfig(vocab_size=50, block_size=T + 16, n_layer=4, n_head=4, n_embd=128)).eval()
    cur = torch.randint(0, 50, (1, prompt)); cache = m.new_cache() if use_cache else None; temps = []
    with torch.no_grad():
        for _ in range(T):
            t0 = time.perf_counter()
            inp = cur if (not use_cache or cache.n_seen == 0) else cur[:, -1:]
            nxt = m(inp, cache=cache).logits[:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], 1); temps.append((time.perf_counter() - t0) * 1000)
    return np.array(temps)

sans, avec = temps_par_pas(False), temps_par_pas(True)
plt.figure(figsize=(6, 3.4)); plt.plot(sans, label="sans cache"); plt.plot(avec, label="avec cache")
plt.xlabel("position générée t"); plt.ylabel("ms / token"); plt.legend(); plt.title("Coût par pas de génération"); plt.show()
print(f"total 200 tokens : sans cache {sans.sum()/1000:.2f}s, avec cache {avec.sum()/1000:.2f}s (x{sans.sum()/avec.sum():.1f})")'''),
md("""**Lecture** : sans cache, le coût par pas *croît* avec $t$ (on retraite tout le préfixe) ; avec cache il reste presque plat. Sur CPU et petit modèle l'accélération est modeste ; elle augmente avec la profondeur, la largeur et $T$.

## 3. Mémoire : formule = mesure
On compare le cache mesuré (`cache.nbytes()`) à la formule, puis on l'applique à des configurations publiées."""),
code('''T = 100
print(f"{'variante':10}{'valeurs/token/couche':>22}{'mesuré (octets)':>18}{'formule (octets)':>18}")
for nom in ["MHA", "GQA", "MQA", "MLA"]:
    cfg = ModelConfig(vocab_size=50, block_size=128, n_layer=2, n_head=4, n_embd=32, **VARIANTES[nom])
    m = LM(cfg).eval(); c = m.new_cache(); m(torch.randint(0, 50, (1, T)), cache=c)
    f = kv_cache_floats_per_token(cfg)
    print(f"{nom:10}{f:>22}{c.nbytes():>18,}{T*cfg.n_layer*f*4:>18,}")

def kv_gib(layers, valeurs, T, B=1, octets=2): return B * T * layers * valeurs * octets / 2**30
print("\\nConfigurations publiées (formules ; bf16 ; contexte 32 768, lot 1) :")
pub = {"Llama-2-7B (MHA, 32 couches, 32 têtes, d_h=128)": (32, 2*32*128),
       "Llama-3-8B (GQA, 32 couches, 8 têtes KV, d_h=128)": (32, 2*8*128),
       "Type DeepSeek-V2 si MHA (60 couches, 128 têtes, d_h=128)": (60, 2*128*128),
       "DeepSeek-V2 avec MLA (60 couches, d_c=512, d_R=64)": (60, 512 + 64)}
for k, (L_, v_) in pub.items(): print(f"  {k:60s} {kv_gib(L_, v_, 32768):8.2f} GiB")'''),
md("""*Les dimensions publiées sont reprises de [4, 6, 7] ; recopiez-les depuis les configurations officielles avant de les citer dans un manuscrit.*

## 4. Cache roulant (fenêtre glissante)
Avec fenêtre $W$, on ne garde que les $W$ derniers tokens : mémoire **bornée**, indépendante de $T$."""),
code('''cfg = ModelConfig(vocab_size=50, block_size=512, n_layer=2, n_head=4, n_embd=32, window=16)
m = LM(cfg).eval(); c = m.new_cache(); tailles = []
with torch.no_grad():
    m(torch.randint(0, 50, (1, 8)), cache=c)
    for _ in range(120):
        m(torch.randint(0, 50, (1, 1)), cache=c); tailles.append(c.nbytes())
print(f"fenêtre W={cfg.window} ; taille du cache (octets) après 20, 60 et 120 tokens générés : {tailles[19]:,} / {tailles[59]:,} / {tailles[119]:,}")
print("bornée :", tailles[19] == tailles[59] == tailles[119])'''),
md("""**Attention** : une fenêtre pure perd le contexte ancien ; StreamingLLM [8] observe que conserver aussi les premiers tokens (« attention sinks ») stabilise la génération infiniment longue.

## 5. Quantification du cache (int8 / int4 par token)
Quantification symétrique par vecteur : $\\hat x=\\text{round}(x/s)\\,s$, $s=\\max|x|/(2^{b-1}-1)$. On mesure l'impact sur la **perte** en décodage token par token (teacher forcing) d'un modèle entraîné."""),
code(DATA_CELL),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
res = compare({"MHA": dict()}, DATA, BASE_M, dict(steps=250, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=250, eval_iters=10), seeds=(0,), keep_models=True, verbose=False)
modele = res["MHA"]["models"][0].eval()

def fake_quant(x, bits):
    if bits >= 32: return x
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(x / s).clamp(-qmax, qmax) * s

class QuantCache(KVCache):
    def __init__(self, *a, bits=32, **k): super().__init__(*a, **k); self.bits = bits
    def update(self, layer, dim, **t): return super().update(layer, dim, **{n: fake_quant(v, self.bits) for n, v in t.items()})

@torch.no_grad()
def nll_decodage(model, bits, n_seq=24, T=31):
    g = torch.Generator().manual_seed(0); x, y = get_batch(VAL, n_seq, T, "cpu", g); cache = QuantCache(model.cfg.n_layer, bits=bits)
    pertes = [torch.nn.functional.cross_entropy(model(x[:, t:t+1], cache=cache).logits[:, -1], y[:, t]).item() for t in range(T)]
    return float(np.mean(pertes))

base = nll_decodage(modele, 32)
print(f"{'cache':8}{'perte':>9}{'Δ':>10}{'mémoire relative':>18}")
for bits in (32, 8, 4, 3, 2):
    l = nll_decodage(modele, bits); print(f"{'int'+str(bits) if bits<32 else 'fp32':8}{l:>9.4f}{l-base:>+10.4f}{bits/32:>18.2f}")'''),
md("""**Lecture** : dans l'exécution fournie, int8 est indiscernable de fp32, int4 coûte quelques millièmes, int3 quelques centièmes et int2 s'effondre. Les méthodes publiées (KIVI [9], KVQuant [10]) quantifient les clés *par canal* et les valeurs *par token* pour tenir 2 bits ; ici, quantification naïve par vecteur (simulée : la mémoire est théorique).

## 6. PagedAttention (concept)
Réserver un tampon contigu de taille maximale par requête gaspille de la mémoire (fragmentation, sur-réservation). PagedAttention [5] découpe le cache en **pages** de taille fixe, allouées à la demande via une table de pages (comme la mémoire virtuelle), et permet le **partage de préfixes** entre requêtes. C'est une optimisation de *serveur d'inférence* (vLLM), pas de modèle : non reproduite ici.

## Bilan
| Levier | Gain mémoire | Coût / risque |
|---|---|---|
| GQA/MQA | $h/g$ | qualité (à mesurer, cf. notebook 06) |
| MLA | $\\approx 2hd_h/(d_c+d_h^R)$ | complexité, paramètres de compression |
| Fenêtre | borne $T\\to W$ | perte de portée directe |
| Quantification | $32/b$ | erreur d'arrondi (mesurée §5) |
| Paging | moins de gaspillage | ingénierie serveur |

**Limites** : mesures de vitesse sur CPU et petit modèle (variables selon la machine) ; quantification simulée ; les configurations publiées de la §3 sont à revérifier.

""" + refs_block("pope2022", "shazeer2019", "ainslie2023", "deepseekv2", "kwon2023", "touvron2023b", "dubey2024", "xiao2024", "liu2024kivi", "hooper2024")),
]
