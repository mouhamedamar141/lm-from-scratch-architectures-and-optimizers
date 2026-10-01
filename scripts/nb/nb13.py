from .lib import md, code, tool, refs_block, PRELUDE_IMPORTS, DATA_CELL

FILE = "13-boucle-entrainement-complete.ipynb"
TITLE = "Boucle d'entraînement complète avec toutes les options"


def cells():
    return [
md(r"""# 13 · La boucle d'entraînement (toutes les options)

**À retenir** — une seule fonction `train(model, train_tokens, val_tokens, TrainConfig(...))` pilote l'entraînement ; **chaque option est un champ** de `TrainConfig`, donc chaque expérience est un dictionnaire reproductible.

| Famille | Options |
|---|---|
| Budget | `steps`, `batch_size`, `grad_accum`, `block_size` |
| Optimiseur | `optimizer` ∈ {sgd, sgd_momentum, nesterov, rmsprop, adagrad, adam, adamw, lion, muon}, `lr`, `weight_decay`, `betas`, `eps`, `momentum` |
| Schedule | `schedule` ∈ {constant, linear, cosine, warmup_cosine, warmup_linear, inv_sqrt, wsd}, `warmup_steps`, `min_lr_ratio`, `decay_frac` |
| Stabilité | `grad_clip`, `z_loss`, `label_smoothing` |
| Numérique/mémoire | `dtype` (float32/bfloat16/float16), `compile`, `grad_checkpointing`, `ema_decay` |
| Suivi | `eval_interval`, `eval_iters`, `log_interval`, `early_stop_patience`, callback |
| Persistance | `checkpoint`, `save_best_only`, `resume`, `seed` |

**Pas de l'algorithme** (par itération $s$) :
1. $\eta_s\leftarrow\eta_{\max}\cdot f_{\text{schedule}}(s)$ ; 2. pour chaque micro-lot : perte $\mathcal L+\mathcal L_{\text{aux}}\,(+\,z\text{-loss}, \text{lissage})$ divisée par `grad_accum`, rétropropagation ; 3. écrêtage ; 4. mise à jour de l'optimiseur ; 5. mise à jour de l'EMA $\bar\theta\leftarrow\delta\bar\theta+(1-\delta)\theta$ ; 6. évaluation périodique sur des **lots de validation fixes** (comparables entre modèles), checkpoint, arrêt anticipé.

**Conventions importantes**
- Les pertes rapportées (`train_loss`, `val_loss`) sont l'**entropie croisée seule** ; la perte auxiliaire MoE est optimisée mais suivie à part (`aux_loss`).
- Le weight decay ne s'applique **pas** aux normes, biais et embeddings.
- L'évaluation utilise l'EMA si `ema_decay` est défini."""),
code(PRELUDE_IMPORTS),
tool(),
code(DATA_CELL),
md("## 1. Code de la boucle (extrait commenté)\nLa fonction complète est dans la cellule technique ci-dessus (`train`). Cœur de l'itération :"),
code('''import inspect
src = inspect.getsource(train)
debut = src.index("for step in range(start, cfg.steps):"); fin = src.index("s = step + 1")
print(src[debut:fin])'''),
md("## 2. Exemple minimal"),
code('''cfg_m = ModelConfig(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, n_embd=64)
modele = LM(cfg_m)
h = train(modele, TRAIN, VAL, TrainConfig(steps=200, batch_size=16, lr=3e-3, eval_interval=50, eval_iters=10, log_interval=50, device="cpu"))
print("paramètres :", h["params"], "| meilleure val :", round(h["best_val_loss"], 4))'''),
md("""## 3. Toutes les options activées en même temps
AdamW + WSD + accumulation ×2 + bf16 + z-loss + label smoothing + EMA + arrêt anticipé + checkpoint, avec un **callback** qui alimente un graphique."""),
code('''import tempfile, os
tmp = tempfile.mkdtemp(); ck = os.path.join(tmp, "run.pt")
suivi = {"step": [], "val": []}
def callback(step, hist): suivi["step"].append(step); suivi["val"].append(hist["val_loss"][-1])

cfg_all = TrainConfig(steps=240, batch_size=8, grad_accum=2, optimizer="adamw", lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95),
                      schedule="wsd", warmup_steps=15, decay_frac=0.25, min_lr_ratio=0.1, grad_clip=1.0,
                      dtype="bfloat16", z_loss=1e-4, label_smoothing=0.05, ema_decay=0.99,
                      grad_checkpointing=True, eval_interval=40, eval_iters=10, early_stop_patience=4,
                      checkpoint=ck, seed=0, device="cpu")
m2 = LM(ModelConfig(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, n_embd=64, moe=True, n_experts=4, top_k=2, expert_hidden=88))
h2 = train(m2, TRAIN, VAL, cfg_all, callback=callback)
print("étapes évaluées :", h2["step"]); print("val :", [round(v, 3) for v in h2["val_loss"]]); print("perte auxiliaire MoE :", [round(v, 4) for v in h2["aux_loss"]])
print("paramètres totaux / actifs :", h2["params"], "/", h2["active_params"])'''),
md("""*La perte de validation ici est celle du modèle **EMA** ; le lissage de labels et la z-loss ne s'appliquent qu'à l'entraînement, pas à l'évaluation.*

## 4. Reprise exacte depuis un checkpoint
Test de la propriété : *entraîner 2×N pas d'un trait* ≡ *N pas, sauvegarde, reprise, N pas* (poids identiques). Le checkpoint contient poids, EMA, état de l'optimiseur, étape, meilleure perte et **état du générateur de lots**."""),
code('''def run(steps, ckpt=None, resume=False):
    torch.manual_seed(0); m = LM(ModelConfig(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=4, n_embd=64))
    kw = dict(batch_size=8, lr=3e-3, eval_interval=20, eval_iters=5, ema_decay=0.9, save_best_only=False, device="cpu", seed=3)
    return m, train(m, TRAIN, VAL, TrainConfig(steps=steps, checkpoint=ckpt, resume=resume, **kw))

a, ha = run(40)
ck2 = os.path.join(tmp, "r.pt"); run(20, ck2); c, hc = run(40, ck2, resume=True)
print("perte finale continue :", round(ha["final_val_loss"], 6), "| reprise :", round(hc["final_val_loss"], 6))
print("poids identiques      :", all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), c.parameters())))'''),
md("""## 5. Une boucle, beaucoup de comparaisons
Le même appel sert à comparer n'importe quel facteur. Exemple : trois combinaisons (optimiseur, schedule) au budget identique ; c'est exactement ce que font les notebooks 11, 12 et 16."""),
code('''BASE_M = dict(block_size=32, n_layer=2, n_head=4, n_embd=64)
res = compare({"adamw + cosinus": dict(model={}, train=dict(optimizer="adamw", schedule="warmup_cosine")),
               "adamw + wsd": dict(model={}, train=dict(optimizer="adamw", schedule="wsd")),
               "lion + cosinus": dict(model={}, train=dict(optimizer="lion", lr=1e-3, schedule="warmup_cosine"))},
              DATA, BASE_M, dict(steps=200, batch_size=16, lr=3e-3, warmup_steps=20, eval_interval=100, eval_iters=10), seeds=(0, 1))
plot_curves(res, title="perte de validation"); plt.show()'''),
md("""**Lecture** : WSD est meilleur que cosinus dans cette exécution à 200 pas (mais pas à 300 pas dans le notebook 12 : l'écart dépend du budget). Explication plausible, **non testée ici** : WSD conserve le pic de lr plus longtemps avant de décroître sur les derniers pas. Lion est beaucoup moins bon **parce que son lr (1e-3) n'est pas réglé** : ce n'est pas un verdict sur Lion (cf. notebook 11, où le lr est balayé pour chaque optimiseur).

## Bilan et limites
- **Reproductibilité** : graine unique pilote initialisation, lots et bruit ; lots de validation fixes ; reprise exacte vérifiée (test unitaire `test_resume_is_identical_to_continuous_training`).
- **`compile=True`** (torch.compile) et **fp16 + GradScaler** existent mais ne sont pas exercés ici (pas de compilateur/GPU dans l'environnement d'exécution des notebooks).
- **Parallélisme** (DDP/FSDP, parallélisme d'experts) : hors périmètre ; la boucle est mono-processus.
- Le débit `tok/s` de la boucle sur CPU n'est indicatif que pour comparer des variantes entre elles sur la même machine.

""" + refs_block("micikevicius2018", "chen2016", "loshchilov2019", "hu2024", "chowdhery2022", "pascanu2013")),
]
