# LLM From Scratch — Tokenizers, Transformers modernes, MoE, optimiseurs

Dépôt pédagogique **et expérimental** : chaque mécanisme d'un LLM moderne est réimplémenté explicitement avec les primitives PyTorch

> **Portée des résultats.** Le corpus par défaut est un corpus **synthétique** (grammaire probabiliste) et les modèles font ~10⁵
> paramètres. Les expériences valident les *implémentations* et illustrent le *protocole de comparaison* ; elles ne prédisent pas
> les performances de modèles de milliards de paramètres sur du texte naturel. 

## Ce que contient le dépôt

| Domaine | Variantes implémentées (`src/llm/`) |
|---|---|
| **Tokenizers** | BPE caractère, **BPE octets** (style GPT-2, sans `<unk>`), **WordPiece**, Unigram (approximation) |
| **Positions** | sinusoïdales, apprises, **RoPE** (+ interpolation linéaire, NTK-aware, YaRN), **ALiBi**, NoPE |
| **Attention** | **MHA, MQA, GQA, MLA** (DeepSeek), fenêtre glissante, **QK-Norm**, SDPA |
| **KV cache** | cache par couche (MHA/GQA/MQA/MLA), cache roulant, quantification simulée, mesure mémoire |
| **Normalisation** | LayerNorm, RMSNorm, Pre-LN / Post-LN |
| **FFN** | ReLU, GELU, SiLU, ReGLU, GeGLU, **SwiGLU** (largeur à paramètres égaux) |
| **MoE** | top-k softmax + perte **Switch**, **z-loss**, routeur **sans perte auxiliaire** (biais, DeepSeek-V3), **Expert Choice**, capacité, experts **fins** et **partagés**, couches denses initiales |
| **Optimiseurs** (écrits à la main) | **SGD, momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW**, Lion, Muon |
| **Entraînement** | schedules (constant, linéaire, cosinus, warmup-cosinus, inv-sqrt, **WSD**), écrêtage, accumulation, bf16/fp16, EMA, z-loss, label smoothing, gradient checkpointing, `torch.compile`, checkpoint + **reprise exacte**, arrêt anticipé |
| **Génération** | greedy, température, top-k, top-p, min-p, pénalité de répétition, avec/sans KV cache |

## Notebooks (16, autonomes, en français)

Détail dans [`docs/GUIDE_NOTEBOOKS.md`](docs/GUIDE_NOTEBOOKS.md). Ordre conseillé : 01–05 (fondations) → 06–09 (composants)
→ **10 (MoE)** → **11 (SGD/Adam/AdamW, formules LaTeX)** → 12–13 (**boucle d'entraînement à options**) → 14–16 (génération, architectures modernes, ablations).

Principes : peu de théorie (formules essentielles + « À retenir »), **une expérience contrôlée par notebook**, limites explicites,
références numérotées en fin de notebook. Aucun notebook n'importe de fichier local : la boîte à outils est intégrée (cellule repliée),
générée depuis `src/llm/` par `scripts/build_notebooks.py`, donc **identique au code testé**.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments,test]"
pytest                       # 71 tests : causalité, équivalence cache/sans cache, MoE, optimiseurs vs torch.optim, reprise exacte...
jupyter lab notebooks/
```

## Pipeline en ligne de commande

```bash
python scripts/train_tokenizer.py --type byte-bpe --vocab-size 300     # bpe | byte-bpe | wordpiece | unigram
python scripts/prepare_data.py                                          # option : --synthetic-chars 200000
python scripts/train.py --config configs/tiny.yaml
python scripts/evaluate.py --config configs/tiny.yaml
python scripts/generate.py --config configs/tiny.yaml --prompt "Le langage" --top-p 0.9 --temperature 0.8
python scripts/run_experiments.py --field attn_type --values mha,gqa,mqa,mla --seeds 0,1,2
```

Préréglages d'architecture (à fusionner avec `configs/tiny.yaml`) : `configs/architectures/{gpt2,llama,mistral,mixtral,deepseek}.yaml`.

## Utilisation en Python

```python
from llm.model import LM, ModelConfig
from llm.training import TrainConfig, train, compare

cfg = ModelConfig(vocab_size=31, block_size=32, n_layer=2, n_head=8, n_embd=64,
                  attn_type="gqa", n_kv_head=4, ffn_type="swiglu",
                  moe=True, n_experts=8, top_k=2, expert_hidden=88, router="sigmoid_bias")
model = LM(cfg)
hist = train(model, train_tokens, val_tokens,
             TrainConfig(steps=300, optimizer="adamw", schedule="wsd", grad_accum=2, dtype="bfloat16", ema_decay=0.99))
```

## Structure

```
src/llm/
  tokenizer/   bpe.py  byte_bpe.py  wordpiece.py  unigram.py
  model/       config.py  positions.py  norms.py  attention.py  cache.py  ffn.py  moe.py  block.py  model.py
  training/    optimizers.py  schedules.py  trainer.py  compare.py  checkpoint.py
  generation/  sampling.py  generate.py
  data/        preprocessing.py  synthetic.py
notebooks/     16 notebooks autonomes (générés par scripts/build_notebooks.py)
scripts/       CLI + build_notebooks.py + nb/ (sources des notebooks)
configs/       tiny.yaml  small.yaml  architectures/
docs/          REFERENCES.md  references.bib  GUIDE_NOTEBOOKS.md
tests/         71 tests
```

## Références

Bibliographie structurée : [`docs/REFERENCES.md`](docs/REFERENCES.md) et [`docs/references.bib`](docs/references.bib) (générées depuis
`scripts/nb/lib.py`). **Avant tout dépôt (thèse, article), vérifiez chaque entrée** (identifiants arXiv, versions, venues) : les liens ont été saisis
manuellement ; seuls quelques-uns ont été recontrôlés en ligne.

## Limites connues

- Boucle sur les experts MoE non optimisée (pas de noyaux groupés) : le débit MoE mesuré n'est pas représentatif d'un GPU.
- MLA implémenté en version « naïve » (K, V reconstruits depuis le latent) ; l'absorption des matrices d'inférence n'est pas reproduite.
- Quantification du cache KV **simulée** ; PagedAttention décrit mais non implémenté ; décodage spéculatif limité à la variante gloutonne.
- `torch.compile`, fp16 + `GradScaler` et l'entraînement multi-GPU (DDP/FSDP) ne sont pas exercés par les tests (CPU).
- Unigram est une approximation pédagogique (EM « dur » + élagage), pas SentencePiece.
