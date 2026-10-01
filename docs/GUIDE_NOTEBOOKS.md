# Guide des notebooks

| # | Notebook | Contenu | Comparaisons |
|---|---|---|---|
| 01 | Tokenisation : bases | caractère / mot / sous-mot, coût du vocabulaire | compression, fertilité, OOV |
| 02 | BPE from scratch | algorithme, BPE octets, pré-tokenisation | compression vs nombre de fusions |
| 03 | Variantes de tokenizers | BPE, BPE octets, WordPiece, Unigram | compression, robustesse, vitesse |
| 04 | Embeddings | lookup, gradient creux, co-occurrences, weight tying | tying vs sans tying |
| 05 | Encodages de position | sinusoïdal, appris, RoPE, ALiBi, NoPE, PI/NTK/YaRN | extrapolation de longueur |
| 06 | Attention | MHA, MQA, GQA, MLA, fenêtre, QK-Norm, SDPA | qualité vs cache KV |
| 07 | **KV cache** | principe, exactitude, vitesse, mémoire, fenêtre, quantification, paging | avec/sans cache, int8/4/3/2 |
| 08 | Normalisation et résidus | LayerNorm, RMSNorm, Pre/Post-LN, QK-Norm | stabilité (2 régimes de lr) |
| 09 | FFN et activations | ReLU, GELU, SiLU, ReGLU, GeGLU, SwiGLU | 6 variantes à paramètres égaux |
| 10 | **Mixture of Experts** | top-k, équilibrage, z-loss, sans perte auxiliaire, Expert Choice, capacité, experts fins/partagés | 9 variantes à calcul actif égal |
| 11 | **Optimiseurs** | SGD, momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, Lion, Muon (formules LaTeX) | lr réglé par optimiseur ; L2 vs découplé |
| 12 | Schedules, écrêtage, précision | warmup, cosinus, WSD, clipping, accumulation, bf16 | 7 schedules, 3 warmups, 3 écrêtages |
| 13 | **Boucle d'entraînement** | toutes les options de `TrainConfig`, reprise exacte | — |
| 14 | Génération | greedy, T, top-k, top-p, min-p, pénalité, décodage spéculatif | qualité (grammaire) vs diversité |
| 15 | Architectures modernes | GPT-2, LLaMA, Mistral, Mixtral, Qwen3, DeepSeek-V3 (styles réduits) | qualité vs cache/calcul |
| 16 | Ablations finales | un facteur à la fois, différences appariées ± SE | 21 variantes, 3 seeds |

## Conventions d'expérience
- **Données** : corpus synthétique français à grammaire (`llm.data.synthetic`), reproductible, avec dépendances courtes et longues ;
  remplaçable par votre texte (`TEXTE = open(...).read()`).
- **Comparaison** : `compare(variants, data, base_model, base_train, seeds)` — mêmes seeds, mêmes lots de validation fixes ; tableau
  `val moyenne ± écart-type`, paramètres totaux/actifs, débit ; `*` marque la meilleure moyenne.
- **Honnêteté statistique** : un écart inférieur à ~2 écarts-types n'est pas conclusif avec 2–3 seeds. Les commentaires sous chaque
  expérience décrivent l'exécution fournie ; relire les tableaux après toute ré-exécution.

## Régénérer / exécuter
```bash
pip install -e ".[experiments,test]"
python scripts/build_notebooks.py                 # régénère les .ipynb (sans sorties)
python scripts/build_notebooks.py --execute       # régénère et exécute (≈ 1 h sur 1 cœur CPU)
python scripts/build_notebooks.py --execute --only 07 11
```
