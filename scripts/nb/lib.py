"""Outils de génération des notebooks : cellules, boîte à outils autonome, références, exécution."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "llm"

# ----------------------------------------------------------------------------- références
# clé : (auteurs, année, titre, venue, url)
REFS = {
 "vaswani2017": ("Vaswani et al.", 2017, "Attention Is All You Need", "NeurIPS", "https://arxiv.org/abs/1706.03762"),
 "sennrich2016": ("Sennrich, Haddow & Birch", 2016, "Neural Machine Translation of Rare Words with Subword Units", "ACL", "https://arxiv.org/abs/1508.07909"),
 "gage1994": ("Gage", 1994, "A New Algorithm for Data Compression", "The C Users Journal 12(2)", ""),
 "radford2019": ("Radford et al.", 2019, "Language Models are Unsupervised Multitask Learners (GPT-2)", "OpenAI technical report", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
 "wu2016": ("Wu et al.", 2016, "Google's Neural Machine Translation System (WordPiece)", "arXiv", "https://arxiv.org/abs/1609.08144"),
 "kudo2018": ("Kudo", 2018, "Subword Regularization: Improving NMT Models with Multiple Subword Candidates (Unigram LM)", "ACL", "https://arxiv.org/abs/1804.10959"),
 "kudo2018sp": ("Kudo & Richardson", 2018, "SentencePiece: A simple and language independent subword tokenizer", "EMNLP (demo)", "https://arxiv.org/abs/1808.06226"),
 "mikolov2013": ("Mikolov et al.", 2013, "Efficient Estimation of Word Representations in Vector Space", "arXiv", "https://arxiv.org/abs/1301.3781"),
 "press2017": ("Press & Wolf", 2017, "Using the Output Embedding to Improve Language Models", "EACL", "https://arxiv.org/abs/1608.05859"),
 "ba2016": ("Ba, Kiros & Hinton", 2016, "Layer Normalization", "arXiv", "https://arxiv.org/abs/1607.06450"),
 "zhang2019": ("Zhang & Sennrich", 2019, "Root Mean Square Layer Normalization", "NeurIPS", "https://arxiv.org/abs/1910.07467"),
 "xiong2020": ("Xiong et al.", 2020, "On Layer Normalization in the Transformer Architecture", "ICML", "https://arxiv.org/abs/2002.04745"),
 "henry2020": ("Henry et al.", 2020, "Query-Key Normalization for Transformers", "Findings of EMNLP", "https://arxiv.org/abs/2010.04245"),
 "dehghani2023": ("Dehghani et al.", 2023, "Scaling Vision Transformers to 22 Billion Parameters", "ICML", "https://arxiv.org/abs/2302.05442"),
 "he2016": ("He et al.", 2016, "Deep Residual Learning for Image Recognition", "CVPR", "https://arxiv.org/abs/1512.03385"),
 "shazeer2020": ("Shazeer", 2020, "GLU Variants Improve Transformer", "arXiv", "https://arxiv.org/abs/2002.05202"),
 "dauphin2017": ("Dauphin et al.", 2017, "Language Modeling with Gated Convolutional Networks", "ICML", "https://arxiv.org/abs/1612.08083"),
 "hendrycks2016": ("Hendrycks & Gimpel", 2016, "Gaussian Error Linear Units (GELUs)", "arXiv", "https://arxiv.org/abs/1606.08415"),
 "ramachandran2017": ("Ramachandran, Zoph & Le", 2017, "Searching for Activation Functions (Swish/SiLU)", "arXiv", "https://arxiv.org/abs/1710.05941"),
 "su2021": ("Su et al.", 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding", "Neurocomputing 2024", "https://arxiv.org/abs/2104.09864"),
 "press2022": ("Press, Smith & Lewis", 2022, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)", "ICLR", "https://arxiv.org/abs/2108.12409"),
 "kazemnejad2023": ("Kazemnejad et al.", 2023, "The Impact of Positional Encoding on Length Generalization in Transformers (NoPE)", "NeurIPS", "https://arxiv.org/abs/2305.19466"),
 "chen2023pi": ("Chen et al.", 2023, "Extending Context Window of Large Language Models via Positional Interpolation", "arXiv", "https://arxiv.org/abs/2306.15595"),
 "peng2023": ("Peng et al.", 2023, "YaRN: Efficient Context Window Extension of Large Language Models", "ICLR 2024", "https://arxiv.org/abs/2309.00071"),
 "shazeer2019": ("Shazeer", 2019, "Fast Transformer Decoding: One Write-Head is All You Need (MQA)", "arXiv", "https://arxiv.org/abs/1911.02150"),
 "ainslie2023": ("Ainslie et al.", 2023, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "EMNLP", "https://arxiv.org/abs/2305.13245"),
 "deepseekv2": ("DeepSeek-AI", 2024, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA)", "arXiv", "https://arxiv.org/abs/2405.04434"),
 "deepseekv3": ("DeepSeek-AI", 2024, "DeepSeek-V3 Technical Report", "arXiv", "https://arxiv.org/abs/2412.19437"),
 "beltagy2020": ("Beltagy, Peters & Cohan", 2020, "Longformer: The Long-Document Transformer", "arXiv", "https://arxiv.org/abs/2004.05150"),
 "jiang2023": ("Jiang et al.", 2023, "Mistral 7B", "arXiv", "https://arxiv.org/abs/2310.06825"),
 "dao2022": ("Dao et al.", 2022, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "NeurIPS", "https://arxiv.org/abs/2205.14135"),
 "dao2023": ("Dao", 2023, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", "ICLR 2024", "https://arxiv.org/abs/2307.08691"),
 "kwon2023": ("Kwon et al.", 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)", "SOSP", "https://arxiv.org/abs/2309.06180"),
 "pope2022": ("Pope et al.", 2022, "Efficiently Scaling Transformer Inference", "MLSys 2023", "https://arxiv.org/abs/2211.05102"),
 "xiao2024": ("Xiao et al.", 2024, "Efficient Streaming Language Models with Attention Sinks (StreamingLLM)", "ICLR", "https://arxiv.org/abs/2309.17453"),
 "liu2024kivi": ("Liu et al.", 2024, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", "ICML", "https://arxiv.org/abs/2402.02750"),
 "hooper2024": ("Hooper et al.", 2024, "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", "NeurIPS", "https://arxiv.org/abs/2401.18079"),
 "shazeer2017": ("Shazeer et al.", 2017, "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", "ICLR", "https://arxiv.org/abs/1701.06538"),
 "lepikhin2021": ("Lepikhin et al.", 2021, "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", "ICLR", "https://arxiv.org/abs/2006.16668"),
 "fedus2022": ("Fedus, Zoph & Shazeer", 2022, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", "JMLR 23", "https://arxiv.org/abs/2101.03961"),
 "zoph2022": ("Zoph et al.", 2022, "ST-MoE: Designing Stable and Transferable Sparse Expert Models", "arXiv", "https://arxiv.org/abs/2202.08906"),
 "zhou2022": ("Zhou et al.", 2022, "Mixture-of-Experts with Expert Choice Routing", "NeurIPS", "https://arxiv.org/abs/2202.09368"),
 "jiang2024": ("Jiang et al.", 2024, "Mixtral of Experts", "arXiv", "https://arxiv.org/abs/2401.04088"),
 "dai2024": ("Dai et al.", 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", "ACL", "https://arxiv.org/abs/2401.06066"),
 "wang2024": ("Wang et al.", 2024, "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", "arXiv", "https://arxiv.org/abs/2408.15664"),
 "muennighoff2024": ("Muennighoff et al.", 2024, "OLMoE: Open Mixture-of-Experts Language Models", "arXiv", "https://arxiv.org/abs/2409.02060"),
 "touvron2023": ("Touvron et al.", 2023, "LLaMA: Open and Efficient Foundation Language Models", "arXiv", "https://arxiv.org/abs/2302.13971"),
 "touvron2023b": ("Touvron et al.", 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models", "arXiv", "https://arxiv.org/abs/2307.09288"),
 "dubey2024": ("Grattafiori et al. (Llama Team)", 2024, "The Llama 3 Herd of Models", "arXiv", "https://arxiv.org/abs/2407.21783"),
 "qwen2024": ("Yang et al. (Qwen Team)", 2024, "Qwen2 Technical Report", "arXiv", "https://arxiv.org/abs/2407.10671"),
 "qwen3": ("Yang et al. (Qwen Team)", 2025, "Qwen3 Technical Report", "arXiv", "https://arxiv.org/abs/2505.09388"),
 "gemma2": ("Gemma Team", 2024, "Gemma 2: Improving Open Language Models at a Practical Size", "arXiv", "https://arxiv.org/abs/2408.00118"),
 "gemma3": ("Gemma Team", 2025, "Gemma 3 Technical Report", "arXiv", "https://arxiv.org/abs/2503.19786"),
 "zhu2024hc": ("Zhu et al.", 2024, "Hyper-Connections", "ICLR 2025", "https://arxiv.org/abs/2409.19606"),
 "xie2025mhc": ("Xie et al. (DeepSeek-AI)", 2025, "mHC: Manifold-Constrained Hyper-Connections", "arXiv", "https://arxiv.org/abs/2512.24880"),
 "kingma2015": ("Kingma & Ba", 2015, "Adam: A Method for Stochastic Optimization", "ICLR", "https://arxiv.org/abs/1412.6980"),
 "loshchilov2019": ("Loshchilov & Hutter", 2019, "Decoupled Weight Decay Regularization (AdamW)", "ICLR", "https://arxiv.org/abs/1711.05101"),
 "loshchilov2017": ("Loshchilov & Hutter", 2017, "SGDR: Stochastic Gradient Descent with Warm Restarts", "ICLR", "https://arxiv.org/abs/1608.03983"),
 "polyak1964": ("Polyak", 1964, "Some methods of speeding up the convergence of iteration methods", "USSR Comput. Math. Math. Phys. 4(5)", ""),
 "nesterov1983": ("Nesterov", 1983, "A method for solving the convex programming problem with convergence rate O(1/k²)", "Dokl. Akad. Nauk SSSR 269", ""),
 "sutskever2013": ("Sutskever et al.", 2013, "On the importance of initialization and momentum in deep learning", "ICML", ""),
 "duchi2011": ("Duchi, Hazan & Singer", 2011, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (Adagrad)", "JMLR 12", "https://jmlr.org/papers/v12/duchi11a.html"),
 "tieleman2012": ("Tieleman & Hinton", 2012, "Lecture 6.5 – RMSProp", "COURSERA: Neural Networks for Machine Learning", ""),
 "reddi2018": ("Reddi, Kale & Kumar", 2018, "On the Convergence of Adam and Beyond", "ICLR", "https://arxiv.org/abs/1904.09237"),
 "chen2023lion": ("Chen et al.", 2023, "Symbolic Discovery of Optimization Algorithms (Lion)", "NeurIPS", "https://arxiv.org/abs/2302.06675"),
 "jordan2024": ("Jordan et al.", 2024, "Muon: An optimizer for hidden layers in neural networks", "billet de blog", "https://kellerjordan.github.io/posts/muon/"),
 "liu2025": ("Liu et al.", 2025, "Muon is Scalable for LLM Training (Moonlight)", "arXiv", "https://arxiv.org/abs/2502.16982"),
 "shazeer2018": ("Shazeer & Stern", 2018, "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost", "ICML", "https://arxiv.org/abs/1804.04235"),
 "pascanu2013": ("Pascanu, Mikolov & Bengio", 2013, "On the difficulty of training recurrent neural networks", "ICML", "https://arxiv.org/abs/1211.5063"),
 "goyal2017": ("Goyal et al.", 2017, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", "arXiv", "https://arxiv.org/abs/1706.02677"),
 "hu2024": ("Hu et al.", 2024, "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (WSD)", "arXiv", "https://arxiv.org/abs/2404.06395"),
 "micikevicius2018": ("Micikevicius et al.", 2018, "Mixed Precision Training", "ICLR", "https://arxiv.org/abs/1710.03740"),
 "chen2016": ("Chen et al.", 2016, "Training Deep Nets with Sublinear Memory Cost", "arXiv", "https://arxiv.org/abs/1604.06174"),
 "kaplan2020": ("Kaplan et al.", 2020, "Scaling Laws for Neural Language Models", "arXiv", "https://arxiv.org/abs/2001.08361"),
 "hoffmann2022": ("Hoffmann et al.", 2022, "Training Compute-Optimal Large Language Models (Chinchilla)", "NeurIPS", "https://arxiv.org/abs/2203.15556"),
 "chowdhery2022": ("Chowdhery et al.", 2022, "PaLM: Scaling Language Modeling with Pathways", "JMLR 24", "https://arxiv.org/abs/2204.02311"),
 "leviathan2023": ("Leviathan, Kalman & Matias", 2023, "Fast Inference from Transformers via Speculative Decoding", "ICML", "https://arxiv.org/abs/2211.17192"),
 "holtzman2020": ("Holtzman et al.", 2020, "The Curious Case of Neural Text Degeneration (top-p)", "ICLR", "https://arxiv.org/abs/1904.09751"),
 "nguyen2024": ("Nguyen et al.", 2024, "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", "ICLR 2025", "https://arxiv.org/abs/2407.01082"),
 "keskar2019": ("Keskar et al.", 2019, "CTRL: A Conditional Transformer Language Model for Controllable Generation", "arXiv", "https://arxiv.org/abs/1909.05858"),
 "gloeckle2024": ("Gloeckle et al.", 2024, "Better & Faster Large Language Models via Multi-token Prediction", "ICML", "https://arxiv.org/abs/2404.19737"),
}


def refs_block(*keys) -> str:
    lines = ["## Références", ""]
    for i, k in enumerate(keys, 1):
        a, y, t, v, u = REFS[k]
        link = f" [lien]({u})" if u else ""
        lines.append(f"{i}. **{a} ({y})**. *{t}*. {v}.{link}")
    return "\n".join(lines)


def cite(key) -> str:
    a, y, *_ = REFS[key]
    return f"{a.split(' et al.')[0].split(',')[0].split(' &')[0]} ({y})" if " et al." not in a else f"{a} ({y})"


def write_reference_files():
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    md = ["# Références bibliographiques", "",
          "Généré par `scripts/build_notebooks.py` à partir de `scripts/nb/lib.py`. **Vérifiez chaque entrée avant tout dépôt** "
          "(identifiants arXiv, versions, venues) : les liens ont été saisis manuellement, seuls certains ont été recontrôlés en ligne.", ""]
    bib = []
    for k, (a, y, t, v, u) in sorted(REFS.items(), key=lambda kv: (kv[1][1], kv[0])):
        md.append(f"- **[{k}]** {a} ({y}). *{t}*. {v}." + (f" <{u}>" if u else ""))
        author = a.replace(" et al.", " and others").replace(" & ", " and ").replace(" (Llama Team)", "").replace(" (Qwen Team)", "").replace(" (DeepSeek-AI)", "")
        bib.append(f"@misc{{{k},\n  author = {{{author}}},\n  title = {{{t}}},\n  year = {{{y}}},\n  note = {{{v}}},\n" + (f"  url = {{{u}}},\n" if u else "") + "}\n")
    (docs / "REFERENCES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (docs / "references.bib").write_text("\n".join(bib), encoding="utf-8")


# ----------------------------------------------------------------------------- boîte à outils autonome
MODULES = ["model/config.py", "model/norms.py", "model/positions.py", "model/cache.py", "model/attention.py",
           "model/ffn.py", "model/moe.py", "model/block.py", "model/model.py", "training/optimizers.py",
           "training/schedules.py", "training/trainer.py", "training/compare.py", "generation/sampling.py",
           "generation/generate.py", "data/synthetic.py"]


def toolbox(names=None) -> str:
    """Concatène les modules du paquet ``llm`` en UN bloc de code sans import relatif (notebooks autonomes)."""
    parts = ['"""Boîte à outils autonome : copie exacte des modules du paquet `llm` (testés par pytest).\n'
             'Aucun import de fichier local : tout est défini ici."""']
    for m in (names or MODULES):
        src = (SRC / m).read_text(encoding="utf-8")
        src = re.sub(r'^"""[\s\S]*?"""\n', "", src, count=1)
        src = "\n".join(l for l in src.splitlines() if not re.match(r"\s*from \.|\s*from llm|\s*from __future__", l))
        parts.append(f"# ===== {m} =====\n{src.strip()}\n")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- cellules
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, hidden: bool = False) -> dict:
    meta = {"jupyter": {"source_hidden": True}, "tags": ["cellule-technique"]} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def tool(*names) -> dict:
    return code(toolbox(list(names) if names else None), hidden=True)


PRELUDE_IMPORTS = """import math, time, random
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
torch.set_num_threads(2)"""

DATA_CELL = """# Corpus synthétique français (grammaire probabiliste) -> tokens caractère. Reproductible, sans téléchargement.
# Pour utiliser votre propre texte : TEXTE = open("mon_corpus.txt", encoding="utf-8").read()
TEXTE = make_corpus(150_000, seed=0)
CHARS = sorted(set(TEXTE)); STOI = {c: i for i, c in enumerate(CHARS)}
IDS = torch.tensor([STOI[c] for c in TEXTE]); N = int(0.9 * len(IDS))
TRAIN, VAL, VOCAB = IDS[:N], IDS[N:], len(CHARS)
DATA = (TRAIN, VAL, VOCAB)
print(f"{len(TEXTE):,} caractères | vocabulaire {VOCAB} | train {len(TRAIN):,} | val {len(VAL):,}")
print(TEXTE[:150])"""


def build(path: Path, cells: list, title: str, keep_outputs: bool = False):
    """Écrit le notebook. Si ``keep_outputs`` : réutilise les sorties d'une version existante pour les cellules de code inchangées."""
    if keep_outputs and path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        by_src = {"".join(c["source"]): c for c in old["cells"] if c["cell_type"] == "code" and c.get("outputs")}
        for c in cells:
            hit = by_src.get("".join(c["source"])) if c["cell_type"] == "code" else None
            if hit:
                c["outputs"], c["execution_count"] = hit["outputs"], hit.get("execution_count")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "title": title},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):
        c["id"] = f"c{i:03d}"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def execute(path: Path, timeout: int = 1800):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)   # en cas d'erreur, on conserve les sorties partielles pour diagnostic
