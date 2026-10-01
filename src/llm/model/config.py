"""Configuration unique du modèle : toutes les variantes d'architecture sont des champs ici."""
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration sérialisable d'un Transformer décodeur (GPT-2, LLaMA, Mixtral, DeepSeek...)."""
    # --- dimensions générales ---
    vocab_size: int = 128
    block_size: int = 64          # longueur de contexte d'entraînement
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False            # les LLM modernes n'utilisent généralement pas de biais
    tie_weights: bool = True      # partage embedding d'entrée / tête de sortie
    emb_scale: bool = False       # multiplie l'embedding par sqrt(n_embd) (convention du Transformer original ; utile avec positions sinusoïdales)

    # --- position : none | learned | sinusoidal | rope | alibi ---
    position_mode: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: Optional[str] = None   # None | linear | ntk | yarn
    rope_factor: float = 1.0             # facteur d'extension de contexte s
    rope_orig_len: Optional[int] = None  # longueur d'origine (YaRN) ; défaut = block_size

    # --- normalisation ---
    norm_type: str = "rmsnorm"    # layernorm | rmsnorm
    pre_norm: bool = True         # True = Pre-LN, False = Post-LN
    norm_eps: float = 1e-5
    qk_norm: bool = False         # RMSNorm sur Q et K avant le produit scalaire

    # --- attention : mha | mqa | gqa | mla ---
    attn_type: str = "mha"
    n_kv_head: Optional[int] = None      # GQA : nombre de têtes K/V (mha = n_head, mqa = 1)
    window: Optional[int] = None         # fenêtre glissante (None = attention complète)
    sdpa: bool = False                   # utiliser F.scaled_dot_product_attention
    # MLA (Multi-head Latent Attention, DeepSeek-V2)
    kv_lora_rank: int = 16               # dimension du latent KV compressé
    q_lora_rank: Optional[int] = None    # compression de Q (None = pas de compression)
    rope_head_dim: int = 8               # partie de la tête portant RoPE (clé partagée)

    # --- FFN : relu | gelu | silu | reglu | geglu | swiglu ---
    ffn_type: str = "swiglu"
    ffn_mult: float = 4.0                # largeur cachée = ffn_mult * n_embd (FFN classique)
    ffn_multiple_of: int = 8             # arrondi de la largeur cachée des FFN à porte

    # --- Mixture of Experts ---
    moe: bool = False
    n_experts: int = 8
    top_k: int = 2
    n_shared_experts: int = 0            # experts partagés toujours actifs (DeepSeekMoE)
    expert_hidden: Optional[int] = None  # largeur d'un expert (None = largeur FFN dense)
    router: str = "softmax"              # softmax | sigmoid_bias (sans perte auxiliaire) | expert_choice
    balance_coef: float = 0.01           # coefficient de la perte d'équilibrage (Switch)
    z_loss_coef: float = 0.0             # coefficient de la z-loss du routeur (ST-MoE)
    capacity_factor: Optional[float] = None  # None = pas de capacité (aucun token abandonné)
    n_dense_layers: int = 0              # premières couches restant denses
    bias_update_speed: float = 1e-3      # pas de mise à jour du biais (sigmoid_bias)

    def __post_init__(self):
        if self.n_embd % self.n_head:
            raise ValueError("n_embd doit être divisible par n_head")
        if self.attn_type not in {"mha", "mqa", "gqa", "mla"}:
            raise ValueError(f"attn_type inconnu : {self.attn_type}")
        if self.attn_type == "mha":
            self.n_kv_head = self.n_head
        elif self.attn_type == "mqa":
            self.n_kv_head = 1
        elif self.attn_type == "gqa":
            self.n_kv_head = self.n_kv_head or max(1, self.n_head // 2)
            if self.n_head % self.n_kv_head:
                raise ValueError("n_head doit être divisible par n_kv_head")
        if self.position_mode not in {"none", "learned", "sinusoidal", "rope", "alibi"}:
            raise ValueError(f"position_mode inconnu : {self.position_mode}")
        if self.position_mode == "rope" and self.head_dim % 2 and self.attn_type != "mla":
            raise ValueError("RoPE exige une dimension de tête paire")
        if self.attn_type == "mla" and self.position_mode == "rope" and self.rope_head_dim % 2:
            raise ValueError("rope_head_dim doit être pair")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError(f"norm_type inconnu : {self.norm_type}")
        if self.ffn_type not in {"relu", "gelu", "silu", "reglu", "geglu", "swiglu"}:
            raise ValueError(f"ffn_type inconnu : {self.ffn_type}")
        if self.router not in {"softmax", "sigmoid_bias", "expert_choice"}:
            raise ValueError(f"router inconnu : {self.router}")
        if self.moe and not (1 <= self.top_k <= self.n_experts):
            raise ValueError("1 <= top_k <= n_experts est requis")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
