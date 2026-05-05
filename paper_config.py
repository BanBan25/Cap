"""
Paper-specific experiment configuration.
Covers only the models/datasets/methods in this paper:
  - ViT-Base + CIFAR-100 (vision)
  - LLaMA3-8B / Qwen3-14B + Alpaca / GSM8K / Commonsense15K (language)
  - FLoRA / EGWSA / FlexLoRA aggregation
  - fixed / random / heuristic / adaptive rank policy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

TaskType = Literal["vision", "language"]
AggregationMethod = Literal["flora", "egwsa", "flexlora", "hetlora", "raflora"]
RankPolicyName = Literal["fixed", "random", "heuristic", "adaptive"]
PartitionMethod = Literal["dirichlet", "patho"]
PersonalizedEvalAgg = Literal["uniform", "sample_weighted"]


# ---- Defaults per task type ----
VISION_DEFAULTS = dict(
    num_rounds=100,
    learning_rate=5e-4,
    batch_size=64,
    eval_batch_size=128,
    weight_decay=0.01,
    local_epochs=1,
    lora_target_modules=["query", "key", "value", "attention.output.dense"],
    num_classes=100,
    image_size=224,
    adaptive_k=100,
    max_seq_len=0,
)

LANGUAGE_DEFAULTS = dict(
    num_rounds=20,
    learning_rate=3e-4,
    batch_size=4,
    eval_batch_size=4,
    weight_decay=0.01,
    local_epochs=1,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    num_classes=0,
    image_size=0,
    adaptive_k=10,
    max_seq_len=512,
    candidate_ranks=[4, 8, 16, 32, 64],
    dynamic_padding=True,
    enable_gradient_checkpointing=True,
    max_cached_language_models=1,
)


@dataclass
class PaperFedConfig:
    """Unified config for all paper experiments."""

    # ---- Task / model / dataset ----
    task_type: TaskType = "vision"
    dataset: str = "cifar100"
    model_name: str = "google/vit-base-patch16-224"

    # ---- Federated setup ----
    num_clients: int = 10
    participation_rate: float = 1.0
    partition_method: PartitionMethod = "dirichlet"
    dirichlet_alpha: float = 0.5
    patho_shards_per_client: int = 2

    # ---- Training ----
    num_rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 5e-4
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.01

    # ---- LoRA ----
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "attention.output.dense"]
    )
    rank_policy: RankPolicyName = "fixed"
    fixed_rank: int = 16
    candidate_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 48, 64])

    # ---- Aggregation ----
    aggregation_method: AggregationMethod = "flora"
    egwsa_num_iters: int = 5
    hetlora_pruning_ratio: float = 0.3

    # ---- Evaluation (used by eval_checkpoint.py post-training) ----
    personalized_eval_aggregation: PersonalizedEvalAgg = "uniform"
    eval_batch_size: int = 128
    eval_data_root: str = ""         # if empty, defaults to {data_root}/benchmarks
    eval_max_samples: int = 200      # max samples per benchmark for efficiency

    # ---- vLLM acceleration (used when --use-vllm is set) ----
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: Optional[int] = None  # None = auto from model config
    vllm_dtype: str = "auto"                  # "auto", "float16", "bfloat16"

    # ---- Vision-specific ----
    num_classes: int = 100
    image_size: int = 224
    adaptive_k: int = 100

    # ---- Language-specific ----
    max_seq_len: int = 512
    gradient_accumulation_steps: int = 4
    dynamic_padding: bool = True
    enable_gradient_checkpointing: bool = True
    max_cached_language_models: int = 1

    # ---- Infrastructure ----
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    data_root: str = "./data"
    output_dir: str = "./outputs"
    split_cache_dir: str = "./data/fed_splits"
    log_every_round: bool = True

    def apply_task_defaults(self) -> None:
        """Apply task-specific defaults for fields that were not explicitly set via CLI."""
        defaults = VISION_DEFAULTS if self.task_type == "vision" else LANGUAGE_DEFAULTS
        for k, v in defaults.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def resolved_output_dir(self) -> str:
        import os
        if self.partition_method == "dirichlet":
            partition_tag = f"a{self.dirichlet_alpha}"
        else:
            partition_tag = f"patho_spc{self.patho_shards_per_client}"
        rank_tag = f"_r{self.fixed_rank}" if self.rank_policy == "fixed" else ""
        return os.path.join(
            self.output_dir,
            f"{self.task_type}_{self.dataset}_{self.aggregation_method}_{self.rank_policy}_c{self.num_clients}_{partition_tag}{rank_tag}",
        )
