from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

AggregationMethod = Literal["flora", "egwsa", "flexlora", "hetlora", "raflora"]
RankPolicyName = Literal["fixed", "random", "heuristic", "adaptive"]
PartitionMethod = Literal["dirichlet", "patho"]
PersonalizedEvalAgg = Literal["uniform", "sample_weighted"]


@dataclass
class VisionFedConfig:
    """Minimal experiment configuration for federated vision (extensible via new fields)."""

    dataset: str = "cifar100"
    model_name: str = "google/vit-base-patch16-224"
    num_clients: int = 10
    participation_rate: float = 1.0
    partition_method: PartitionMethod = "dirichlet"
    dirichlet_alpha: float = 0.5
    patho_shards_per_client: int = 2
    num_rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 5e-4
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.01
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "attention.output.dense"]
    )
    rank_policy: RankPolicyName = "fixed"
    fixed_rank: int = 16
    candidate_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 48, 64])
    aggregation_method: AggregationMethod = "flora"
    egwsa_num_iters: int = 5
    hetlora_pruning_ratio: float = 0.3
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0
    data_root: str = "./data"
    output_dir: str = "./outputs/vision_fed"
    split_cache_dir: str = "./data/cifar100_fed_splits"
    image_size: int = 224
    num_classes: int = 100
    adaptive_k: int = 100
    eval_batch_size: int = 128
    log_every_round: bool = True
    personalized_eval_aggregation: PersonalizedEvalAgg = "uniform"

    def resolved_output_dir(self) -> str:
        import os
        if self.partition_method == "dirichlet":
            partition_tag = f"a{self.dirichlet_alpha}"
        else:
            partition_tag = f"patho_spc{self.patho_shards_per_client}"
        rank_tag = f"_r{self.fixed_rank}" if self.rank_policy == "fixed" else ""

        return os.path.join(
            self.output_dir,
            f"{self.dataset}_{self.aggregation_method}_{self.rank_policy}_c{self.num_clients}_{partition_tag}{rank_tag}",
        )
