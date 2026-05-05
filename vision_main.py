"""
Vision federated learning entrypoint (ViT + CIFAR-100).
Run from repo root, e.g.:
  python vision_main.py --aggregation flora --rank-policy fixed --num-rounds 2
"""

from __future__ import annotations

import argparse

from vision.config import VisionFedConfig
from vision.engine import run_vision_federated


def main() -> None:
    p = argparse.ArgumentParser(description="Flora vision federated (ViT / CIFAR-100)")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--partition-method", choices=["dirichlet", "patho"], default="dirichlet")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--patho-shards-per-client", type=int, default=2)
    p.add_argument("--num-rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--fixed-rank", type=int, default=16)
    p.add_argument("--rank-policy", choices=["fixed", "random", "heuristic", "adaptive"], default="fixed")
    p.add_argument("--aggregation", choices=["flora", "egwsa", "flexlora", "hetlora", "raflora"], default="flora")
    p.add_argument("--egwsa-num-iters", type=int, default=5)
    p.add_argument("--hetlora-pruning-ratio", type=float, default=0.3,
                   help="HETLORA local self-pruning ratio (default: 0.3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./outputs/vision_fed")
    p.add_argument("--split-cache-dir", type=str, default="./data/cifar100_fed_splits")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--personalized-eval-aggregation",
        choices=["uniform", "sample_weighted"],
        default="uniform",
    )
    args = p.parse_args()

    cfg = VisionFedConfig(
        dataset=args.dataset,
        model_name=args.model_name,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        patho_shards_per_client=args.patho_shards_per_client,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fixed_rank=args.fixed_rank,
        rank_policy=args.rank_policy,
        aggregation_method=args.aggregation,
        egwsa_num_iters=args.egwsa_num_iters,
        hetlora_pruning_ratio=args.hetlora_pruning_ratio,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_cache_dir=args.split_cache_dir,
        num_workers=args.num_workers,
        personalized_eval_aggregation=args.personalized_eval_aggregation,
    )
    run_vision_federated(cfg)


if __name__ == "__main__":
    main()
