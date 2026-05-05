"""
Unified paper-specific federated experiment entry point.

Covers all model/dataset combinations in the paper:
  Vision:   ViT-Base + CIFAR-100
  Language:  LLaMA3-8B + {Alpaca, GSM8K, Commonsense15K}
             Qwen3-14B + {Alpaca, GSM8K, Commonsense15K}

Usage examples:
  # Vision (same as vision_main.py)
  python paper_main.py --task-type vision --dataset cifar100 --model-name google/vit-base-patch16-224 --aggregation flora

  # Language
  python paper_main.py --task-type language --dataset alpaca --model-name meta-llama/Meta-Llama-3-8B --aggregation egwsa
  python paper_main.py --task-type language --dataset gsm8k --model-name Qwen/Qwen3-14B --aggregation flora
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

from paper_config import PaperFedConfig, VISION_DEFAULTS, LANGUAGE_DEFAULTS


MODEL_ALIASES = {
    "vit_base": "google/vit-base-patch16-224",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

VALID_COMBINATIONS = {
    "vision": {
        "datasets": ["cifar100"],
        "models": ["google/vit-base-patch16-224", "vit_base"],
    },
    "language": {
        "datasets": ["alpaca", "gsm8k", "commonsense15k"],
        "models": [
            "meta-llama/Meta-Llama-3-8B", "llama3_8b",
            "Qwen/Qwen3-14B", "qwen3_14b",
        ],
    },
}

PAPER_MODEL_FAMILIES: dict[str, list[dict]] = {
    "vision": [
        {
            "basename_re": re.compile(r"vit[\-_]?base[\-_]?patch16[\-_]?224", re.IGNORECASE),
            "hf_marker": "preprocessor_config.json",
        },
    ],
    "language": [
        {
            "basename_re": re.compile(r"(meta[\-_]?)?llama[\-_]?3[\-_]?8b", re.IGNORECASE),
            "hf_marker": None,
        },
        {
            "basename_re": re.compile(r"qwen[\-_]?3[\-_]?14b", re.IGNORECASE),
            "hf_marker": None,
        },
    ],
}


def _resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _is_local_model_dir(path: str) -> bool:
    """True when *path* points to an existing directory that looks like a HF model checkout."""
    if not os.path.isdir(path):
        return False
    children = set(os.listdir(path))
    return bool(children & {"config.json", "adapter_config.json"})


def _infer_paper_model_family_for_task(
    task_type: str, path: str
) -> Optional[str]:
    """Return the matched basename pattern string if *path* belongs to a paper-allowed
    model family for *task_type*, else ``None``."""
    families = PAPER_MODEL_FAMILIES.get(task_type, [])
    basename = os.path.basename(os.path.normpath(path))
    for fam in families:
        if fam["basename_re"].search(basename):
            marker = fam["hf_marker"]
            if marker is not None and not os.path.isfile(os.path.join(path, marker)):
                continue
            return fam["basename_re"].pattern
    return None


def _is_allowed_local_model_for_task(task_type: str, path: str) -> bool:
    """Check whether a local directory is a paper-allowed model for *task_type*.

    A language model dir must contain at least one tokenizer artifact;
    a vision model dir must contain ``preprocessor_config.json``.
    Both must contain ``config.json``.
    """
    if not _is_local_model_dir(path):
        return False
    if _infer_paper_model_family_for_task(task_type, path) is None:
        return False
    children = set(os.listdir(path))
    if task_type == "language":
        tokenizer_artifacts = {"tokenizer.json", "tokenizer_config.json", "tokenizer.model"}
        if not (children & tokenizer_artifacts):
            return False
    return True


def _validate_paper_combination(task_type: str, dataset: str, model_raw: str) -> None:
    """Reject any task/dataset/model combination not covered by this paper.

    Accepts three forms of ``model_raw``:
      1. Alias  (``vit_base``, ``llama3_8b``, ``qwen3_14b``)
      2. HF repo name (``google/vit-base-patch16-224``, …)
      3. Local model directory whose basename matches a paper model family
    """
    if task_type not in VALID_COMBINATIONS:
        raise ValueError(
            f"Invalid task_type '{task_type}'. "
            f"This paper only supports: {list(VALID_COMBINATIONS.keys())}"
        )
    valid = VALID_COMBINATIONS[task_type]

    if dataset not in valid["datasets"]:
        raise ValueError(
            f"Dataset '{dataset}' is not supported for task_type='{task_type}'. "
            f"Supported datasets: {valid['datasets']}"
        )

    resolved = _resolve_model_name(model_raw)
    allowed_resolved = {_resolve_model_name(m) for m in valid["models"]}
    if resolved in allowed_resolved:
        return

    if _is_allowed_local_model_for_task(task_type, model_raw):
        print(f"[paper_main] Accepted local model directory: {model_raw}")
        return

    friendly = [m for m in valid["models"] if m not in MODEL_ALIASES.values()]
    hf_names = [m for m in valid["models"] if m in MODEL_ALIASES.values()]
    raise ValueError(
        f"Model '{model_raw}' is not supported for task_type='{task_type}'. "
        f"Supported aliases: {friendly}, HF names: {hf_names}. "
        f"You may also pass a local directory whose basename matches one of "
        f"these models (e.g. /path/to/Meta-Llama-3-8B)."
    )


def build_config_from_args(args: argparse.Namespace) -> PaperFedConfig:
    _validate_paper_combination(args.task_type, args.dataset, args.model_name)
    # Local paths must be kept as-is; only aliases get resolved to HF names.
    model_name = (
        args.model_name
        if os.path.isdir(args.model_name)
        else _resolve_model_name(args.model_name)
    )

    cfg = PaperFedConfig(
        task_type=args.task_type,
        dataset=args.dataset,
        model_name=model_name,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        patho_shards_per_client=args.patho_shards_per_client,
        aggregation_method=args.aggregation,
        rank_policy=args.rank_policy,
        fixed_rank=args.fixed_rank,
        egwsa_num_iters=args.egwsa_num_iters,
        hetlora_pruning_ratio=args.hetlora_pruning_ratio,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_cache_dir=args.split_cache_dir,
        num_workers=args.num_workers,
        personalized_eval_aggregation=args.personalized_eval_aggregation,
        eval_data_root=args.eval_data_root,
        eval_max_samples=args.eval_max_samples,
    )

    # Apply task-specific defaults
    defaults = VISION_DEFAULTS if args.task_type == "vision" else LANGUAGE_DEFAULTS
    for k, v in defaults.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Paper runs only need final-round metrics for these datasets.
    dataset_round_overrides = {
        "cifar100": 20,
        "gsm8k": 10,
        "commonsense15k": 10,
    }
    if args.dataset in dataset_round_overrides:
        cfg.num_rounds = dataset_round_overrides[args.dataset]

    # CLI overrides (only if user explicitly passed them)
    if args.num_rounds is not None:
        cfg.num_rounds = args.num_rounds
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps

    auto_use_vllm = args.task_type == "language" and args.dataset == "gsm8k"
    cfg.use_vllm = auto_use_vllm if args.use_vllm is None else args.use_vllm
    if args.vllm_tp is not None:
        cfg.vllm_tensor_parallel_size = args.vllm_tp
    if args.vllm_gpu_util is not None:
        cfg.vllm_gpu_memory_utilization = args.vllm_gpu_util
    if args.vllm_max_model_len is not None:
        cfg.vllm_max_model_len = args.vllm_max_model_len
    if args.vllm_dtype is not None:
        cfg.vllm_dtype = args.vllm_dtype

    return cfg


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paper-specific federated experiment runner (FLoRA / EGWSA / FlexLoRA / HETLORA)"
    )

    # ---- Required ----
    p.add_argument("--task-type", choices=["vision", "language"], required=True)
    p.add_argument("--dataset", type=str, required=True,
                   help="cifar100 | alpaca | gsm8k | commonsense15k")
    p.add_argument("--model-name", type=str, required=True,
                   help="HF name, alias (vit_base/llama3_8b/qwen3_14b), or local dir")
    p.add_argument("--aggregation", choices=["flora", "egwsa", "flexlora", "hetlora", "raflora"], required=True)

    # ---- Federated ----
    p.add_argument("--rank-policy", choices=["fixed", "random", "heuristic", "adaptive"], default="fixed")
    p.add_argument("--fixed-rank", type=int, default=16)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--partition-method", choices=["dirichlet", "patho"], default="dirichlet")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--patho-shards-per-client", type=int, default=2,
                   help="Shards per client for pathological non-IID split.")
    p.add_argument("--egwsa-num-iters", type=int, default=5)
    p.add_argument("--hetlora-pruning-ratio", type=float, default=0.3,
                   help="HETLORA local self-pruning ratio (default: 0.3)")

    # ---- Training (None = use task defaults) ----
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)

    # ---- Eval ----
    p.add_argument("--personalized-eval-aggregation",
                   choices=["uniform", "sample_weighted"], default="uniform")
    p.add_argument("--eval-data-root", type=str, default="",
                   help="Root dir for benchmark eval data. Default: {data_root}/benchmarks")
    p.add_argument("--eval-max-samples", type=int, default=200,
                   help="Max samples per benchmark during evaluation (default: 200)")
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=None,
                   help="Use vLLM for evaluation. Defaults to enabled for GSM8K language runs.")
    p.add_argument("--vllm-tp", type=int, default=None,
                   help="vLLM tensor_parallel_size (default: 1). Set >1 for multi-GPU.")
    p.add_argument("--vllm-gpu-util", type=float, default=None,
                   help="vLLM gpu_memory_utilization (default: 0.85).")
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                   help="vLLM max_model_len. None = auto from model config.")
    p.add_argument("--vllm-dtype", type=str, default=None,
                   choices=["auto", "float16", "bfloat16"],
                   help="vLLM dtype (default: auto). Use float16 for T4/V100.")

    # ---- Infra ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--split-cache-dir", type=str, default="./data/fed_splits")
    p.add_argument("--num-workers", type=int, default=0)

    args = p.parse_args()
    cfg = build_config_from_args(args)

    print(f"[paper_main] task={cfg.task_type} dataset={cfg.dataset} model={cfg.model_name}")
    print(f"[paper_main] aggregation={cfg.aggregation_method} rank_policy={cfg.rank_policy} "
          f"rank={cfg.fixed_rank} clients={cfg.num_clients}")
    if cfg.partition_method == "dirichlet":
        print(f"[paper_main] partition=dirichlet alpha={cfg.dirichlet_alpha}")
    else:
        print(f"[paper_main] partition=patho shards_per_client={cfg.patho_shards_per_client}")
    print(f"[paper_main] rounds={cfg.num_rounds} lr={cfg.learning_rate} bs={cfg.batch_size}")
    if cfg.task_type == "language":
        print(
            f"[paper_main] eval_use_vllm={cfg.use_vllm} "
            f"vllm_tp={cfg.vllm_tensor_parallel_size} "
            f"vllm_dtype={cfg.vllm_dtype}"
        )

    if cfg.task_type == "vision":
        from vision.engine import run_vision_federated
        from vision.config import VisionFedConfig

        vcfg = VisionFedConfig(
            dataset=cfg.dataset,
            model_name=cfg.model_name,
            num_clients=cfg.num_clients,
            participation_rate=cfg.participation_rate,
            partition_method=cfg.partition_method,
            dirichlet_alpha=cfg.dirichlet_alpha,
            patho_shards_per_client=cfg.patho_shards_per_client,
            num_rounds=cfg.num_rounds,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            fixed_rank=cfg.fixed_rank,
            rank_policy=cfg.rank_policy,
            aggregation_method=cfg.aggregation_method,
            egwsa_num_iters=cfg.egwsa_num_iters,
            hetlora_pruning_ratio=cfg.hetlora_pruning_ratio,
            seed=cfg.seed,
            device=cfg.device,
            data_root=cfg.data_root,
            output_dir=cfg.output_dir,
            split_cache_dir=cfg.split_cache_dir,
            num_workers=cfg.num_workers,
            personalized_eval_aggregation=cfg.personalized_eval_aggregation,
        )
        run_vision_federated(vcfg)

    elif cfg.task_type == "language":
        from language.engine import run_language_federated
        run_language_federated(cfg)

    else:
        print(f"Unknown task_type: {cfg.task_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
