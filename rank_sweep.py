"""
Rank-sweep experiment for RQ3 (Fig.5 / Fig.6).

For each selected client, keep all other clients at their adaptive ranks and
sweep the target client's rank over a candidate set. Record final global /
personalized evaluation metrics for each sweep point.

Output JSON also includes ``rq3_behavior_aggregate``: mean metric vs. assigned
rank for three client groups (C_lg / C_md / C_sm) by tertiles of theoretical
r_i^min among ``--sweep-clients`` — for rq3_behavior-style plots.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import tempfile
import time
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from paper_config import LANGUAGE_DEFAULTS, VISION_DEFAULTS, PaperFedConfig
from shared.aggregators import (
    EGWSAAggregator,
    FLoRAAggregator,
    FlexLoRAAggregator,
    HETLORAAggregator,
    raFLoRAAggregator,
)
from shared.hetlora_pruning import prune_lora_state
from shared.lora_ops import infer_lora_rank
from shared.rank_policies import AdaptiveRankPolicy, theoretical_rank_from_ni
from shared.seed import set_seed
from shared.flora_log_context import reset_sweep_context, set_sweep_context
from shared.types import ClientInitState, ClientTrainPayload


def parse_args() -> Namespace:
    p = ArgumentParser(description="Rank-sweep experiment for RQ3")
    p.add_argument("--task-type", choices=["vision", "language"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--aggregation", choices=["flora", "egwsa", "flexlora", "hetlora", "raflora"], default="egwsa")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--partition-method", choices=["dirichlet", "patho"], default="dirichlet")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--patho-shards-per-client", type=int, default=2)
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--sweep-clients", type=int, nargs="+", required=True)
    p.add_argument("--sweep-ranks", type=int, nargs="+", default=None)
    p.add_argument("--inflection-tol", type=float, default=0.0025,
                   help="Absolute tolerance on metric (fraction units) for empirical inflection rank.")
    p.add_argument("--eval-max-samples", type=int, default=200)
    p.add_argument("--use-vllm", action=BooleanOptionalAction, default=None,
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
    p.add_argument("--output-dir", default="outputs/rank_sweep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--split-cache-dir", default="./data/fed_splits")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def build_cfg(args: Namespace) -> PaperFedConfig:
    defaults = VISION_DEFAULTS if args.task_type == "vision" else LANGUAGE_DEFAULTS
    model_name = args.model_name
    if model_name is None:
        model_name = (
            "google/vit-base-patch16-224"
            if args.task_type == "vision"
            else "meta-llama/Meta-Llama-3-8B"
        )
    cfg = PaperFedConfig(
        task_type=args.task_type,
        dataset=args.dataset,
        model_name=model_name,
        aggregation_method=args.aggregation,
        rank_policy="adaptive",
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        patho_shards_per_client=args.patho_shards_per_client,
        num_rounds=args.num_rounds or defaults["num_rounds"],
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        split_cache_dir=args.split_cache_dir,
        num_workers=args.num_workers,
        eval_max_samples=args.eval_max_samples,
    )
    for k, v in defaults.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.num_rounds = args.num_rounds or defaults["num_rounds"]
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.model_name = model_name
    cfg.num_clients = args.num_clients
    cfg.participation_rate = args.participation_rate
    cfg.partition_method = args.partition_method
    cfg.dirichlet_alpha = args.dirichlet_alpha
    cfg.patho_shards_per_client = args.patho_shards_per_client
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.data_root = args.data_root
    cfg.split_cache_dir = args.split_cache_dir
    cfg.num_workers = args.num_workers
    cfg.eval_max_samples = args.eval_max_samples
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


def make_aggregator(name: str):
    mapping = {
        "flora": FLoRAAggregator,
        "egwsa": EGWSAAggregator,
        "flexlora": FlexLoRAAggregator,
        "hetlora": HETLORAAggregator,
        "raflora": raFLoRAAggregator,
    }
    return mapping[name]()


def summarize_plot_metric(dataset: str, metrics: Dict[str, float]) -> tuple[str, float | None]:
    if dataset == "cifar100":
        return "top1", metrics.get("top1")
    if dataset == "gsm8k":
        return "em", metrics.get("em")
    if dataset == "commonsense15k":
        keys = ["hellaswag_acc", "winogrande_acc", "arc_challenge_acc", "piqa_acc"]
        vals = [metrics[k] for k in keys if isinstance(metrics.get(k), (int, float))]
        return "commonsense_avg", (sum(vals) / len(vals) if vals else None)
    if dataset == "alpaca":
        return "mmlu", metrics.get("mmlu")
    return "metric", None


def compact_eval_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    return {k: v for k, v in metrics.items() if not k.startswith("__")}


def build_manual_ranks(base_ranks: Dict[int, int], sweep_client: int, sweep_rank: int) -> Dict[int, int]:
    ranks = dict(base_ranks)
    ranks[sweep_client] = sweep_rank
    return ranks


def assign_rq3_clusters(swept_cids: List[int], theoretical_ranks: Dict[str, int]) -> Dict[str, List[int]]:
    """Assign swept clients to C_lg / C_md / C_sm by descending theoretical r_i^min.

    C_lg = largest bounds (high sample complexity clients); C_sm = smallest.
    Degenerate sizes: n=1 → only C_md; n=2 → C_lg + C_sm.
    """
    ids = sorted(set(swept_cids), key=lambda c: theoretical_ranks[str(c)], reverse=True)
    n = len(ids)
    out: Dict[str, List[int]] = {"C_lg": [], "C_md": [], "C_sm": []}
    if n == 0:
        return out
    if n == 1:
        out["C_md"] = ids
        return out
    if n == 2:
        out["C_lg"] = [ids[0]]
        out["C_sm"] = [ids[1]]
        return out
    for i, cid in enumerate(ids):
        b = min(2, (i * 3) // n)
        key = ("C_lg", "C_md", "C_sm")[b]
        out[key].append(cid)
    return out


def compute_rq3_behavior_aggregate(
    results: Dict[str, Dict[str, dict]],
    theoretical_ranks: Dict[str, int],
    sweep_ranks: List[int],
    dataset: str,
) -> dict:
    """Aggregate sweep curves for rq3_behavior-style plots (mean per client cluster × rank)."""
    swept = sorted(int(k) for k in results.keys())
    clusters = assign_rq3_clusters(swept, theoretical_ranks)

    plot_metric_name: Optional[str] = None
    for cid_str in results:
        for rank_str, payload in results[cid_str].items():
            nm = payload.get("plot_metric_name")
            if isinstance(nm, str):
                plot_metric_name = nm
                break
        if plot_metric_name:
            break

    def cluster_stats(cids: List[int]) -> dict:
        tr_vals = [float(theoretical_ranks[str(c)]) for c in cids if str(c) in theoretical_ranks]
        if not tr_vals:
            return {"theoretical_rmin_median": None, "theoretical_rmin_min": None, "theoretical_rmin_max": None}
        tr_sorted = sorted(tr_vals)
        med = statistics.median(tr_sorted)
        return {
            "theoretical_rmin_median": float(med),
            "theoretical_rmin_min": float(min(tr_sorted)),
            "theoretical_rmin_max": float(max(tr_sorted)),
        }

    curves: Dict[str, dict] = {}
    any_val: List[float] = []
    for ck, cids in clusters.items():
        y_means: List[Optional[float]] = []
        for r in sweep_ranks:
            vals: List[float] = []
            for cid in cids:
                cell = results.get(str(cid), {}).get(str(r))
                if not cell:
                    continue
                pv = cell.get("plot_metric_value")
                if isinstance(pv, (int, float)):
                    vals.append(float(pv))
            if vals:
                m = sum(vals) / len(vals)
                y_means.append(m)
                any_val.append(m)
            else:
                y_means.append(None)

        stats = cluster_stats(cids)
        curves[ck] = {
            "client_ids": cids,
            **stats,
            "y_mean": y_means,
        }

    use_percent = False
    if any_val and max(any_val) <= 1.000001:
        use_percent = True
    # Language benchmarks sometimes already report 0–100 in edge evaluators; guard.
    if any_val and max(any_val) > 1.5:
        use_percent = False

    for ck in curves:
        raw = curves[ck]["y_mean"]
        if use_percent:
            curves[ck]["y_mean_percent"] = [None if v is None else round(v * 100.0, 6) for v in raw]
        else:
            curves[ck]["y_mean_percent"] = [None if v is None else round(v, 6) for v in raw]

    return {
        "description": (
            "RQ3 behavior-style aggregate: clients partitioned into C_lg/C_md/C_sm by tertiles "
            "of theoretical r_i^min among sweep_clients; y is mean plot_metric_value per rank."
        ),
        "dataset": dataset,
        "plot_metric_name": plot_metric_name,
        "y_primary": "y_mean_percent" if use_percent else "y_mean",
        "values_are_fraction": use_percent,
        "cluster_assignment": clusters,
        "sweep_ranks": list(sweep_ranks),
        "curves": curves,
    }


def empirical_inflection_rank(
    ranked_results: Dict[str, dict],
    tol: float,
) -> float | None:
    """Smallest rank whose metric is within tol of the maximum sweep metric."""
    pairs: List[tuple[int, float]] = []
    for rank_str, payload in ranked_results.items():
        metric = payload.get("plot_metric_value")
        if isinstance(metric, (int, float)):
            pairs.append((int(rank_str), float(metric)))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    best = max(v for _, v in pairs)
    for rank, val in pairs:
        if best - val <= tol:
            return float(rank)
    return float(pairs[-1][0])


def build_sweep_payload(
    cfg: PaperFedConfig,
    base_ranks: Dict[int, int],
    sample_counts: Dict[int, int],
    theoretical_ranks: Dict[str, int],
    sweep_ranks: List[int],
    inflection_tol: float,
    results: Dict[str, Dict[str, dict]],
    last_checkpoint: dict | None = None,
) -> dict:
    max_n = max(sample_counts.values()) if sample_counts else 1
    normalized_sample_counts = {
        str(cid): (sample_counts[cid] / max_n if max_n > 0 else 0.0)
        for cid in sample_counts
    }
    empirical_inflections = {
        cid: empirical_inflection_rank(rank_results, inflection_tol)
        for cid, rank_results in results.items()
    }
    rq3_agg = compute_rq3_behavior_aggregate(
        results, theoretical_ranks, sweep_ranks, cfg.dataset,
    )
    payload = {
        "config": asdict(cfg),
        "base_ranks": {str(k): v for k, v in base_ranks.items()},
        "sample_counts": {str(k): v for k, v in sample_counts.items()},
        "normalized_sample_counts": normalized_sample_counts,
        "theoretical_ranks": theoretical_ranks,
        "empirical_inflection_ranks": empirical_inflections,
        "sweep_ranks": sweep_ranks,
        "inflection_tol": inflection_tol,
        "sweep_results": results,
        "rq3_behavior_aggregate": rq3_agg,
    }
    if last_checkpoint is not None:
        payload["last_checkpoint"] = last_checkpoint
    return payload


def atomic_write_json(path: str, obj: dict) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".sweep_", suffix=".json.tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)


def run_vision_sweep_once(
    cfg: PaperFedConfig,
    manual_ranks: Dict[int, int],
) -> Dict[str, float]:
    from vision.config import VisionFedConfig
    from vision.data import CIFAR100FedDataModule
    from vision.evaluators import VisionClassificationEvaluator
    from vision.models.vit_lora import (
        build_vit_lora,
        get_classifier_state_dict,
        get_lora_state_dict,
        load_federated_state,
    )
    from vision.trainer import VisionClientTrainer

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
        candidate_ranks=list(cfg.candidate_ranks),
        aggregation_method=cfg.aggregation_method,
        egwsa_num_iters=cfg.egwsa_num_iters,
        hetlora_pruning_ratio=cfg.hetlora_pruning_ratio,
        seed=cfg.seed,
        device=cfg.device,
        data_root=cfg.data_root,
        split_cache_dir=cfg.split_cache_dir,
        num_workers=cfg.num_workers,
        personalized_eval_aggregation=cfg.personalized_eval_aggregation,
    )

    from shared.flora_log_context import log_prefix

    dm = CIFAR100FedDataModule(vcfg)
    trainer = VisionClientTrainer(vcfg)
    evaluator = VisionClassificationEvaluator(vcfg)
    clients = list(range(max(1, int(round(vcfg.num_clients * vcfg.participation_rate)))))
    sample_counts = {cid: dm.client_num_samples(cid) for cid in clients}
    steps_round = sum(vcfg.local_epochs * len(dm.get_client_loader(c)) for c in clients)
    total_global_steps = max(1, vcfg.num_rounds * steps_round)

    print(
        f"{log_prefix()}[rank-sweep][fed] start  rounds={vcfg.num_rounds}  "
        f"clients={len(clients)}  agg={vcfg.aggregation_method}",
        flush=True,
    )

    client_init: Dict[int, ClientInitState] = {}
    prev_global_lora = None

    for round_idx in range(vcfg.num_rounds):
        print(
            f"{log_prefix()}[rank-sweep][fed] round {round_idx + 1}/{vcfg.num_rounds}",
            flush=True,
        )
        payloads: List[ClientTrainPayload] = []
        step_cursor = round_idx * steps_round
        models_by_rank: Dict[int, torch.nn.Module] = {}
        init_lora_by_rank: Dict[int, dict] = {}
        init_cls_by_rank: Dict[int, dict] = {}

        for cid in clients:
            if vcfg.aggregation_method == "flora" and prev_global_lora is not None:
                rank_use = infer_lora_rank(prev_global_lora)
            else:
                rank_use = manual_ranks[cid]

            if rank_use not in models_by_rank:
                model = build_vit_lora(vcfg, rank_use)
                models_by_rank[rank_use] = model
                init_lora_by_rank[rank_use] = get_lora_state_dict(model)
                init_cls_by_rank[rank_use] = get_classifier_state_dict(model)

            model = models_by_rank[rank_use]
            init_state = client_init.get(cid)
            if init_state is not None:
                load_federated_state(model, init_state.lora_state, init_state.classifier_state)
            else:
                load_federated_state(model, init_lora_by_rank[rank_use], init_cls_by_rank[rank_use])

            loader = dm.get_client_loader(cid, shuffle=True)
            out = trainer.train_one_round(model, loader, step_cursor, total_global_steps)
            step_cursor += vcfg.local_epochs * len(loader)

            lora_state = out["lora_state"]
            if vcfg.aggregation_method == "hetlora":
                lora_state = prune_lora_state(lora_state, vcfg.hetlora_pruning_ratio)

            payloads.append(
                ClientTrainPayload(
                    client_id=cid,
                    num_samples=sample_counts[cid],
                    lora_state=lora_state,
                    rank=manual_ranks[cid],
                    classifier_state=out["classifier_state"],
                )
            )

        aggregator = make_aggregator(vcfg.aggregation_method)
        template_lora = prev_global_lora if prev_global_lora is not None else payloads[0].lora_state
        agg = aggregator.aggregate(payloads, vcfg, template_lora)
        client_init = agg.client_init
        if agg.server_state.lora_state is not None:
            prev_global_lora = agg.server_state.lora_state

        for model in models_by_rank.values():
            del model
        torch.cuda.empty_cache()

    print(f"{log_prefix()}[rank-sweep][fed] training done → eval", flush=True)
    test_loader = dm.test_loader()
    if vcfg.aggregation_method in {"flora", "hetlora", "raflora"}:
        return evaluator.evaluate_flora_global(
            agg.server_state.lora_state,
            agg.server_state.classifier_state,
            test_loader,
        )
    return evaluator.evaluate_personalized(
        client_init,
        manual_ranks,
        test_loader,
        sample_counts,
        vcfg.personalized_eval_aggregation,
    )


def run_language_sweep_once(
    cfg: PaperFedConfig,
    manual_ranks: Dict[int, int],
) -> Dict[str, float]:
    from language.data import LANGUAGE_DATA_MODULES
    from language.evaluators import LanguageEvaluator
    from language.models.causal_lm_lora import (
        build_causal_lm_lora,
        build_tokenizer,
        get_lora_state_dict,
        load_lora_state,
    )
    from language.trainer import LanguageClientTrainer

    from shared.flora_log_context import log_prefix

    tokenizer = build_tokenizer(cfg)
    dm = LANGUAGE_DATA_MODULES[cfg.dataset](cfg, tokenizer)
    dm.setup()
    trainer = LanguageClientTrainer(cfg)
    evaluator = LanguageEvaluator(cfg, tokenizer, use_vllm=getattr(cfg, "use_vllm", False))

    clients = list(range(max(1, int(round(cfg.num_clients * cfg.participation_rate)))))
    sample_counts = {cid: dm.client_num_samples(cid) for cid in clients}
    steps_round = sum(cfg.local_epochs * len(dm.get_client_loader(c)) for c in clients)
    total_global_steps = max(1, cfg.num_rounds * steps_round)

    print(
        f"{log_prefix()}[rank-sweep][fed] start  rounds={cfg.num_rounds}  "
        f"clients={len(clients)}  agg={cfg.aggregation_method}  dataset={cfg.dataset}",
        flush=True,
    )

    client_init: Dict[int, ClientInitState] = {}
    prev_global_lora = None

    for round_idx in range(cfg.num_rounds):
        print(
            f"{log_prefix()}[rank-sweep][fed] round {round_idx + 1}/{cfg.num_rounds}",
            flush=True,
        )
        payloads: List[ClientTrainPayload] = []
        step_cursor = round_idx * steps_round
        rank_to_clients: Dict[int, List[int]] = {}
        for cid in clients:
            if cfg.aggregation_method == "flora" and prev_global_lora is not None:
                rank_use = infer_lora_rank(prev_global_lora)
            else:
                rank_use = manual_ranks[cid]
            rank_to_clients.setdefault(rank_use, []).append(cid)

        for rank_use, rank_clients in rank_to_clients.items():
            model, _, _ = build_causal_lm_lora(cfg, rank_use, tokenizer=tokenizer)
            init_lora_state = copy.deepcopy(get_lora_state_dict(model))
            for cid in rank_clients:
                init_state = client_init.get(cid)
                if init_state is not None:
                    load_lora_state(model, init_state.lora_state)
                else:
                    load_lora_state(model, init_lora_state)

                loader = dm.get_client_loader(cid, shuffle=True)
                out = trainer.train_one_round(model, loader, step_cursor, total_global_steps)
                step_cursor += cfg.local_epochs * len(loader)

                lora_state = out["lora_state"]
                if cfg.aggregation_method == "hetlora":
                    lora_state = prune_lora_state(lora_state, cfg.hetlora_pruning_ratio)

                payloads.append(
                    ClientTrainPayload(
                        client_id=cid,
                        num_samples=sample_counts[cid],
                        lora_state=lora_state,
                        rank=manual_ranks[cid],
                        classifier_state=None,
                    )
                )
            del model
            torch.cuda.empty_cache()

        aggregator = make_aggregator(cfg.aggregation_method)
        template_lora = prev_global_lora if prev_global_lora is not None else payloads[0].lora_state
        agg = aggregator.aggregate(payloads, cfg, template_lora)
        client_init = agg.client_init
        if agg.server_state.lora_state is not None:
            prev_global_lora = agg.server_state.lora_state

        torch.cuda.empty_cache()

    print(f"{log_prefix()}[rank-sweep][fed] training done → eval", flush=True)
    if cfg.aggregation_method in {"flora", "hetlora", "raflora"}:
        metrics = evaluator.evaluate_flora_global(agg.server_state.lora_state)
    else:
        metrics = evaluator.evaluate_personalized(
            client_init,
            manual_ranks,
            client_sample_counts=sample_counts,
            aggregation=cfg.personalized_eval_aggregation,
        )
    return compact_eval_metrics(metrics)


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    adaptive = AdaptiveRankPolicy()
    if cfg.task_type == "vision":
        from vision.config import VisionFedConfig
        from vision.data import CIFAR100FedDataModule

        vcfg = VisionFedConfig(
            num_clients=cfg.num_clients,
            participation_rate=cfg.participation_rate,
            partition_method=cfg.partition_method,
            dirichlet_alpha=cfg.dirichlet_alpha,
            patho_shards_per_client=cfg.patho_shards_per_client,
            seed=cfg.seed,
            data_root=cfg.data_root,
            split_cache_dir=cfg.split_cache_dir,
            num_workers=cfg.num_workers,
        )
        dm = CIFAR100FedDataModule(vcfg)
        clients = list(range(max(1, int(round(cfg.num_clients * cfg.participation_rate)))))
        sample_counts = {cid: dm.client_num_samples(cid) for cid in clients}
    else:
        from language.data import LANGUAGE_DATA_MODULES
        from language.models.causal_lm_lora import build_tokenizer

        tokenizer = build_tokenizer(cfg)
        dm = LANGUAGE_DATA_MODULES[cfg.dataset](cfg, tokenizer)
        dm.setup()
        clients = list(range(max(1, int(round(cfg.num_clients * cfg.participation_rate)))))
        sample_counts = {cid: dm.client_num_samples(cid) for cid in clients}

    base_ranks = adaptive.ranks_for_round(cfg, sample_counts, round_idx=0)
    theoretical_ranks = {
        str(cid): theoretical_rank_from_ni(int(n), cfg.adaptive_k)
        for cid, n in sample_counts.items()
    }
    sweep_ranks = args.sweep_ranks or [2, 4, 8, 16, 32, 64]

    if cfg.partition_method == "dirichlet":
        partition_tag = f"a{cfg.dirichlet_alpha}"
    else:
        partition_tag = f"patho_spc{cfg.patho_shards_per_client}"
    out_path = os.path.join(
        args.output_dir,
        f"sweep_{cfg.task_type}_{cfg.dataset}_{cfg.aggregation_method}_{partition_tag}.json",
    )

    results: Dict[str, Dict[str, dict]] = {}
    for cid in args.sweep_clients:
        results[str(cid)] = {}
        for rank in sweep_ranks:
            manual_ranks = build_manual_ranks(base_ranks, cid, rank)
            print(f"\n{'=' * 72}")
            print(f"[rank_sweep] client={cid}  rank={rank}  aggregation={cfg.aggregation_method}")
            print(f"{'=' * 72}")
            t0 = time.time()
            tok = set_sweep_context(cid, rank)
            prev_quiet = os.environ.get("FLORA_QUIET_VIT_LOAD")
            prev_hf_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            prev_tf_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
            os.environ["FLORA_QUIET_VIT_LOAD"] = "1"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            try:
                if cfg.task_type == "vision":
                    metrics = run_vision_sweep_once(cfg, manual_ranks)
                else:
                    metrics = run_language_sweep_once(cfg, manual_ranks)
            finally:
                reset_sweep_context(tok)
                if prev_quiet is None:
                    os.environ.pop("FLORA_QUIET_VIT_LOAD", None)
                else:
                    os.environ["FLORA_QUIET_VIT_LOAD"] = prev_quiet
                if prev_hf_disable is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_disable
                if prev_tf_verbosity is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_tf_verbosity
            elapsed = round(time.time() - t0, 2)
            metric_name, metric_value = summarize_plot_metric(cfg.dataset, metrics)
            results[str(cid)][str(rank)] = {
                "metrics": metrics,
                "plot_metric_name": metric_name,
                "plot_metric_value": metric_value,
                "elapsed_s": elapsed,
            }
            print(f"[rank_sweep] metrics={metrics}  plot_metric={metric_name}:{metric_value}  elapsed={elapsed:.1f}s")

            payload = build_sweep_payload(
                cfg,
                base_ranks,
                sample_counts,
                theoretical_ranks,
                sweep_ranks,
                args.inflection_tol,
                results,
                last_checkpoint={
                    "sweep_client": cid,
                    "sweep_rank": rank,
                    "wall_time_epoch_s": round(time.time(), 3),
                },
            )
            atomic_write_json(out_path, payload)
            print(f"[rank_sweep] checkpoint saved → {out_path}")

    print(f"\n[rank_sweep] Final results at {out_path}")


if __name__ == "__main__":
    main()
