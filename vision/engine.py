from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from shared.lora_ops import (
    compute_comm_bytes,
    compute_subspace_retention,
    find_q_proj_key,
    pairs_from_state,
)
from vision.config import VisionFedConfig
from vision.data import CIFAR100FedDataModule
from vision.evaluators.classification import VisionClassificationEvaluator
from vision.federated.lora_ops import infer_lora_rank
from vision.federated.aggregators import EGWSAAggregator, FLoRAAggregator, FlexLoRAAggregator, HETLORAAggregator, raFLoRAAggregator
from vision.federated.types import ClientInitState, ClientTrainPayload
from shared.hetlora_pruning import prune_lora_state
from vision.models.vit_lora import build_vit_lora, get_classifier_state_dict, get_lora_state_dict, load_federated_state
from vision.rank_policies import AdaptiveRankPolicy, FixedRankPolicy, HeuristicRankPolicy, RandomRankPolicy
from vision.trainer import VisionClientTrainer
from vision.utils import set_seed


def _select_clients(cfg: VisionFedConfig) -> List[int]:
    n = max(1, int(round(cfg.num_clients * cfg.participation_rate)))
    return list(range(n))


def _steps_per_round(dm: CIFAR100FedDataModule, clients: List[int], cfg: VisionFedConfig) -> int:
    total = 0
    for c in clients:
        total += cfg.local_epochs * len(dm.get_client_loader(c))
    return total


def run_vision_federated(cfg: VisionFedConfig) -> None:
    set_seed(cfg.seed)
    out_dir = cfg.resolved_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    dm = CIFAR100FedDataModule(cfg)
    trainer = VisionClientTrainer(cfg)

    _RANK_POLICIES = {
        "fixed": FixedRankPolicy,
        "random": RandomRankPolicy,
        "heuristic": HeuristicRankPolicy,
        "adaptive": AdaptiveRankPolicy,
    }
    if cfg.rank_policy not in _RANK_POLICIES:
        raise ValueError(f"Unknown rank_policy: {cfg.rank_policy}")
    rank_policy = _RANK_POLICIES[cfg.rank_policy]()

    if cfg.aggregation_method == "flora":
        aggregator = FLoRAAggregator()
    elif cfg.aggregation_method == "egwsa":
        aggregator = EGWSAAggregator()
    elif cfg.aggregation_method == "flexlora":
        aggregator = FlexLoRAAggregator()
    elif cfg.aggregation_method == "hetlora":
        aggregator = HETLORAAggregator()
    elif cfg.aggregation_method == "raflora":
        aggregator = raFLoRAAggregator()
    else:
        raise ValueError(cfg.aggregation_method)

    clients = _select_clients(cfg)
    sample_counts = {cid: dm.client_num_samples(cid) for cid in clients}

    # ---- Log per-client sample distribution ----
    total_samples = sum(sample_counts.values())
    print(f"\n[vision-engine] === Client data distribution ({len(clients)} clients) ===")
    for cid in clients:
        n = sample_counts[cid]
        bar = "#" * (n // 50)
        print(f"  Client {cid:2d}: {n:6d} samples  {bar}")
    print(f"  Total  : {total_samples:6d} samples  "
          f"min={min(sample_counts.values())}  "
          f"max={max(sample_counts.values())}  "
          f"ratio={max(sample_counts.values())/max(1,min(sample_counts.values())):.2f}x")

    dist_info = {
        "num_clients": len(clients),
        "total_samples": total_samples,
        "min_samples": min(sample_counts.values()),
        "max_samples": max(sample_counts.values()),
        "imbalance_ratio": round(max(sample_counts.values()) / max(1, min(sample_counts.values())), 4),
        "client_samples": {str(cid): sample_counts[cid] for cid in clients},
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "partition_method": cfg.partition_method,
        "patho_shards_per_client": cfg.patho_shards_per_client,
        "seed": cfg.seed,
    }
    dist_path = os.path.join(out_dir, "client_distribution.json")
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(dist_info, f, indent=2)
    print(f"  [saved] {dist_path}")

    steps_round = _steps_per_round(dm, clients, cfg)
    total_global_steps = max(1, cfg.num_rounds * steps_round)

    # Per-client federated state: LoRA + classifier
    client_init: Dict[int, ClientInitState] = {}
    # Server-side global LoRA (used by FLoRA for rank inference)
    prev_global_lora: Optional[dict] = None

    # Evaluation & communication tracking
    evaluator = VisionClassificationEvaluator(cfg)
    test_loader = dm.test_loader()
    cumulative_upload = 0
    cumulative_download = 0

    for round_idx in range(cfg.num_rounds):
        round_t0 = time.time()
        print(f"\n[vision-engine] === Round {round_idx}/{cfg.num_rounds} ===")
        ranks = rank_policy.ranks_for_round(cfg, sample_counts, round_idx)

        payloads: List[ClientTrainPayload] = []
        step_cursor = round_idx * steps_round

        # Per-round model cache: one base model instance per distinct LoRA rank.
        # Avoids reloading ViT pretrained weights for every client.
        # Before each client trains we reload its correct init state so no
        # client's training result contaminates the next client.
        models_by_rank: Dict[int, torch.nn.Module] = {}
        initial_lora_by_rank: Dict[int, dict] = {}
        initial_clf_by_rank: Dict[int, dict] = {}

        client_losses: Dict[int, float] = {}
        train_times: Dict[int, float] = {}

        for c in clients:
            # Determine rank for model construction
            if cfg.aggregation_method == "flora":
                if prev_global_lora is None:
                    rank_use = ranks[c]
                else:
                    rank_use = infer_lora_rank(prev_global_lora)
            else:
                rank_use = ranks[c]

            model_build_t0 = time.time()
            if rank_use not in models_by_rank:
                print(f"[vision-engine] Build cached model for rank={rank_use}")
                model = build_vit_lora(cfg, rank_use)
                models_by_rank[rank_use] = model
                initial_lora_by_rank[rank_use] = get_lora_state_dict(model)
                initial_clf_by_rank[rank_use] = get_classifier_state_dict(model)
            model_build_s = time.time() - model_build_t0

            model = models_by_rank[rank_use]

            # Load previous round's federated state (LoRA + classifier)
            init_state = client_init.get(c)
            if init_state is not None:
                load_federated_state(model, init_state.lora_state, init_state.classifier_state)
            else:
                load_federated_state(model, initial_lora_by_rank[rank_use], initial_clf_by_rank[rank_use])

            loader = dm.get_client_loader(c, shuffle=True)
            n_samples = sample_counts[c]
            n_batches_est = len(loader)

            client_t0 = time.time()
            out = trainer.train_one_round(model, loader, step_cursor, total_global_steps)
            client_train_s = time.time() - client_t0

            step_cursor += cfg.local_epochs * len(loader)
            client_losses[c] = out["mean_loss"]
            train_times[c] = client_train_s

            print(f"  [client {c:2d}] samples={n_samples:5d}  rank={rank_use:2d}  "
                  f"batches={n_batches_est:4d}  loss={out['mean_loss']:.4f}  "
                  f"model_prep={model_build_s:.1f}s  train={client_train_s:.1f}s")

            lora_state = out["lora_state"]
            if cfg.aggregation_method == "hetlora":
                lora_state = prune_lora_state(lora_state, cfg.hetlora_pruning_ratio)

            payloads.append(
                ClientTrainPayload(
                    client_id=c,
                    num_samples=dm.client_num_samples(c),
                    lora_state=lora_state,
                    rank=ranks[c],
                    classifier_state=out["classifier_state"],
                )
            )

        total_train_s = sum(train_times.values())
        avg_loss = sum(client_losses.values()) / max(1, len(client_losses))

        # Communication: upload bytes
        upload_bytes = sum(compute_comm_bytes(p.lora_state) for p in payloads)
        print(f"  [round {round_idx}] avg_loss={avg_loss:.4f}  "
              f"total_train_time={total_train_s:.1f}s  "
              f"per_client_losses={json.dumps({c: round(l, 4) for c, l in client_losses.items()})}")

        for _m in models_by_rank.values():
            del _m
        del models_by_rank, initial_lora_by_rank, initial_clf_by_rank
        torch.cuda.empty_cache()

        agg_t0 = time.time()
        template_lora = prev_global_lora if prev_global_lora is not None else payloads[0].lora_state
        agg = aggregator.aggregate(payloads, cfg, template_lora)
        agg_s = time.time() - agg_t0

        # Distribution phase: copy init states to clients
        dist_t0 = time.time()
        client_init = agg.client_init
        dist_s = time.time() - dist_t0
        print(f"  [round {round_idx}] aggregation_time={agg_s:.2f}s  dist_time={dist_s:.2f}s")

        # Communication: download bytes
        download_bytes = sum(
            compute_comm_bytes(s.lora_state) for s in client_init.values()
        )
        cumulative_upload += upload_bytes
        cumulative_download += download_bytes

        # Track global LoRA for FLoRA rank inference in next round
        if agg.server_state.lora_state is not None:
            prev_global_lora = agg.server_state.lora_state

        # ---- Final-round-only evaluation ----
        should_eval = round_idx == cfg.num_rounds - 1
        eval_result = {}
        eval_s = 0.0
        if should_eval:
            eval_t0 = time.time()
            _GLOBAL_METHODS = {"flora", "hetlora", "raflora"}
            if cfg.aggregation_method in _GLOBAL_METHODS:
                eval_result = evaluator.evaluate_flora_global(
                    agg.server_state.lora_state,
                    agg.server_state.classifier_state,
                    test_loader,
                )
            else:
                eval_result = evaluator.evaluate_personalized(
                    client_init, ranks, test_loader,
                    sample_counts, cfg.personalized_eval_aggregation,
                )
            eval_s = time.time() - eval_t0
            print(f"  [round {round_idx}] eval top1={eval_result['top1']:.4f}  "
                  f"top5={eval_result['top5']:.4f}  eval_time={eval_s:.2f}s")
        else:
            print(f"  [round {round_idx}] eval skipped (final round only)")

        # ---- Metrics ----
        meta = agg.server_state.metadata
        signal_retention = None
        if payloads and client_init:
            before_pairs = {p.client_id: pairs_from_state(p.lora_state) for p in payloads}
            sample_pairs = next(iter(before_pairs.values()), {})
            retention_q_key = find_q_proj_key(list(sample_pairs.keys()))
            if retention_q_key is not None:
                per_client = {}
                ordered_values = []
                for cid in clients:
                    init_state = client_init.get(cid)
                    if init_state is None or cid not in before_pairs:
                        continue
                    after_pairs = pairs_from_state(init_state.lora_state)
                    if retention_q_key not in before_pairs[cid] or retention_q_key not in after_pairs:
                        continue
                    before_A, before_B = before_pairs[cid][retention_q_key]
                    after_A, after_B = after_pairs[retention_q_key]
                    score = compute_subspace_retention(
                        before_B.float() @ before_A.float(),
                        after_B.float() @ after_A.float(),
                    )
                    score = round(score, 6)
                    per_client[str(cid)] = score
                    ordered_values.append(score)
                if ordered_values:
                    signal_retention = {
                        "pair_id": retention_q_key,
                        "per_client": per_client,
                        "values": ordered_values,
                        "mean": round(sum(ordered_values) / len(ordered_values), 6),
                    }

        round_s = time.time() - round_t0
        print(f"  [round {round_idx}] total_round_time={round_s:.1f}s")

        row = {
            "round": round_idx,
            "avg_loss": round(avg_loss, 6),
            "client_losses": {str(c): round(l, 6) for c, l in client_losses.items()},
            "eval": {
                "top1": round(eval_result["top1"], 6),
                "top5": round(eval_result["top5"], 6),
            } if eval_result else {},
            "comm": {
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
                "round_total_bytes": upload_bytes + download_bytes,
                "cumulative_upload_bytes": cumulative_upload,
                "cumulative_download_bytes": cumulative_download,
                "cumulative_total_bytes": cumulative_upload + cumulative_download,
            },
            "timing": {
                "train_s": round(total_train_s, 1),
                "agg_s": round(agg_s, 2),
                "dist_s": round(dist_s, 2),
                "eval_s": round(eval_s, 2),
                "round_s": round(round_s, 1),
            },
            "aggregation": cfg.aggregation_method,
            "rank_policy": cfg.rank_policy,
            "ranks": {str(c): ranks[c] for c in clients},
            "agg_mode": meta.get("mode", ""),
            "agg_implementation": meta.get("implementation", "paper_faithful"),
            "strict_paper_faithful": meta.get("strict_paper_faithful", True),
        }
        # Energy ratios (every round)
        if "energy_ratios" in meta:
            row["energy_ratios"] = meta["energy_ratios"]
            er = meta["energy_ratios"]
            q_key = find_q_proj_key(list(er.keys()))
            if q_key is not None:
                row["energy_ratio_q"] = er[q_key]
            if er:
                row["energy_ratio_mean"] = sum(float(v) for v in er.values()) / len(er)
        if "sv_snapshot" in meta:
            row["sv_after"] = meta["sv_snapshot"]
            row["sv_snapshot"] = meta["sv_snapshot"]
        if "sv_before" in meta:
            row["sv_before"] = meta["sv_before"]
        if signal_retention is not None:
            row["signal_retention"] = signal_retention
            row["signal_retention_values"] = signal_retention["values"]
            row["signal_retention_mean"] = signal_retention["mean"]
        if cfg.log_every_round:
            print(json.dumps(row, ensure_ascii=False))

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # ---- Checkpoint: save LoRA weights every round ----
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_t0 = time.time()
        if agg.server_state.lora_state is not None:
            ckpt_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}_global.pt")
            torch.save({
                "lora_state": agg.server_state.lora_state,
                "classifier_state": agg.server_state.classifier_state,
            }, ckpt_path)
        else:
            for cid, init_state in agg.client_init.items():
                ckpt_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}_client_{cid}.pt")
                torch.save({
                    "lora_state": init_state.lora_state,
                    "classifier_state": init_state.classifier_state,
                }, ckpt_path)
        ckpt_s = time.time() - ckpt_t0
        print(f"  [round {round_idx}] checkpoint_saved  path={ckpt_dir}  save_time={ckpt_s:.2f}s")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
