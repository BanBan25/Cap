"""
Language federated learning engine.
Runs the federated round loop for LLaMA3-8B / Qwen3-14B + Alpaca / GSM8K / Commonsense15K.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from language.data import LANGUAGE_DATA_MODULES
from language.models.causal_lm_lora import build_causal_lm_lora, build_tokenizer, get_lora_state_dict, load_lora_state
from language.trainer import LanguageClientTrainer
from paper_config import PaperFedConfig
from shared.aggregators import EGWSAAggregator, FLoRAAggregator, FlexLoRAAggregator, HETLORAAggregator, raFLoRAAggregator
from shared.hetlora_pruning import prune_lora_state
from shared.lora_ops import (
    compute_comm_bytes,
    compute_subspace_retention,
    find_q_proj_key,
    infer_lora_rank,
    pairs_from_state,
)
from shared.rank_policies import AdaptiveRankPolicy, FixedRankPolicy, HeuristicRankPolicy, RandomRankPolicy
from shared.seed import set_seed
from shared.types import ClientInitState, ClientTrainPayload
from language.evaluators.language_evaluator import LanguageEvaluator


def _select_clients(cfg: PaperFedConfig) -> List[int]:
    n = max(1, int(round(cfg.num_clients * cfg.participation_rate)))
    return list(range(n))


def run_language_federated(cfg: PaperFedConfig) -> None:
    set_seed(cfg.seed)
    out_dir = cfg.resolved_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    # ---- Tokenizer only — no model loaded at this stage ----
    print(f"[lang-engine] Loading tokenizer from {cfg.model_name} ...")
    tokenizer = build_tokenizer(cfg)
    evaluator = LanguageEvaluator(cfg, tokenizer, use_vllm=getattr(cfg, 'use_vllm', False))

    # ---- Data ----
    dm_cls = LANGUAGE_DATA_MODULES.get(cfg.dataset)
    if dm_cls is None:
        raise ValueError(f"Unknown language dataset: {cfg.dataset}. "
                         f"Supported: {list(LANGUAGE_DATA_MODULES.keys())}")
    dm = dm_cls(cfg, tokenizer)
    print(f"[lang-engine] Setting up {cfg.dataset} data module ...")
    dm.setup()

    trainer = LanguageClientTrainer(cfg)

    # ---- Rank policy ----
    _RANK_POLICIES = {
        "fixed": FixedRankPolicy,
        "random": RandomRankPolicy,
        "heuristic": HeuristicRankPolicy,
        "adaptive": AdaptiveRankPolicy,
    }
    if cfg.rank_policy not in _RANK_POLICIES:
        raise ValueError(f"Unknown rank_policy: {cfg.rank_policy}")
    rank_policy = _RANK_POLICIES[cfg.rank_policy]()

    # ---- Aggregator ----
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
    print(f"\n[lang-engine] === Client data distribution ({len(clients)} clients) ===")
    for cid in clients:
        n = sample_counts[cid]
        bar = "#" * (n // 200)
        print(f"  Client {cid:2d}: {n:6d} samples  {bar}")
    print(f"  Total  : {total_samples:6d} samples  "
          f"min={min(sample_counts.values())}  "
          f"max={max(sample_counts.values())}  "
          f"ratio={max(sample_counts.values())/max(1,min(sample_counts.values())):.2f}x")

    # Save client distribution to disk
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

    def _steps_per_round() -> int:
        total = 0
        for c in clients:
            total += cfg.local_epochs * len(dm.get_client_loader(c))
        return total

    steps_round = _steps_per_round()
    total_global_steps = max(1, cfg.num_rounds * steps_round)

    # Save config at start so eval_checkpoint.py can use it even if training is interrupted
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    client_init: Dict[int, ClientInitState] = {}
    prev_global_lora: Optional[dict] = None

    # ---- Bounded model cache across rounds ----
    # Keep only a small number of full HF models on GPU. This dramatically
    # lowers VRAM for adaptive rank policies used by GSM8K / Commonsense15K.
    models_by_rank: "OrderedDict[int, torch.nn.Module]" = OrderedDict()
    initial_lora_by_rank: Dict[int, dict] = {}
    max_cached_models = max(1, int(getattr(cfg, "max_cached_language_models", 1)))

    def _evict_rank(rank: int) -> None:
        model_to_drop = models_by_rank.pop(rank, None)
        initial_lora_by_rank.pop(rank, None)
        if model_to_drop is not None:
            del model_to_drop
            gc.collect()
            torch.cuda.empty_cache()

    def _remember_rank(rank: int, model: torch.nn.Module) -> None:
        if rank in models_by_rank:
            models_by_rank.pop(rank)
        models_by_rank[rank] = model
        while len(models_by_rank) > max_cached_models:
            evict_rank, evict_model = models_by_rank.popitem(last=False)
            initial_lora_by_rank.pop(evict_rank, None)
            del evict_model
            gc.collect()
            torch.cuda.empty_cache()

    def _release_training_model_cache() -> None:
        """Free cached HF training models before vLLM evaluation claims GPU."""
        nonlocal models_by_rank, initial_lora_by_rank
        if not models_by_rank:
            return
        print("[lang-engine] Releasing cached training models before vLLM eval ...")
        for _model in models_by_rank.values():
            del _model
        models_by_rank = OrderedDict()
        initial_lora_by_rank = {}
        gc.collect()
        torch.cuda.empty_cache()

    cumulative_upload_bytes = 0
    cumulative_download_bytes = 0

    for round_idx in range(cfg.num_rounds):
        round_t0 = time.time()
        print(f"\n[lang-engine] === Round {round_idx}/{cfg.num_rounds} ===")
        ranks = rank_policy.ranks_for_round(cfg, sample_counts, round_idx)

        payloads: List[ClientTrainPayload] = []
        step_cursor = round_idx * steps_round

        client_losses: Dict[int, float] = {}
        train_times: Dict[int, float] = {}
        client_timing: Dict[int, dict] = {}
        round_upload_bytes = 0

        for c in clients:
            if cfg.aggregation_method == "flora":
                rank_use = infer_lora_rank(prev_global_lora) if prev_global_lora else ranks[c]
            else:
                rank_use = ranks[c]

            # Build model for this rank on first encounter; reuse afterwards
            model_build_t0 = time.time()
            if rank_use not in models_by_rank:
                print(f"[lang-engine] Build cached model for rank={rank_use}")
                model, _, _ = build_causal_lm_lora(cfg, rank_use, tokenizer=tokenizer)
                # Snapshot the freshly-initialised LoRA weights so we can reset
                # the model for any client that has no federated init state yet
                initial_lora_by_rank[rank_use] = copy.deepcopy(get_lora_state_dict(model))
                _remember_rank(rank_use, model)
            else:
                model = models_by_rank.pop(rank_use)
                _remember_rank(rank_use, model)
            model_build_s = time.time() - model_build_t0

            model = models_by_rank[rank_use]

            # Load this client's init state (federated or clean LoRA init)
            load_t0 = time.time()
            init_state = client_init.get(c)
            if init_state is not None:
                load_lora_state(model, init_state.lora_state)
            else:
                load_lora_state(model, initial_lora_by_rank[rank_use])
            load_lora_s = time.time() - load_t0

            loader = dm.get_client_loader(c, shuffle=True)
            n_samples = sample_counts[c]
            n_batches_est = len(loader)

            client_t0 = time.time()
            out = trainer.train_one_round(model, loader, step_cursor, total_global_steps)
            client_train_s = time.time() - client_t0
            tr = out.get("timing") or {}
            trainer_setup_s = float(tr.get("trainer_setup_s", 0.0))
            train_loop_s = float(tr.get("train_loop_s", 0.0))

            step_cursor += cfg.local_epochs * len(loader)
            client_losses[c] = out["mean_loss"]
            train_times[c] = client_train_s
            client_timing[c] = {
                "model_cache_block_s": round(model_build_s, 3),
                "load_lora_s": round(load_lora_s, 3),
                "train_wall_s": round(client_train_s, 3),
                "trainer_setup_s": round(trainer_setup_s, 3),
                "train_loop_s": round(train_loop_s, 3),
                "client_compute_wall_s": round(load_lora_s + client_train_s, 3),
            }

            print(
                f"  [client {c:2d}] samples={n_samples:5d}  rank={rank_use:2d}  "
                f"batches={n_batches_est:4d}  loss={out['mean_loss']:.4f}  "
                f"model_cache={model_build_s:.1f}s  load_lora={load_lora_s:.2f}s  "
                f"train={client_train_s:.1f}s  (setup={trainer_setup_s:.2f}s  loop={train_loop_s:.1f}s)"
            )

            lora_state = out["lora_state"]
            if cfg.aggregation_method == "hetlora":
                lora_state = prune_lora_state(lora_state, cfg.hetlora_pruning_ratio)

            payloads.append(
                ClientTrainPayload(
                    client_id=c,
                    num_samples=dm.client_num_samples(c),
                    lora_state=lora_state,
                    rank=ranks[c],
                    classifier_state=None,
                )
            )
            round_upload_bytes += compute_comm_bytes(lora_state)
            model = None

        total_train_s = sum(train_times.values())
        avg_loss = sum(client_losses.values()) / max(1, len(client_losses))
        print(f"  [round {round_idx}] avg_loss={avg_loss:.4f}  "
              f"total_train_time={total_train_s:.1f}s  "
              f"per_client_losses={json.dumps({c: round(l, 4) for c, l in client_losses.items()})}")

        # Reclaim unused GPU cache (models stay alive across rounds)
        torch.cuda.empty_cache()

        agg_t0 = time.time()
        template_lora = prev_global_lora if prev_global_lora is not None else payloads[0].lora_state
        agg = aggregator.aggregate(payloads, cfg, template_lora)
        agg_s = time.time() - agg_t0
        print(f"  [round {round_idx}] DEBUG after aggregation")

        dist_t0 = time.time()
        client_init = agg.client_init

        if agg.server_state.lora_state is not None:
            prev_global_lora = agg.server_state.lora_state

        # Compute download bytes from distributed init states
        round_download_bytes = 0
        for cid, init_st in client_init.items():
            round_download_bytes += compute_comm_bytes(init_st.lora_state)
        dist_s = time.time() - dist_t0
        print(f"  [round {round_idx}] DEBUG after distribution")

        cumulative_upload_bytes += round_upload_bytes
        cumulative_download_bytes += round_download_bytes
        print(f"  [round {round_idx}] aggregation_time={agg_s:.2f}s  distribution_time={dist_s:.2f}s")

        # ---- Final-round-only evaluation ----
        print(f"  [round {round_idx}] DEBUG before eval")
        should_eval = round_idx == cfg.num_rounds - 1
        eval_result = {}
        eval_timing = {}
        eval_s = 0.0
        if should_eval:
            if getattr(cfg, "use_vllm", False):
                _release_training_model_cache()
            eval_t0 = time.time()
            if cfg.aggregation_method in ("flora", "hetlora", "raflora"):
                eval_result = evaluator.evaluate_flora_global(
                    lora_state=agg.server_state.lora_state,
                )
            else:
                eval_result = evaluator.evaluate_personalized(
                    per_client_init=agg.client_init,
                    client_ranks=ranks,
                    client_sample_counts=sample_counts,
                )
            eval_timing = {k: eval_result.pop(k) for k in list(eval_result.keys()) if k.startswith("__")}
            eval_s = time.time() - eval_t0
            print(f"  [round {round_idx}] DEBUG after eval")
            print(f"  [round {round_idx}] eval_time={eval_s:.2f}s  eval_result={eval_result}")
        else:
            print(f"  [round {round_idx}] eval skipped (final round only)")

        # ---- Metrics ----
        meta = agg.server_state.metadata
        signal_retention = None
        if payloads and agg.client_init:
            before_pairs = {p.client_id: pairs_from_state(p.lora_state) for p in payloads}
            sample_pairs = next(iter(before_pairs.values()), {})
            retention_q_key = find_q_proj_key(list(sample_pairs.keys()))
            if retention_q_key is not None:
                per_client = {}
                ordered_values = []
                for cid in clients:
                    init_state = agg.client_init.get(cid)
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
        print(f"  [round {round_idx}] DEBUG before row")
        row = {
            "round": round_idx,
            "avg_loss": round(avg_loss, 6),
            "client_losses": {str(c): round(l, 6) for c, l in client_losses.items()},
            "client_timing": {str(c): v for c, v in client_timing.items()},
            "timing": {
                "train_s": round(total_train_s, 1),
                "agg_s": round(agg_s, 2),
                "dist_s": round(dist_s, 2),
                "eval_s": round(eval_s, 2),
                "round_s": round(time.time() - round_t0, 1),
            },
            "aggregation": cfg.aggregation_method,
            "rank_policy": cfg.rank_policy,
            "ranks": {str(c): ranks[c] for c in clients},
            "agg_mode": meta.get("mode", ""),
            "agg_implementation": meta.get("implementation", "paper_faithful"),
            "strict_paper_faithful": meta.get("strict_paper_faithful", True),
            "comm": {
                "upload_bytes": round_upload_bytes,
                "download_bytes": round_download_bytes,
                "round_total_bytes": round_upload_bytes + round_download_bytes,
                "cumulative_upload_bytes": cumulative_upload_bytes,
                "cumulative_download_bytes": cumulative_download_bytes,
                "cumulative_total_bytes": cumulative_upload_bytes + cumulative_download_bytes,
            },
            "eval": {k: round(v, 6) if isinstance(v, float) else v for k, v in eval_result.items()},
            "eval_timing": eval_timing,
            "energy_ratios": meta.get("energy_ratios"),
        }
        if row["energy_ratios"]:
            er = row["energy_ratios"]
            q_key = find_q_proj_key(list(er.keys()))
            if q_key is not None:
                row["energy_ratio_q"] = er[q_key]
            row["energy_ratio_mean"] = sum(float(v) for v in er.values()) / len(er)
        if "sv_snapshot" in meta:
            row["sv_after"] = meta.get("sv_snapshot")
            row["sv_snapshot"] = meta.get("sv_snapshot")
        if "sv_before" in meta:
            row["sv_before"] = meta.get("sv_before")
        if signal_retention is not None:
            row["signal_retention"] = signal_retention
            row["signal_retention_values"] = signal_retention["values"]
            row["signal_retention_mean"] = signal_retention["mean"]

        round_s = time.time() - round_t0
        print(f"  [round {round_idx}] total_round_time={round_s:.1f}s")

        if cfg.log_every_round:
            print(f"  [round {round_idx}] DEBUG before json print")
            print(json.dumps(row, ensure_ascii=False))

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # ---- Checkpoint: save LoRA weights every round ----
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_t0 = time.time()
        if agg.server_state.lora_state is not None:
            # Global model (FLoRA / HETLoRA / raFLoRA): single file
            ckpt_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}_global.pt")
            torch.save(agg.server_state.lora_state, ckpt_path)
        else:
            # Personalized models (FlexLoRA / EGWSA): one file per client
            for cid, init_state in agg.client_init.items():
                ckpt_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}_client_{cid}.pt")
                torch.save(init_state.lora_state, ckpt_path)
        ckpt_s = time.time() - ckpt_t0
        print(f"  [round {round_idx}] checkpoint_saved  path={ckpt_dir}  save_time={ckpt_s:.2f}s")

    print(f"[lang-engine] Done. Results in {out_dir}")
