"""
Diagnostic script: compare EGWSAAggregator vs FlexLoRAAggregator output
on the same payloads, per round, per client, per LoRA pair.

Design: Fixed Reference Trajectory
-----------------------------------
Payloads each round are generated from a single real federated trajectory
(FlexLoRA by default). At each round, the same batch of payloads is fed
to both EGWSA and FlexLoRA aggregators, and their outputs are compared.
Only the reference aggregator's client_init is used to initialise the
next round's local training — there is NO averaging or mixing of the two
aggregator outputs.

This means:
  - Every round's payloads come from one genuine, reproducible trajectory.
  - The comparison shows "given identical payloads, how different are the
    two aggregators' outputs?" without any contamination from the other.

Only supports: task_type=vision, dataset=cifar100, model=vit_base.

Usage:
    python tools/compare_egwsa_flexlora.py \
        --model-name google/vit-base-patch16-224 \
        --num-rounds 3 --num-clients 10 --device cuda

Outputs:
    ./outputs/diagnostics/egwsa_vs_flexlora_details.jsonl
    ./outputs/diagnostics/egwsa_vs_flexlora_summary.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.aggregators.egwsa import EGWSAAggregator
from shared.aggregators.flexlora import FlexLoRAAggregator
from shared.lora_ops import pairs_from_state
from shared.seed import set_seed
from shared.rank_policies import (
    FixedRankPolicy,
    RandomRankPolicy,
    HeuristicRankPolicy,
    AdaptiveRankPolicy,
)
from shared.types import ClientTrainPayload
from vision.config import VisionFedConfig
from vision.data import CIFAR100FedDataModule
from vision.models.vit_lora import build_vit_lora, load_federated_state
from vision.trainer import VisionClientTrainer


REFERENCE_TRAJECTORY = "flexlora"

_RANK_POLICIES = {
    "fixed": FixedRankPolicy,
    "random": RandomRankPolicy,
    "heuristic": HeuristicRankPolicy,
    "adaptive": AdaptiveRankPolicy,
}


def _select_clients(cfg: VisionFedConfig):
    n = max(1, int(round(cfg.num_clients * cfg.participation_rate)))
    return list(range(n))


def _steps_per_round(dm, clients, cfg):
    total = 0
    for c in clients:
        total += cfg.local_epochs * len(dm.get_client_loader(c))
    return total


def compare_one_round(
    payloads,
    cfg,
    template_lora,
    round_idx: int,
    detail_f,
    summary_f,
):
    egwsa = EGWSAAggregator()
    flexlora = FlexLoRAAggregator()

    payloads_for_egwsa = copy.deepcopy(payloads)
    payloads_for_flex = copy.deepcopy(payloads)
    template_e = copy.deepcopy(template_lora)
    template_f = copy.deepcopy(template_lora)

    agg_e = egwsa.aggregate(payloads_for_egwsa, cfg, template_e)
    agg_f = flexlora.aggregate(payloads_for_flex, cfg, template_f)

    all_rel = []
    all_a_fro = []
    all_b_fro = []
    num_pairs = 0

    cids = sorted(set(agg_e.client_init.keys()) & set(agg_f.client_init.keys()))
    for cid in cids:
        pairs_e = pairs_from_state(agg_e.client_init[cid].lora_state)
        pairs_f = pairs_from_state(agg_f.client_init[cid].lora_state)
        pids = sorted(set(pairs_e.keys()) & set(pairs_f.keys()))
        if num_pairs == 0:
            num_pairs = len(pids)

        for pid in pids:
            A_e, B_e = pairs_e[pid]
            A_f, B_f = pairs_f[pid]
            A_e = A_e.float()
            B_e = B_e.float()
            A_f = A_f.float()
            B_f = B_f.float()

            rank = int(A_e.shape[0])
            a_diff = torch.norm(A_e - A_f).item()
            b_diff = torch.norm(B_e - B_f).item()

            delta_e = B_e @ A_e
            delta_f = B_f @ A_f
            delta_diff = torch.norm(delta_e - delta_f).item()
            delta_flex_fro = torch.norm(delta_f).item()
            rel = delta_diff / (delta_flex_fro + 1e-12)

            row = {
                "round": round_idx,
                "client_id": cid,
                "pair_id": pid,
                "rank": rank,
                "A_diff_fro": a_diff,
                "B_diff_fro": b_diff,
                "delta_diff_fro": delta_diff,
                "delta_flex_fro": delta_flex_fro,
                "rel_delta_diff": rel,
            }
            detail_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_rel.append(rel)
            all_a_fro.append(a_diff)
            all_b_fro.append(b_diff)

    mean_rel = sum(all_rel) / max(len(all_rel), 1)
    max_rel = max(all_rel) if all_rel else 0.0
    mean_a = sum(all_a_fro) / max(len(all_a_fro), 1)
    mean_b = sum(all_b_fro) / max(len(all_b_fro), 1)

    summary = {
        "round": round_idx,
        "mean_rel_delta_diff": mean_rel,
        "max_rel_delta_diff": max_rel,
        "mean_A_diff_fro": mean_a,
        "mean_B_diff_fro": mean_b,
        "num_clients": len(cids),
        "num_pairs": num_pairs,
    }
    summary_f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(
        f"round {round_idx}: "
        f"mean_rel_delta_diff={mean_rel:.6e}, "
        f"max_rel_delta_diff={max_rel:.6e}"
    )
    return agg_e, agg_f


def main():
    p = argparse.ArgumentParser(description="EGWSA vs FlexLoRA diagnostic comparison")

    p.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./outputs/diagnostics")
    p.add_argument("--rank-policy", choices=list(_RANK_POLICIES.keys()), default="fixed")
    p.add_argument("--fixed-rank", type=int, default=16)
    p.add_argument("--num-rounds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--participation-rate", type=float, default=1.0)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--egwsa-num-iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()
    set_seed(args.seed)

    cfg = VisionFedConfig(
        dataset="cifar100",
        model_name=args.model_name,
        num_clients=args.num_clients,
        participation_rate=args.participation_rate,
        dirichlet_alpha=args.dirichlet_alpha,
        num_rounds=args.num_rounds,
        local_epochs=1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fixed_rank=args.fixed_rank,
        rank_policy=args.rank_policy,
        aggregation_method="flexlora",
        egwsa_num_iters=args.egwsa_num_iters,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "egwsa_vs_flexlora_details.jsonl")
    summary_path = os.path.join(args.output_dir, "egwsa_vs_flexlora_summary.jsonl")

    dm = CIFAR100FedDataModule(cfg)
    trainer = VisionClientTrainer(cfg)
    rank_policy = _RANK_POLICIES[args.rank_policy]()

    clients = _select_clients(cfg)
    sample_counts = {cid: dm.client_num_samples(cid) for cid in clients}
    steps_round = _steps_per_round(dm, clients, cfg)
    total_global_steps = max(1, cfg.num_rounds * steps_round)

    # Only one trajectory is maintained: the reference (FlexLoRA).
    client_init_ref = {}

    print(f"[diag] reference_trajectory={REFERENCE_TRAJECTORY}")
    print(f"[diag] model={cfg.model_name} rank_policy={cfg.rank_policy} "
          f"fixed_rank={cfg.fixed_rank} clients={cfg.num_clients} "
          f"rounds={cfg.num_rounds} egwsa_iters={cfg.egwsa_num_iters}")
    print(f"[diag] detail -> {detail_path}")
    print(f"[diag] summary -> {summary_path}")
    print()

    with open(detail_path, "w", encoding="utf-8") as df, \
         open(summary_path, "w", encoding="utf-8") as sf:

        for round_idx in range(cfg.num_rounds):
            ranks = rank_policy.ranks_for_round(cfg, sample_counts, round_idx)
            print(f"--- round {round_idx} | ranks={dict(sorted(ranks.items()))} ---")

            payloads = []
            step_cursor = round_idx * steps_round

            for c in clients:
                rank_use = ranks[c]
                model = build_vit_lora(cfg, rank_use)

                init_ref = client_init_ref.get(c)
                if init_ref is not None:
                    load_federated_state(
                        model, init_ref.lora_state, init_ref.classifier_state
                    )

                loader = dm.get_client_loader(c, shuffle=True)
                out = trainer.train_one_round(model, loader, step_cursor, total_global_steps)
                step_cursor += cfg.local_epochs * len(loader)

                payloads.append(
                    ClientTrainPayload(
                        client_id=c,
                        num_samples=dm.client_num_samples(c),
                        lora_state=out["lora_state"],
                        rank=ranks[c],
                        classifier_state=out["classifier_state"],
                    )
                )
                del model
                torch.cuda.empty_cache()

            template_lora = payloads[0].lora_state

            agg_e, agg_f = compare_one_round(
                payloads, cfg, template_lora, round_idx, df, sf,
            )
            df.flush()
            sf.flush()

            # Advance only along the reference trajectory (FlexLoRA).
            client_init_ref = agg_f.client_init

    print()
    print(f"[diag] Done. Results written to:")
    print(f"  {detail_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
