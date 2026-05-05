"""
raFLoRA aggregator – rank-partitioned aggregation to prevent rank collapse.

Reference: "raFLoRA: Preventing Rank Collapse in Federated Low-Rank Adaptation
with Client Heterogeneity", Algorithm 1 / Eq.(8).

Core idea:
  Unlike FedAvg-style LoRA aggregation that uses a rank-agnostic uniform weight
  for all rank components, raFLoRA partitions rank dimensions by the set of
  unique client ranks.  Each partition [l:h] is aggregated *only* over clients
  whose local rank r_k >= h (the effective contributors), preventing high-rank
  singular directions from being systematically diluted by zero-padded columns.

Algorithm (per LoRA layer j):
  1. Compute boundaries R = sorted(cfg.candidate_ranks)   ← GLOBAL rank levels, not payload-derived
     Partitions: (0, R[0]), (R[0], R[1]), ..., (R[-2], R[-1])
  2. For each partition (l, h):
       C_h = {k | r_k >= h}
       N_h = sum_{k in C_h} n_k
       ΔW_h = sum_{k in C_h} (n_k / N_h) * B_k[:, l:h] @ A_k[l:h, :]
  3. ΔW_g = sum_h ΔW_h          (full-rank d×n update)
  4. U, S, V^T = svd(ΔW_g),  truncate to r_max
       B_g = U[:, :r_max] * sqrt(S[:r_max])
       A_g = sqrt(S[:r_max]) * V^T[:r_max, :]
  5. Distribute to client k:  B_g[:, :r_k],  A_g[:r_k, :]

Server state:
  - Maintains global LoRA at max_rank (B_g, A_g).
  - Evaluation uses global LoRA (same branch as FLoRA / HETLORA).

Classifier (vision only): sample-count weighted average, unchanged.
Language tasks pass classifier_state=None and are handled gracefully.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch

from shared.lora_ops import (
    compute_energy_ratio,
    compute_singular_values,
    find_q_proj_key,
    merge_state_template,
    pairs_from_state,
)
from shared.types import (
    AggregationResult,
    ClassifierStateDict,
    ClientInitState,
    ClientTrainPayload,
    GlobalServerState,
)


def _weighted_avg_classifier(
    payloads: List[ClientTrainPayload],
    omega: List[float],
) -> Optional[ClassifierStateDict]:
    has_cls = [p for p in payloads if p.classifier_state is not None]
    if not has_cls:
        return None
    ref_keys = list(has_cls[0].classifier_state.keys())
    avg: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        acc = None
        for p, w in zip(payloads, omega):
            if p.classifier_state is None:
                continue
            t = p.classifier_state[k].float() * w
            acc = t if acc is None else acc + t
        avg[k] = acc
    return avg


class raFLoRAAggregator:
    """
    raFLoRA: rank-partitioned aggregation.

    Each rank partition [l:h] is aggregated only over clients with r_k >= h,
    preventing rank collapse caused by zero-padding dilution.
    """

    def aggregate(
        self,
        payloads: List[ClientTrainPayload],
        cfg,
        template_lora: dict,
    ) -> AggregationResult:
        if not payloads:
            raise ValueError("empty payloads")

        dev = torch.device("cpu")

        # ---- Sample weights (global, for classifier + fallback) ----
        ns = torch.tensor([p.num_samples for p in payloads], dtype=torch.float32, device=dev)
        omega = (ns / ns.sum()).tolist()

        # ---- Validate LoRA key consistency ----
        pair_ids = list(pairs_from_state(payloads[0].lora_state).keys())
        for p in payloads[1:]:
            if set(pairs_from_state(p.lora_state).keys()) != set(pair_ids):
                raise ValueError("LoRA key mismatch across clients for raFLoRA")

        # ---- Pre-extract pairs per client ----
        per_client_pairs: Dict[int, Dict[str, tuple]] = {
            p.client_id: pairs_from_state(p.lora_state) for p in payloads
        }

        # ── Before-aggregation SVD snapshot (for Fig.8 violin) ──
        q_key = find_q_proj_key(pair_ids)
        sv_before: Dict[str, list] = {}
        if q_key is not None:
            client_svs = []
            for p in payloads:
                A, B = per_client_pairs[p.client_id][q_key]
                dw = (B.float() @ A.float()).detach()
                client_svs.append(compute_singular_values(dw))
            sv_before[q_key] = client_svs

        # ---- Classifier (vision) ----
        avg_classifier = _weighted_avg_classifier(payloads, omega)

        # ---- Rank metadata ----
        # r_max and partition boundaries MUST come from the global rank level set
        # (cfg.candidate_ranks), NOT from the currently participating payloads.
        # Under partial participation, high-rank clients may be absent for a round;
        # deriving r_max from payloads would silently shrink the global LoRA and
        # permanently drop high-rank singular directions from the server state.
        max_rank = max(cfg.candidate_ranks)
        # Partition boundaries: sorted global rank levels
        boundaries: List[int] = sorted(set(cfg.candidate_ranks))
        # Build partition list: (low, high) pairs
        partitions: List[tuple] = []
        prev = 0
        for h in boundaries:
            if h > prev:
                partitions.append((prev, h))
            prev = h

        # ---- Per-layer rank-partitioned aggregation ----
        global_pairs: Dict[str, tuple] = {}
        layer_delta_ws: Dict[str, torch.Tensor] = {}

        for pid in pair_ids:
            # Infer dimensions from payloads
            out_dim = in_dim = 0
            for p in payloads:
                A_p, B_p = per_client_pairs[p.client_id][pid]
                out_dim = max(out_dim, B_p.shape[0])
                in_dim = max(in_dim, A_p.shape[1])

            ref_dtype = per_client_pairs[payloads[0].client_id][pid][0].dtype

            # Accumulate partitioned full-matrix update
            dW_g = torch.zeros(out_dim, in_dim, dtype=torch.float32, device=dev)

            for (l, h) in partitions:
                # Effective contributors: clients with r_k >= h
                contrib = [(p, p.num_samples) for p in payloads if p.rank >= h]
                if not contrib:
                    continue

                N_h = float(sum(n for _, n in contrib))

                dW_h = torch.zeros(out_dim, in_dim, dtype=torch.float32, device=dev)
                for p, n_k in contrib:
                    A_k, B_k = per_client_pairs[p.client_id][pid]
                    A_k = A_k.to(dev, dtype=torch.float32)
                    B_k = B_k.to(dev, dtype=torch.float32)
                    # Slice the [l:h] rank partition
                    dW_h += (n_k / N_h) * (B_k[:, l:h] @ A_k[l:h, :])

                dW_g += dW_h

            # ---- SVD → global LoRA at max_rank ----
            layer_delta_ws[pid] = dW_g.clone()
            U, S, VT = torch.linalg.svd(dW_g, full_matrices=False)
            S = S.clamp(min=0.0)

            r = min(max_rank, S.shape[0])
            S_sqrt = torch.sqrt(S[:r])
            B_g = U[:, :r] * S_sqrt.unsqueeze(0)          # (out_dim, r)
            A_g = S_sqrt.unsqueeze(1) * VT[:r, :]          # (r, in_dim)

            # Pad to max_rank if SVD returned fewer singular values
            if r < max_rank:
                pad_B = torch.zeros(out_dim, max_rank - r, dtype=torch.float32, device=dev)
                pad_A = torch.zeros(max_rank - r, in_dim, dtype=torch.float32, device=dev)
                B_g = torch.cat([B_g, pad_B], dim=1)
                A_g = torch.cat([A_g, pad_A], dim=0)

            global_pairs[pid] = (A_g.to(ref_dtype), B_g.to(ref_dtype))

        # ── Metrics for logging ──
        energy_ratios = {}
        sv_snapshot = {}
        for pair_id, dw in layer_delta_ws.items():
            energy_ratios[pair_id] = compute_energy_ratio(dw, max_rank)
        snap_key = q_key if (q_key and q_key in layer_delta_ws) else (
            next(iter(layer_delta_ws)) if layer_delta_ws else None
        )
        if snap_key is not None:
            sv_snapshot[snap_key] = compute_singular_values(
                layer_delta_ws[snap_key]
            )

        # ---- Build global server LoRA state (at max_rank) ----
        global_lora = merge_state_template(template_lora, global_pairs)

        # ---- Distribution via truncation: per-client init ----
        client_init: Dict[int, ClientInitState] = {}
        for p in payloads:
            r_k = p.rank
            trunc_pairs: Dict[str, tuple] = {}
            for pid in pair_ids:
                A_g, B_g = global_pairs[pid]
                trunc_pairs[pid] = (A_g[:r_k, :].clone(), B_g[:, :r_k].clone())

            trunc_lora = merge_state_template(copy.deepcopy(p.lora_state), trunc_pairs)
            client_init[p.client_id] = ClientInitState(
                lora_state=trunc_lora,
                classifier_state=copy.deepcopy(avg_classifier) if avg_classifier else None,
            )

        server = GlobalServerState(
            lora_state=global_lora,
            classifier_state=avg_classifier,
            metadata={
                "mode": "raflora_rank_partitioned",
                "implementation": "paper_oriented_baseline",
                "strict_paper_faithful": False,
                "note": (
                    "raFLoRA rank-partitioned aggregation: each rank partition is "
                    "aggregated only over effective contributors, then summed and "
                    "factorized by SVD.  rank_policy / evaluator / data pipeline "
                    "adapted to the Flora unified framework."
                ),
                "partitions": str(partitions),
                "max_rank": max_rank,
                "energy_ratios": energy_ratios,
                "sv_snapshot": sv_snapshot,
                "sv_before": sv_before,
            },
        )
        return AggregationResult(server_state=server, client_init=client_init)
