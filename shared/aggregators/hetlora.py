"""
HETLORA aggregator – server-side sparsity-weighted aggregation.

By the time payloads reach this aggregator, each client has already performed
local rank self-pruning (see ``shared.hetlora_pruning``).  The server's job is:

  1. **Detect sparsity** – identify which rank components each client retained
     (non-zero columns in B / rows in A).
  2. **Sparsity-Weighted Aggregation** – for each rank component position i,
     compute a sample-count weighted average *only* over clients that actively
     retained that component.  Pruned (zero) components do not dilute the average.
  3. **Distribution via Truncation** – construct per-client init states by
     truncating the global LoRA to each client's target rank.

This is a *global* aggregation method: the server maintains a single global LoRA
state at max_rank and distributes rank-truncated copies to each client.

Classifier (vision only): sample-count weighted average, broadcast to all clients.
Language tasks pass classifier_state=None and this aggregator handles it gracefully.
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


def _detect_active_mask(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return a bool mask (r,) where True means the component is non-zero."""
    return (torch.norm(B, dim=0) + torch.norm(A, dim=1)) > 0


class HETLORAAggregator:
    """
    HETLORA server-side: sparsity-weighted aggregation of already-pruned
    client LoRA factors, followed by rank-truncated redistribution.

    Clients are expected to have performed self-pruning before upload.
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
        weights = torch.tensor(
            [p.num_samples for p in payloads], dtype=torch.float32, device=dev
        )
        omega = (weights / weights.sum()).tolist()

        pair_ids = list(pairs_from_state(payloads[0].lora_state).keys())
        for p in payloads[1:]:
            if set(pairs_from_state(p.lora_state).keys()) != set(pair_ids):
                raise ValueError("LoRA key mismatch across clients for HETLORA")

        avg_classifier = _weighted_avg_classifier(payloads, omega)

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

        max_rank = max(p.rank for p in payloads)
        ref_dtype = next(iter(per_client_pairs[payloads[0].client_id].values()))[0].dtype

        # ---- Per-layer sparsity-weighted aggregation ----
        global_pairs: Dict[str, tuple] = {}

        for pid in pair_ids:
            out_dim = in_dim = 0
            for p in payloads:
                A_p, B_p = per_client_pairs[p.client_id][pid]
                out_dim = max(out_dim, B_p.shape[0])
                in_dim = max(in_dim, A_p.shape[1])

            B_acc = torch.zeros(out_dim, max_rank, dtype=torch.float32, device=dev)
            A_acc = torch.zeros(max_rank, in_dim, dtype=torch.float32, device=dev)
            w_per_comp = torch.zeros(max_rank, dtype=torch.float32, device=dev)

            for p, w in zip(payloads, omega):
                A_p, B_p = per_client_pairs[p.client_id][pid]
                A_p = A_p.to(dev, dtype=torch.float32)
                B_p = B_p.to(dev, dtype=torch.float32)
                r_k = A_p.shape[0]

                # Detect which components the client retained after self-pruning
                active = _detect_active_mask(A_p, B_p).nonzero(as_tuple=False).squeeze(-1)
                if active.numel() > 0:
                    B_acc[:, active] += w * B_p[:, active]
                    A_acc[active, :] += w * A_p[active, :]
                    w_per_comp[active] += w

            # Re-normalise: each component averaged only over contributing clients
            active_global = w_per_comp > 0
            if active_global.any():
                B_acc[:, active_global] /= w_per_comp[active_global].unsqueeze(0)
                A_acc[active_global, :] /= w_per_comp[active_global].unsqueeze(1)

            global_pairs[pid] = (
                A_acc.to(ref_dtype),
                B_acc.to(ref_dtype),
            )

        global_lora = merge_state_template(template_lora, global_pairs)

        # ── Metrics for logging ──
        layer_delta_ws: Dict[str, torch.Tensor] = {}
        for pid, (A_g, B_g) in global_pairs.items():
            layer_delta_ws[pid] = B_g.float() @ A_g.float()
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

        # ---- Distribution via Truncation: per-client init ----
        client_init: Dict[int, ClientInitState] = {}
        for p in payloads:
            r_k = p.rank
            trunc_pairs: Dict[str, tuple] = {}
            for pid in pair_ids:
                A_g, B_g = global_pairs[pid]
                trunc_pairs[pid] = (A_g[:r_k, :].clone(), B_g[:, :r_k].clone())

            trunc_lora = merge_state_template(
                copy.deepcopy(p.lora_state), trunc_pairs
            )
            client_init[p.client_id] = ClientInitState(
                lora_state=trunc_lora,
                classifier_state=copy.deepcopy(avg_classifier) if avg_classifier else None,
            )

        pruning_ratio = getattr(cfg, "hetlora_pruning_ratio", 0.3)
        server = GlobalServerState(
            lora_state=global_lora,
            classifier_state=avg_classifier,
            metadata={
                "mode": "hetlora_sparse_weighted",
                "implementation": "paper_oriented_baseline",
                "strict_paper_faithful": False,
                "energy_ratios": energy_ratios,
                "sv_snapshot": sv_snapshot,
                "sv_before": sv_before,
                "note": (
                    "HETLORA sparsity-weighted aggregation. Clients perform "
                    "magnitude-based self-pruning before upload; server detects "
                    "active components and re-normalises the weighted average "
                    f"per rank component. pruning_ratio={pruning_ratio}."
                ),
            },
        )
        return AggregationResult(server_state=server, client_init=client_init)
