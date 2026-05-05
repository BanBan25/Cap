"""
FLoRA aggregator – engineering approximation.

NOTE: This is NOT a strict reproduction of FLoRA as described in the paper.
The paper's FLoRA performs block-diagonal stacking of heterogeneous-rank LoRA
factors each round.  However, naively stacking every round causes exponential
rank growth (R → R*N each round).

Current engineering compromise (two modes):
  1. **flora_stack** (round 0 or whenever each client still trains at its own
     policy rank): weighted block stacking  B_cat A_cat ≈ Σ ω_i B_i A_i.
  2. **flora_fedavg_same_rank** (subsequent rounds when all clients share the
     same broadcast rank): element-wise weighted average of LoRA tensors at
     the shared rank (equivalent to FedAvg on adapter parameters).

Classifier (vision only): sample-count weighted average, broadcast to all clients.
Language tasks pass classifier_state=None and this aggregator handles it gracefully.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import torch

from shared.lora_ops import (
    compute_energy_ratio,
    compute_singular_values,
    find_q_proj_key,
    infer_lora_rank,
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
    """Sample-weighted average of classifier states. Returns None if no client has one."""
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


class FLoRAAggregator:
    """
    FLoRA-style composition of client LoRA factors (engineering approximation).
    See module docstring for deviation from the strict paper formulation.
    """

    def aggregate(
        self,
        payloads: List[ClientTrainPayload],
        cfg,
        template_lora: dict,
    ) -> AggregationResult:
        if not payloads:
            raise ValueError("empty payloads")

        weights = torch.tensor(
            [p.num_samples for p in payloads], dtype=torch.float32
        )
        weights = weights / weights.sum()
        omega = weights.tolist()

        pair_ids = list(pairs_from_state(payloads[0].lora_state).keys())
        for p in payloads[1:]:
            if set(pairs_from_state(p.lora_state).keys()) != set(pair_ids):
                raise ValueError("LoRA key mismatch across clients for FLoRA")

        per_client_pairs = {p.client_id: pairs_from_state(p.lora_state) for p in payloads}
        q_key = find_q_proj_key(pair_ids)
        sv_before: Dict[str, list] = {}
        if q_key is not None:
            sv_before[q_key] = []
            for p in payloads:
                A0, B0 = per_client_pairs[p.client_id][q_key]
                sv_before[q_key].append(
                    compute_singular_values((B0.float() @ A0.float()).detach())
                )

        avg_classifier = _weighted_avg_classifier(payloads, omega)

        stack_eligible = all(
            infer_lora_rank(p.lora_state) == p.rank for p in payloads
        )

        if stack_eligible:
            merged_pairs: Dict[str, tuple] = {}
            for pid in pair_ids:
                B_blocks: List[torch.Tensor] = []
                A_blocks: List[torch.Tensor] = []
                for p, w in zip(payloads, omega):
                    A, B = per_client_pairs[p.client_id][pid]
                    sw = math.sqrt(float(w))
                    B_blocks.append(B * sw)
                    A_blocks.append(A * sw)
                B_cat = torch.cat(B_blocks, dim=1)
                A_cat = torch.cat(A_blocks, dim=0)
                merged_pairs[pid] = (A_cat, B_cat)
            global_lora = merge_state_template(template_lora, merged_pairs)
            mode = "flora_stack"
        else:
            keys = list(payloads[0].lora_state.keys())
            global_lora = {}
            for k in keys:
                acc = None
                for p, w in zip(payloads, omega):
                    t = p.lora_state[k].float() * w
                    acc = t if acc is None else acc + t
                global_lora[k] = acc
            mode = "flora_fedavg_same_rank"

        # ── Metrics for logging ──
        global_pairs_for_metrics = pairs_from_state(global_lora)
        max_r = max(p.rank for p in payloads)
        layer_delta_ws: Dict[str, torch.Tensor] = {}
        for pid, (A_g, B_g) in global_pairs_for_metrics.items():
            layer_delta_ws[pid] = B_g.float() @ A_g.float()
        energy_ratios: Dict[str, float] = {}
        sv_snapshot: Dict[str, object] = {}
        for pair_id, dw in layer_delta_ws.items():
            energy_ratios[pair_id] = compute_energy_ratio(dw, max_r)
        snap_key = q_key if (q_key and q_key in layer_delta_ws) else (
            next(iter(layer_delta_ws)) if layer_delta_ws else None
        )
        if snap_key is not None:
            sv_snapshot[snap_key] = compute_singular_values(
                layer_delta_ws[snap_key]
            )

        server = GlobalServerState(
            lora_state=global_lora,
            classifier_state=avg_classifier,
            metadata={
                "mode": mode,
                "implementation": "engineering_approximation",
                "strict_paper_faithful": False,
                "energy_ratios": energy_ratios,
                "sv_snapshot": sv_snapshot,
                "sv_before": sv_before,
                "note": (
                    "Round-0 uses weighted block stacking; subsequent rounds "
                    "fall back to FedAvg-style same-rank averaging. This is NOT "
                    "the strict multi-round FLoRA described in the paper."
                ),
            },
        )
        client_init = {
            p.client_id: ClientInitState(
                lora_state=copy.deepcopy(global_lora),
                classifier_state=copy.deepcopy(avg_classifier) if avg_classifier else None,
            )
            for p in payloads
        }
        return AggregationResult(server_state=server, client_init=client_init)
