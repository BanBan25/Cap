"""
FlexLoRA aggregator – paper-faithful SVD-based heterogeneous baseline.

Algorithm (per layer j):
    1. Collect all client LoRA factors:  ΔW_j = B_j @ A_j
    2. Weighted average:  ΔW_bar = Σ_j ω_j ΔW_j   (ω_j = N_j / ΣN_j)
    3. Single SVD:  U, S, V^T = svd(ΔW_bar)
    4. Per client i with target rank r_i, truncate to top-r_i components:
         U_i = U[:, :r_i],  S_i = S[:r_i],  V_i^T = V^T[:r_i, :]
    5. Balanced factorization:
         B_i = U_i @ diag(sqrt(S_i))
         A_i = diag(sqrt(S_i)) @ V_i^T

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


class FlexLoRAAggregator:
    """
    FlexLoRA: SVD-based heterogeneous aggregation baseline.
    LoRA: personalized per client via rank-specific truncated SVD.
    Classifier: sample-count weighted average (vision) or None (language).
    """

    def aggregate(
        self,
        payloads: List[ClientTrainPayload],
        cfg,
        template_lora: dict,
    ) -> AggregationResult:
        del template_lora
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
                raise ValueError("LoRA key mismatch across clients for FlexLoRA")

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

        avg_classifier = _weighted_avg_classifier(payloads, omega)

        client_init: Dict[int, ClientInitState] = {}
        layer_delta_ws: Dict[str, torch.Tensor] = {}

        for pid in pair_ids:
            # Compact SVD: exploit the fact that ΔW_bar is a sum of rank-r matrices.
            # dw_bar = Σ ωᵢ Bᵢ Aᵢ = B̃ · Ã
            #   B̃ = [√ω₁ B₁ | ... | √ωₙ Bₙ]   shape (out_dim × n*r)
            #   Ã = [√ω₁ A₁ ; ... ; √ωₙ Aₙ]   shape (n*r × in_dim)
            # SVD via thin QR:
            #   QR(B̃) = Q_B R_B,  QR(Ã.T) = Q_A R_A
            #   SVD(R_B @ R_A.T) = U_m S V_m.T   ← tiny (n*r × n*r) matrix
            #   U = Q_B @ U_m,  VT = V_m.T @ Q_A.T
            # This avoids materialising the (out_dim × in_dim) full delta matrix
            # and reduces SVD cost from O(out_dim³) to O((n·r)³).
            sqrt_omega = [w ** 0.5 for w in omega]
            B_tilde_cols, A_tilde_rows = [], []
            for q, sq in zip(payloads, sqrt_omega):
                A_j, B_j = per_client_pairs[q.client_id][pid]
                B_tilde_cols.append(sq * B_j.to(dev, dtype=torch.float32))
                A_tilde_rows.append(sq * A_j.to(dev, dtype=torch.float32))

            B_tilde = torch.cat(B_tilde_cols, dim=1)   # (out_dim × n*r)
            A_tilde = torch.cat(A_tilde_rows, dim=0)   # (n*r × in_dim)

            Q_B, R_B = torch.linalg.qr(B_tilde)        # Q_B: (out × n*r), R_B: (n*r × n*r)
            Q_A, R_A = torch.linalg.qr(A_tilde.T)      # Q_A: (in × n*r), R_A: (n*r × n*r)
            U_m, S, V_m_T = torch.linalg.svd(R_B @ R_A.T, full_matrices=False)
            S = S.clamp(min=0.0)
            U  = Q_B @ U_m                              # (out × n*r)
            VT = V_m_T @ Q_A.T                          # (n*r × in)
            # Reconstruct dw_bar for metrics
            layer_delta_ws[pid] = (U * S.unsqueeze(0)) @ VT

            for p in payloads:
                r_i = p.rank
                ref_dtype = per_client_pairs[p.client_id][pid][0].dtype

                U_i = U[:, :r_i]
                S_i_sqrt = torch.sqrt(S[:r_i])
                VT_i = VT[:r_i, :]

                B_i = U_i * S_i_sqrt.unsqueeze(0)       # (out, r_i)
                A_i = S_i_sqrt.unsqueeze(1) * VT_i       # (r_i, in)

                if p.client_id not in client_init:
                    client_init[p.client_id] = {"pairs": {}, "payload": p}
                client_init[p.client_id]["pairs"][pid] = (
                    A_i.to(ref_dtype),
                    B_i.to(ref_dtype),
                )

        # ── Metrics for logging ──
        max_r = max(p.rank for p in payloads)
        energy_ratios = {}
        sv_snapshot = {}
        for pair_id, dw in layer_delta_ws.items():
            energy_ratios[pair_id] = compute_energy_ratio(dw, max_r)
        snap_key = q_key if (q_key and q_key in layer_delta_ws) else (
            next(iter(layer_delta_ws)) if layer_delta_ws else None
        )
        if snap_key is not None:
            sv_snapshot[snap_key] = compute_singular_values(
                layer_delta_ws[snap_key]
            )

        result: Dict[int, ClientInitState] = {}
        for cid, info in client_init.items():
            p = info["payload"]
            lora_init = merge_state_template(
                copy.deepcopy(p.lora_state), info["pairs"]
            )
            result[cid] = ClientInitState(
                lora_state=lora_init,
                classifier_state=copy.deepcopy(avg_classifier) if avg_classifier else None,
            )

        server = GlobalServerState(
            lora_state=None,
            classifier_state=avg_classifier,
            metadata={
                "mode": "flexlora_svd_personalized",
                "implementation": "paper_faithful_baseline",
                "strict_paper_faithful": True,
                "energy_ratios": energy_ratios,
                "sv_snapshot": sv_snapshot,
                "sv_before": sv_before,
                "note": (
                    "Weighted-average update followed by client-rank-specific "
                    "truncated SVD redistribution."
                ),
            },
        )
        return AggregationResult(server_state=server, client_init=result)
