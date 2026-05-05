"""
EGWSA aggregator – paper-faithful implementation.

Objective (per layer, per target client i with rank r_i):
    max_{U_i, V_i}  Σ_j ω_j ‖U_i^T ΔW_j V_i‖_F^2
    s.t. U_i^T U_i = I,  V_i^T V_i = I

Alternating update:
    M_i(V) = Σ_j ω_j  ΔW_j V V^T ΔW_j^T          (out × out)
    N_i(U) = Σ_j ω_j  ΔW_j^T U U^T ΔW_j           (in  × in)
    U_i ← top-r_i eigenvectors of M_i(V)
    V_i ← top-r_i eigenvectors of N_i(U)

After convergence:
    M_bar = Σ_j ω_j ΔW_j                            (weighted average update)
    C_i   = U_i^T  M_bar  V_i                        (r_i × r_i core)
    Balanced factorization via SVD of C_i:
        C_i = Uc Σc Vc^T
        B_i = U_i Uc Σc^{1/2}      (out × r_i)
        A_i = Σc^{1/2} Vc^T V_i^T  (r_i × in)

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


def _top_eigvecs(mat: torch.Tensor, k: int) -> torch.Tensor:
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(mat)
    order = torch.argsort(evals, descending=True)
    return evecs[:, order[:k]]


def _balanced_factorize(
    U: torch.Tensor, V: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    Uc, Sc, VcT = torch.linalg.svd(C, full_matrices=False)
    Sc_sqrt = torch.sqrt(Sc.clamp(min=0.0))
    B = U @ Uc * Sc_sqrt.unsqueeze(0)
    A = Sc_sqrt.unsqueeze(1) * (VcT @ V.T)
    return A, B


def _weighted_avg_classifier(
    payloads: List[ClientTrainPayload],
    omega: List[float],
) -> Optional[Dict[str, torch.Tensor]]:
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


class EGWSAAggregator:
    """
    Layer-wise EGWSA (paper §3.2) with per-client personalized LoRA initialization.
    LoRA: personalized per client via EGWSA subspace optimization.
    Classifier: sample-count weighted average (vision) or None (language).

    Optimized: per-layer shared quantities (deltas, M_bar, weighted gram)
    are computed once and reused across all target clients.
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

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor(
            [p.num_samples for p in payloads], dtype=torch.float32, device=dev
        )
        omega = (weights / weights.sum()).tolist()

        pair_ids = list(pairs_from_state(payloads[0].lora_state).keys())
        for p in payloads[1:]:
            if set(pairs_from_state(p.lora_state).keys()) != set(pair_ids):
                raise ValueError("LoRA key mismatch across clients for EGWSA")

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

        egwsa_num_iters = getattr(cfg, "egwsa_num_iters", 5)

        # Group clients by rank — the alternating U/V optimisation depends
        # ONLY on the target rank r_i, so same-rank clients share the result.
        unique_ranks = sorted({p.rank for p in payloads})

        # Process one layer at a time to keep GPU memory bounded.
        # layer_rank_results[pid][rank] = (A_star_cpu, B_star_cpu)
        layer_rank_results: Dict[str, Dict[int, tuple]] = {}
        layer_delta_ws: Dict[str, torch.Tensor] = {}
        ref_dtype = per_client_pairs[payloads[0].client_id][pair_ids[0]][0].dtype

        for pid in pair_ids:
            # -- compute deltas & shared quantities for this layer --
            deltas: List[torch.Tensor] = []
            for q in payloads:
                A_j, B_j = per_client_pairs[q.client_id][pid]
                A_j = A_j.to(dev, dtype=torch.float32)
                B_j = B_j.to(dev, dtype=torch.float32)
                deltas.append(B_j @ A_j)

            M_bar = sum(w * dw for w, dw in zip(omega, deltas))

            # D_w[j] = sqrt(omega_j) * delta_j  (for vectorised gram / einsum)
            sqrt_omega = [w ** 0.5 for w in omega]
            D_w = torch.stack([s * dw for s, dw in zip(sqrt_omega, deltas)])

            # G = Σ omega_j * delta_j^T delta_j  (initial V via top eigvecs)
            G = torch.einsum("jki,jkp->ip", D_w, D_w)
            G = 0.5 * (G + G.T)

            # -- solve per unique rank --
            rank_results: Dict[int, tuple] = {}
            for r_i in unique_ranks:
                V = _top_eigvecs(G, r_i)

                for _ in range(egwsa_num_iters):
                    proj_v = D_w @ V                    # (J, out, r_i)
                    Mi = torch.einsum("jok,jpk->op", proj_v, proj_v)
                    U = _top_eigvecs(Mi, r_i)

                    proj_u = D_w.transpose(1, 2) @ U   # (J, in, r_i)
                    Ni = torch.einsum("jik,jpk->ip", proj_u, proj_u)
                    V = _top_eigvecs(Ni, r_i)

                # Final U from converged V
                proj_v = D_w @ V
                Mi_final = torch.einsum("jok,jpk->op", proj_v, proj_v)
                U = _top_eigvecs(Mi_final, r_i)

                C_i = U.T @ M_bar @ V
                A_star, B_star = _balanced_factorize(U, V, C_i)
                rank_results[r_i] = (
                    A_star.to("cpu", dtype=ref_dtype),
                    B_star.to("cpu", dtype=ref_dtype),
                )

            layer_rank_results[pid] = rank_results
            layer_delta_ws[pid] = M_bar.detach().cpu()

            # Free this layer's GPU tensors before next layer
            del deltas, M_bar, D_w, G
            torch.cuda.empty_cache()

        # ── Metrics for logging ──
        max_r = max(p.rank for p in payloads)
        energy_ratios = {}
        sv_snapshot = {}
        for pair_id, dw in layer_delta_ws.items():
            energy_ratios[pair_id] = compute_energy_ratio(dw, max_r)
        # Target q_proj layer for sv_snapshot (after aggregation)
        snap_key = q_key if (q_key and q_key in layer_delta_ws) else (
            next(iter(layer_delta_ws)) if layer_delta_ws else None
        )
        if snap_key is not None:
            sv_snapshot[snap_key] = compute_singular_values(
                layer_delta_ws[snap_key]
            )

        # -- Assemble per-client init states --
        client_init: Dict[int, ClientInitState] = {}

        for p in payloads:
            new_pairs: Dict[str, tuple] = {}
            for pid in pair_ids:
                new_pairs[pid] = layer_rank_results[pid][p.rank]

            lora_init = merge_state_template(
                copy.deepcopy(p.lora_state), new_pairs
            )
            client_init[p.client_id] = ClientInitState(
                lora_state=lora_init,
                classifier_state=copy.deepcopy(avg_classifier) if avg_classifier else None,
            )

        server = GlobalServerState(
            lora_state=None,
            classifier_state=avg_classifier,
            metadata={
                "mode": "egwsa_personalized",
                "implementation": "paper_faithful",
                "strict_paper_faithful": True,
                "energy_ratios": energy_ratios,
                "sv_snapshot": sv_snapshot,
                "sv_before": sv_before,
                "note": (
                    "Layer-wise EGWSA alternating optimization per paper §3.2. "
                    "LoRA initialization is personalized per client; classifier "
                    "is shared via sample-weighted average (vision) or absent (language)."
                ),
            },
        )
        return AggregationResult(server_state=server, client_init=client_init)
