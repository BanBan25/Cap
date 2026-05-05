"""
LoRA state dict manipulation utilities shared across vision and language.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch

from shared.types import LoRAStateDict

_PAIR_RE = re.compile(
    r"^(?P<prefix>.*)\.lora_(?P<side>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)


def iter_lora_pairs(
    state: LoRAStateDict,
) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """
    Returns list of (pair_id, A, B):
        A shape (r, in_features)   -- lora_A.weight
        B shape (out_features, r)  -- lora_B.weight
    """
    by_pair: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        m = _PAIR_RE.match(k)
        if not m:
            continue
        prefix = m.group("prefix")
        side = m.group("side")
        adapter = m.group("adapter") or ""
        pair_id = f"{prefix}.__adapter__.{adapter}"
        bucket = by_pair.setdefault(pair_id, {})
        bucket[side] = v
    out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
    for pair_id, sides in sorted(by_pair.items()):
        if "A" not in sides or "B" not in sides:
            raise ValueError(f"Incomplete LoRA pair for {pair_id}: keys={list(sides)}")
        out.append((pair_id, sides["A"], sides["B"]))
    return out


def delta_w_from_pair(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return B @ A


def pairs_from_state(state: LoRAStateDict) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    d: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for pair_id, A, B in iter_lora_pairs(state):
        d[pair_id] = (A, B)
    return d


def infer_lora_rank(state: LoRAStateDict) -> int:
    pairs = pairs_from_state(state)
    if not pairs:
        sample_keys = list(state.keys())[:5]
        raise ValueError(
            f"No LoRA pairs parsed from state_dict. "
            f"Check key format against _PAIR_RE. "
            f"Sample keys: {sample_keys}"
        )
    A, _B = next(iter(pairs.values()))
    return int(A.shape[0])


def merge_state_template(
    template: LoRAStateDict,
    pair_updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> LoRAStateDict:
    new_state: LoRAStateDict = dict(template)
    for pair_id, (A, B) in pair_updates.items():
        adapter = pair_id.split(".__adapter__.")[-1]
        prefix = pair_id.split(".__adapter__.")[0]
        if adapter:
            ka = f"{prefix}.lora_A.{adapter}.weight"
            kb = f"{prefix}.lora_B.{adapter}.weight"
        else:
            ka = f"{prefix}.lora_A.weight"
            kb = f"{prefix}.lora_B.weight"
        if ka not in new_state or kb not in new_state:
            raise KeyError(f"Template missing keys for {pair_id}")
        new_state[ka] = A.contiguous()
        new_state[kb] = B.contiguous()
    return new_state


# ── Metrics utilities ────────────────────────────────────────────────

def compute_comm_bytes(state: LoRAStateDict) -> int:
    """Total bytes to transmit a LoRA state dict (upload or download)."""
    total = 0
    for v in state.values():
        total += v.numel() * v.element_size()
    return total


def compute_energy_ratio(delta_w: torch.Tensor, rank: int) -> float:
    """Higher-rank energy ratio: fraction of Frobenius energy beyond top-`rank` singular values.

    Returns a float in [0, 1].  A value near 0 means the top-`rank` components
    capture almost all energy (no rank collapse concern).
    """
    S = torch.linalg.svdvals(delta_w.float())
    total = (S ** 2).sum().item()
    if total < 1e-12:
        return 0.0
    top_k = (S[:rank] ** 2).sum().item()
    return 1.0 - top_k / total


def compute_singular_values(delta_w: torch.Tensor) -> List[float]:
    """Return all singular values of delta_w as a plain Python list."""
    S = torch.linalg.svdvals(delta_w.float())
    return S.tolist()


def compute_subspace_retention(reference_delta: torch.Tensor, redistributed_delta: torch.Tensor) -> float:
    """Measure how much redistributed signal stays in the reference client's subspace.

    The metric is the fraction of redistributed Frobenius energy that lies in the
    left/right singular subspace spanned by the reference update:

        || U_ref^T * redistributed * V_ref ||_F^2 / || redistributed ||_F^2

    This stays in [0, 1] and is suitable for per-client "signal retention"
    comparisons after redistribution.
    """
    # This metric is only for logging, so force both operands onto CPU to avoid
    # cross-device mismatches between uploaded client payloads and redistributed
    # init states.
    ref = reference_delta.detach().to("cpu", dtype=torch.float32)
    red = redistributed_delta.detach().to("cpu", dtype=torch.float32)

    red_energy = torch.linalg.norm(red, ord="fro").item() ** 2
    if red_energy < 1e-12:
        return 0.0

    U, S, Vh = torch.linalg.svd(ref, full_matrices=False)
    active = S > 1e-8
    if not torch.any(active):
        return 0.0

    U_ref = U[:, active]
    V_ref = Vh[active, :].transpose(0, 1)
    projected = U_ref.transpose(0, 1) @ red @ V_ref
    proj_energy = torch.linalg.norm(projected, ord="fro").item() ** 2
    ratio = proj_energy / red_energy
    return max(0.0, min(1.0, float(ratio)))


def find_q_proj_key(keys) -> Optional[str]:
    """Find the pair_id that corresponds to the Q-projection layer."""
    for k in keys:
        if "q_proj" in k or "query" in k or ".q." in k:
            return k
    return next(iter(keys)) if keys else None
