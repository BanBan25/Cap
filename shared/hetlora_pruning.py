"""
HETLORA client-side self-pruning utilities.

Called by vision/language engines after local training, before constructing
ClientTrainPayload.  The pruned LoRA state is what gets uploaded to the server.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from shared.lora_ops import pairs_from_state, merge_state_template
from shared.types import LoRAStateDict


def prune_lora_pair(
    A: torch.Tensor,
    B: torch.Tensor,
    pruning_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Magnitude-based rank self-pruning for a single LoRA pair.

    Importance per rank component:  score_i = ||B[:, i]||_2 * ||A[i, :]||_2
    Keeps top ceil((1 - pruning_ratio) * r) components; zeros out the rest.

    Returns (A_pruned, B_pruned).
    """
    r = A.shape[0]
    scores = torch.norm(B, dim=0) * torch.norm(A, dim=1)

    n_keep = max(1, min(int(math.ceil((1.0 - pruning_ratio) * r)), r))
    _, top_idx = torch.topk(scores, n_keep)
    mask = torch.zeros(r, dtype=torch.bool, device=A.device)
    mask[top_idx] = True

    pruned = ~mask
    A = A.clone()
    B = B.clone()
    A[pruned] = 0.0
    B[:, pruned] = 0.0
    return A, B


def prune_lora_state(state: LoRAStateDict, pruning_ratio: float) -> LoRAStateDict:
    """Apply HETLORA self-pruning to a full LoRA state dict (in-place safe).

    Each (A, B) pair is independently pruned by magnitude-based scoring.
    Returns a new state dict with pruned tensors.
    """
    pairs = pairs_from_state(state)
    updated: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for pid, (A, B) in pairs.items():
        A_pr, B_pr = prune_lora_pair(A, B, pruning_ratio)
        updated[pid] = (A_pr, B_pr)
    return merge_state_template(dict(state), updated)
