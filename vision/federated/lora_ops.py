"""
Vision lora_ops – re-exports from shared.lora_ops for backward compat.
"""
from shared.lora_ops import (  # noqa: F401
    compute_comm_bytes,
    compute_energy_ratio,
    compute_singular_values,
    delta_w_from_pair,
    find_q_proj_key,
    infer_lora_rank,
    iter_lora_pairs,
    merge_state_template,
    pairs_from_state,
)
