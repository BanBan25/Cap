"""
Rank policies shared across vision and language pipelines.

Four policies (matching the paper):
  1. fixed     – every client uses the same rank
  2. random    – each client samples a rank from candidate_ranks per round (deterministic)
  3. heuristic – rank assigned by relative data-volume ratio with fixed thresholds
  4. adaptive  – rank assigned by theoretical NTK score with quantile bucketization

Why heuristic ≠ adaptive:
  *adaptive* uses ``quantile_map_to_candidates`` which only cares about the
  rank-order of scores — and since ``theoretical_rank_from_ni`` is strictly
  monotone in N_i, the rank-order is identical to raw sample counts.

  *heuristic* instead uses ``ratio_threshold_map_to_candidates`` which maps
  each client's **relative** sample ratio  ``N_i / max(N)``  into equal-width
  bins over [0, 1].  This means the actual numeric gaps between sample counts
  affect the outcome, not just their ordering.

  Concrete example with candidates=[8,16,32,48,64]:
    sample_counts = {0:100, 1:200, 2:400, 3:800, 4:1600}
    heuristic → {0:8, 1:8, 2:16, 3:32, 4:64}   (ratio thresholds)
    adaptive  → {0:8, 1:16, 2:32, 3:48, 4:64}   (quantile rank-order)
"""
from __future__ import annotations

import math
import random as _stdlib_random
from typing import Dict, List


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def theoretical_rank_from_ni(N_i: int, k: int) -> float:
    """Theoretical optimal rank from NTK analysis (used by *adaptive*)."""
    return (math.sqrt(8.0 * k * N_i + 1.0) - 1.0) / 2.0


def quantile_map_to_candidates(
    scores: Dict[int, float],
    candidates: List[int],
) -> Dict[int, int]:
    """Map per-client scores to candidate ranks via **rank-order** quantile bucketization.

    Only the relative ordering of scores matters; numeric gaps are ignored.
    Clients with identical scores always receive the same rank (tie-safe).
    Used by *adaptive*.
    """
    c_sorted = sorted(set(candidates))
    n_buckets = len(c_sorted)
    unique_scores = sorted(set(scores.values()))
    n_unique = len(unique_scores)
    score_to_rank: Dict[float, int] = {}
    for pos, s in enumerate(unique_scores):
        bucket_idx = min(pos * n_buckets // n_unique, n_buckets - 1)
        score_to_rank[s] = c_sorted[bucket_idx]
    return {cid: score_to_rank[s] for cid, s in scores.items()}


def ratio_threshold_map_to_candidates(
    values: Dict[int, float],
    candidates: List[int],
) -> Dict[int, int]:
    """Map per-client values to candidate ranks via **ratio-based fixed thresholds**.

    Each value is normalised to ``[0, 1]`` as ``v / max(values)``, then the
    unit interval is split into ``len(candidates)`` equal-width bins:
      bin 0: [0, 1/K)   → smallest candidate rank
      bin 1: [1/K, 2/K) → second candidate rank
      ...
      bin K-1: [(K-1)/K, 1] → largest candidate rank

    Unlike ``quantile_map_to_candidates`` this is **value-sensitive**: the
    actual numeric magnitude affects the bucket, not just the rank-order.
    Ties (identical values) always map to the same rank.
    Monotone: larger value → rank no smaller.
    Used by *heuristic*.
    """
    c_sorted = sorted(set(candidates))
    n_buckets = len(c_sorted)
    max_val = max(values.values())
    if max_val <= 0:
        return {cid: c_sorted[0] for cid in values}
    result: Dict[int, int] = {}
    for cid, v in values.items():
        ratio = v / max_val
        bucket_idx = min(int(ratio * n_buckets), n_buckets - 1)
        result[cid] = c_sorted[bucket_idx]
    return result


# ---------------------------------------------------------------------------
# 1. Fixed
# ---------------------------------------------------------------------------

class FixedRankPolicy:
    def ranks_for_round(self, cfg, client_sample_counts: Dict[int, int], round_idx: int) -> Dict[int, int]:
        return {cid: cfg.fixed_rank for cid in client_sample_counts}


# ---------------------------------------------------------------------------
# 2. Random  (deterministic given seed + round_idx + client_id)
# ---------------------------------------------------------------------------

class RandomRankPolicy:
    def ranks_for_round(self, cfg, client_sample_counts: Dict[int, int], round_idx: int) -> Dict[int, int]:
        candidates = sorted(set(cfg.candidate_ranks))
        return {
            cid: _stdlib_random.Random(cfg.seed + round_idx * 1000 + cid).choice(candidates)
            for cid in client_sample_counts
        }


# ---------------------------------------------------------------------------
# 3. Heuristic  (ratio N_i/max(N) → fixed-threshold bins → candidate rank)
# ---------------------------------------------------------------------------

class HeuristicRankPolicy:
    """Assign ranks by relative data-volume ratio with equal-width thresholds.

    ``ratio_i = N_i / max_j(N_j)`` is computed for every client, and the
    [0, 1] interval is split into ``len(candidate_ranks)`` equal bins.
    This is **value-sensitive** — clients clustered at the low end will all
    land in the first bin even if they have distinct sample counts, whereas
    *adaptive* (quantile-based) would spread them evenly across all bins.
    """

    def ranks_for_round(self, cfg, client_sample_counts: Dict[int, int], round_idx: int) -> Dict[int, int]:
        values = {cid: float(n) for cid, n in client_sample_counts.items()}
        return ratio_threshold_map_to_candidates(values, list(cfg.candidate_ranks))


# ---------------------------------------------------------------------------
# 4. Adaptive  (NTK theoretical score → quantile → candidate rank)
# ---------------------------------------------------------------------------

class AdaptiveRankPolicy:
    def ranks_for_round(self, cfg, client_sample_counts: Dict[int, int], round_idx: int) -> Dict[int, int]:
        scores = {
            cid: theoretical_rank_from_ni(int(n), cfg.adaptive_k)
            for cid, n in client_sample_counts.items()
        }
        return quantile_map_to_candidates(scores, list(cfg.candidate_ranks))
