from __future__ import annotations

from typing import Dict, Protocol


class RankPolicy(Protocol):
    def ranks_for_round(
        self,
        cfg,
        client_sample_counts: Dict[int, int],
        round_idx: int,
    ) -> Dict[int, int]:
        ...
