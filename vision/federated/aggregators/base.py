from __future__ import annotations

from typing import List, Protocol

from shared.types import AggregationResult, ClientTrainPayload


class Aggregator(Protocol):
    def aggregate(
        self,
        payloads: List[ClientTrainPayload],
        cfg,
        template_lora: dict,
    ) -> AggregationResult:
        ...
