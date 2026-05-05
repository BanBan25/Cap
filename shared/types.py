"""
Federated state types shared by vision and language pipelines.

classifier_state is Optional — vision tasks carry a classifier head,
language tasks (causal LM) typically do not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

LoRAStateDict = Dict[str, torch.Tensor]
ClassifierStateDict = Dict[str, torch.Tensor]


@dataclass
class ClientInitState:
    """Per-client next-round initialization."""
    lora_state: LoRAStateDict
    classifier_state: Optional[ClassifierStateDict] = None


@dataclass
class GlobalServerState:
    lora_state: Optional[LoRAStateDict] = None
    classifier_state: Optional[ClassifierStateDict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientTrainPayload:
    """What a client returns after local training."""
    client_id: int
    num_samples: int
    lora_state: LoRAStateDict
    rank: int
    classifier_state: Optional[ClassifierStateDict] = None


@dataclass
class AggregationResult:
    server_state: GlobalServerState
    client_init: Dict[int, ClientInitState]
