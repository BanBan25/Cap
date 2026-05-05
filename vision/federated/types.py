"""
Vision federated types – re-exports from shared.types for backward compat.

NOTE: classifier_state is now Optional everywhere to support language tasks
(which have no classifier head). Vision code continues passing it as non-None.
"""
from shared.types import (  # noqa: F401
    AggregationResult,
    ClassifierStateDict,
    ClientInitState,
    ClientTrainPayload,
    GlobalServerState,
    LoRAStateDict,
)
