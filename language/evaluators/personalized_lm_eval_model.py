"""
lm-eval compatible wrapper that reuses an existing vLLM LLM instance and
injects a per-client LoRARequest.

We patch lm-eval's module-level ``LLM`` symbol instead of mutating
``vllm.LLM`` itself. This avoids polluting the class object across rounds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest


def create_personalized_lm_eval_model(
    llm: "LLM",
    lora_request: "LoRARequest",
):
    """Create an lm-eval VLLM wrapper reusing an existing vllm.LLM."""
    import lm_eval.models.vllm_causallms as vllm_causallms

    LMEvalVLLM = vllm_causallms.VLLM
    model_name = llm.llm_engine.model_config.model

    orig_llm_ctor = vllm_causallms.LLM

    def _patched_llm_ctor(*args, **kwargs):
        return llm

    vllm_causallms.LLM = _patched_llm_ctor
    try:
        wrapper = LMEvalVLLM(
            pretrained=model_name,
            trust_remote_code=True,
            dtype="auto",
        )
    finally:
        vllm_causallms.LLM = orig_llm_ctor

    wrapper.model = llm
    wrapper.lora_request = lora_request
    wrapper.enable_thinking = False
    return wrapper
