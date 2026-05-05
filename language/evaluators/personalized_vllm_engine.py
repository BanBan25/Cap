"""
PersonalizedVLLMEngine — single base-model vLLM instance with dynamic LoRA
adapter switching via LoRARequest.

Usage:
    engine = PersonalizedVLLMEngine(cfg)
    engine.start()
    for cid, lora_state in clients.items():
        adapter_dir = engine.prepare_client(cid, lora_state)
        lm = engine.get_lm_eval_model(cid)   # lm-eval compatible
        # ... run benchmarks with lm ...
    engine.shutdown()
"""
from __future__ import annotations

import gc
import os
from typing import Dict, List, Optional

import torch

from paper_config import PaperFedConfig
from shared.lora_ops import infer_lora_rank
from shared.types import LoRAStateDict

from language.evaluators.client_adapter_cache import ClientAdapterCache


class PersonalizedVLLMEngine:
    """vLLM engine with enable_lora for per-client adapter switching."""

    def __init__(
        self,
        cfg: PaperFedConfig,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        max_lora_rank: int = 64,
    ):
        self._cfg = cfg
        self._tp = tensor_parallel_size
        self._gpu_util = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._max_lora_rank = max_lora_rank
        self._llm = None          # vllm.LLM
        self._adapter_cache = ClientAdapterCache(
            base_model_name=cfg.model_name,
            target_modules=list(cfg.lora_target_modules),
        )
        # placeholder
        self._lm_eval_models: Dict[int, object] = {}

    def start(self) -> None:
        """Initialize the vLLM LLM with enable_lora=True."""
        import multiprocessing
        import os
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_V1", "0")
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        from vllm import LLM
        init_kwargs = dict(
            model=self._cfg.model_name,
            enable_lora=True,
            max_lora_rank=self._max_lora_rank,
            tensor_parallel_size=self._tp,
            gpu_memory_utilization=self._gpu_util,
            trust_remote_code=True,
            dtype=self._dtype,
            max_num_seqs=32,
            max_num_batched_tokens=4096,
            enforce_eager=True,
        )
        if self._max_model_len is not None:
            init_kwargs["max_model_len"] = self._max_model_len
        print(f"[personalized-vllm] Starting base LLM with enable_lora=True ...")
        self._llm = LLM(**init_kwargs)
        print("[personalized-vllm] Base LLM ready.")

    def prepare_client(self, client_id: int, lora_state: LoRAStateDict) -> str:
        """Convert client lora_state to adapter dir. Returns path."""
        return self._adapter_cache.prepare(client_id, lora_state)

    def _make_lora_request(self, client_id: int) -> "LoRARequest":
        from vllm.lora.request import LoRARequest
        adapter_dir = self._adapter_cache.get_dir(client_id)
        if adapter_dir is None:
            raise ValueError(f"Client {client_id} adapter not prepared.")
        return LoRARequest(
            lora_name=f"client_{client_id}",
            lora_int_id=client_id + 1,
            lora_path=adapter_dir,
        )

    def get_lm_eval_model(self, client_id: int):
        """Return a lm-eval compatible wrapper bound to this client's LoRA.

        Uses create_personalized_lm_eval_model which injects the shared
        vllm.LLM + per-client LoRARequest into lm-eval's VLLM class.
        """
        if client_id in self._lm_eval_models:
            # Update the LoRA request in case adapter was re-prepared
            self._lm_eval_models[client_id].lora_request = self._make_lora_request(client_id)
            return self._lm_eval_models[client_id]
        from language.evaluators.personalized_lm_eval_model import (
            create_personalized_lm_eval_model,
        )
        lora_req = self._make_lora_request(client_id)
        wrapper = create_personalized_lm_eval_model(self._llm, lora_req)
        self._lm_eval_models[client_id] = wrapper
        return wrapper

    def batch_generate(
        self,
        client_id: int,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Batched text generation for a specific client's LoRA."""
        from vllm import SamplingParams
        kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        if stop:
            kwargs["stop"] = stop
        if stop_token_ids:
            kwargs["stop_token_ids"] = stop_token_ids
        params = SamplingParams(**kwargs)
        lora_req = self._make_lora_request(client_id)
        outputs = self._llm.generate(
            prompts, params, lora_request=lora_req, use_tqdm=True,
        )
        return [o.outputs[0].text.strip() for o in outputs]

    def shutdown(self) -> None:
        """Release vLLM resources and clean up adapter cache."""
        self._lm_eval_models.clear()
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()
        self._adapter_cache.cleanup()
        print("[personalized-vllm] Engine shut down.")
