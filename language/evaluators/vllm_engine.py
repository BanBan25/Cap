"""
vLLM inference engine for accelerated evaluation.

Workflow:
  1. Deep-copy the PeftModel so the original is NOT mutated
  2. Merge LoRA adapters into base weights on the copy
  3. Save merged model to a temp directory, then delete the copy
  4. Load via vLLM LLM for high-throughput batched inference

Provides two interfaces:
  - get_lm_eval_model(): returns an lm-eval compatible VLLM wrapper
    that reuses the *same* vllm.LLM instance (no double-load)
  - batch_generate(): batched text generation via vLLM SamplingParams
"""
from __future__ import annotations

import copy
import gc
import os
import shutil
import tempfile
from typing import List, Optional

import torch
import torch.nn as nn


class VLLMEngine:
    """Manages a vLLM LLM instance backed by a merged LoRA model.

    Key design decisions:
      - The original PeftModel is deep-copied before merge so callers
        can continue using it (e.g. personalized eval with cached models).
      - The caller must offload / delete the HF model from GPU *before*
        calling ``_ensure_llm()`` to avoid OOM.  ``from_model()`` does
        this automatically; ``from_path()`` skips the merge step entirely.
      - ``get_lm_eval_model()`` reuses the *same* ``vllm.LLM`` instance
        instead of creating a second one.
    """

    def __init__(
        self,
        merged_path: str,
        tokenizer_path: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        owns_tmp_dir: bool = False,
        dtype: str = "auto",
    ):
        self._merged_path = merged_path
        self._tokenizer_path = tokenizer_path
        self._tp = tensor_parallel_size
        self._gpu_util = gpu_memory_utilization
        self._max_model_len = max_model_len  # None = let vLLM infer from model config
        self._dtype = dtype
        self._owns_tmp_dir = owns_tmp_dir
        self._lm_eval_model = None  # lm_eval VLLM wrapper — cached, owns the vllm.LLM

    # ----------------------------------------------------------
    # Factory: from an in-memory PeftModel (safe deep-copy + merge)
    # ----------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
    ) -> "VLLMEngine":
        """Create engine by merging LoRA into a *copy* of the model.

        WARNING: the original ``model`` is moved to CPU (side effect) to free
        GPU memory for the deep-copy.  LoRA weights are NOT mutated.
        After saving, the copy is deleted.  Callers should ``del model``
        after this returns if the original is no longer needed.
        """
        from peft import PeftModel

        tmp_dir = tempfile.mkdtemp(prefix="vllm_merged_")
        print(f"[vllm] Merging LoRA (deep-copy) and saving to {tmp_dir} ...")

        # Move original to CPU first to free GPU for the copy
        model.cpu()

        if isinstance(model, PeftModel):
            model_copy = copy.deepcopy(model)
            merged = model_copy.merge_and_unload()
        else:
            merged = copy.deepcopy(model)

        merged.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        # Free both copies from memory
        del merged
        if isinstance(model, PeftModel):
            del model_copy
        gc.collect()
        torch.cuda.empty_cache()
        print("[vllm] Merged model saved. CPU/GPU copies freed.")

        return cls(
            merged_path=tmp_dir,
            tokenizer_path=tmp_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            owns_tmp_dir=True,
            dtype=dtype,
        )

    # ----------------------------------------------------------
    # Factory: from an already-saved merged checkpoint
    # ----------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
    ) -> "VLLMEngine":
        """Create engine from a pre-merged checkpoint on disk."""
        return cls(
            merged_path=model_path,
            tokenizer_path=tokenizer_path or model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            owns_tmp_dir=False,
            dtype=dtype,
        )

    # ----------------------------------------------------------
    # Lazy init: single vLLM instance via lm-eval's VLLM wrapper
    # ----------------------------------------------------------

    def _ensure_lm_eval_model(self):
        """Create the lm-eval VLLM wrapper (which owns the vllm.LLM).

        This is the *only* place a vllm.LLM is created.  Both
        ``get_lm_eval_model()`` and ``batch_generate()`` share it.
        """
        if self._lm_eval_model is not None:
            return

        import multiprocessing
        import os
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_V1", "0")
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        from lm_eval.models.vllm_causallms import VLLM as LMEvalVLLM

        print(f"[vllm] Initializing lm-eval VLLM wrapper from {self._merged_path} ...")
        init_kwargs = dict(
            pretrained=self._merged_path,
            tokenizer=self._tokenizer_path,
            tensor_parallel_size=self._tp,
            gpu_memory_utilization=self._gpu_util,
            trust_remote_code=True,
            dtype=self._dtype,
        )
        if self._max_model_len is not None:
            init_kwargs["max_model_len"] = self._max_model_len
        self._lm_eval_model = LMEvalVLLM(**init_kwargs)
        print("[vllm] Engine ready.")

    # ----------------------------------------------------------
    # Interface 1: lm-eval VLLM wrapper
    # ----------------------------------------------------------

    def get_lm_eval_model(self):
        """Return the lm-eval compatible VLLM model wrapper."""
        self._ensure_lm_eval_model()
        return self._lm_eval_model

    # ----------------------------------------------------------
    # Interface 2: batched text generation
    # ----------------------------------------------------------

    def batch_generate(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate completions for a batch of prompts using vLLM.

        Reuses the vllm.LLM owned by the lm-eval wrapper.
        Returns results in the same order as ``prompts``.
        """
        self._ensure_lm_eval_model()
        from vllm import SamplingParams

        llm = self._lm_eval_model.model  # the vllm.LLM instance

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
        outputs = llm.generate(prompts, params, use_tqdm=True)
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_with_params(
        self,
        prompts: List[str],
        sampling_params_list: List,
    ) -> List:
        """Low-level generate with per-prompt SamplingParams.

        Returns raw vLLM RequestOutput objects (caller handles extraction).
        This is the public API for MT-Bench style per-prompt seed control.
        """
        self._ensure_lm_eval_model()
        llm = self._lm_eval_model.model
        return llm.generate(prompts, sampling_params_list, use_tqdm=True)

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def shutdown(self):
        """Release vLLM resources and optionally clean up temp dir."""
        if self._lm_eval_model is not None:
            del self._lm_eval_model
            self._lm_eval_model = None
            gc.collect()
            torch.cuda.empty_cache()
        if self._owns_tmp_dir and self._merged_path and os.path.isdir(self._merged_path):
            shutil.rmtree(self._merged_path, ignore_errors=True)
            self._merged_path = None
        print("[vllm] Engine shut down.")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False