"""
ClientAdapterCache — converts per-client lora_state dicts into PEFT adapter
directories that vLLM LoRARequest can consume.

Each adapter directory contains:
  - adapter_config.json   (PEFT LoraConfig metadata)
  - adapter_model.safetensors  (LoRA A/B weights)

Directories are cached under a single temp root so repeated evaluations of the
same client (same rank) can reuse the directory after overwriting weights.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional

import torch

from shared.lora_ops import infer_lora_rank
from shared.types import LoRAStateDict


class ClientAdapterCache:
    """Manages a temp directory tree of PEFT adapter dirs for vLLM."""

    def __init__(
        self,
        base_model_name: str,
        target_modules: List[str],
        cache_root: Optional[str] = None,
    ):
        self._base_model_name = base_model_name
        self._target_modules = target_modules
        self._root = cache_root or tempfile.mkdtemp(prefix="vllm_adapters_")
        self._owns_root = cache_root is None
        self._dirs: Dict[int, str] = {}

    def _write_adapter_config(self, adapter_dir: str, rank: int) -> None:
        """Write adapter_config.json compatible with PEFT / vLLM."""
        config = {
            "peft_type": "LORA",
            "base_model_name_or_path": self._base_model_name,
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "target_modules": self._target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "fan_in_fan_out": False,
        }
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Convert peft get_peft_model_state_dict keys to HF adapter file keys.

        peft state dict keys look like:
          base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        vLLM / PEFT adapter files expect the same format, so we keep as-is
        but ensure the 'base_model.model.' prefix is present.
        """
        if not key.startswith("base_model.model."):
            return f"base_model.model.{key}"
        return key

    def prepare(self, client_id: int, lora_state: LoRAStateDict) -> str:
        """Convert lora_state → PEFT adapter dir. Returns the dir path."""
        rank = infer_lora_rank(lora_state)
        adapter_dir = os.path.join(self._root, f"client_{client_id}")
        os.makedirs(adapter_dir, exist_ok=True)

        self._write_adapter_config(adapter_dir, rank)

        normalized = {
            self._normalize_key(k): v.contiguous().cpu()
            for k, v in lora_state.items()
        }
        from safetensors.torch import save_file
        save_file(normalized, os.path.join(adapter_dir, "adapter_model.safetensors"))

        self._dirs[client_id] = adapter_dir
        return adapter_dir

    def get_dir(self, client_id: int) -> Optional[str]:
        return self._dirs.get(client_id)

    def cleanup(self) -> None:
        if self._owns_root and os.path.isdir(self._root):
            shutil.rmtree(self._root, ignore_errors=True)
        self._dirs.clear()
