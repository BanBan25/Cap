"""
Language model builder for paper experiments.
Supports: LLaMA3-8B, Qwen3-14B (any HuggingFace causal LM path).
LoRA is injected on attention projections; no classifier head for language.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_config import PaperFedConfig


def build_tokenizer(cfg: PaperFedConfig):
    """
    Load only the tokenizer, not model weights.

    Uses padding_side="right" so that answer-only label masking remains simple.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_causal_lm_lora(
    cfg: PaperFedConfig,
    rank: int,
    tokenizer=None,
    base_model=None,
):
    """
    Load a causal LM and inject LoRA adapters.
    Returns (model, tokenizer, base_model_ref).

    If base_model is provided, reuses it instead of loading from disk.
    """
    if tokenizer is None:
        tokenizer = build_tokenizer(cfg)

    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        base_model.config.use_cache = False
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(
        base_model,
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.lora_target_modules),
        ),
    )

    if getattr(cfg, "enable_gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    return model, tokenizer, base_model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detach LoRA weights to CPU so federated payloads do not pin GPU memory."""
    state = get_peft_model_state_dict(model)
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def _resize_lora_state_to_model(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Align LoRA state dict shapes to the current model's expected shapes."""
    model_sd = get_peft_model_state_dict(model)
    resized: Dict[str, torch.Tensor] = {}
    skipped = 0

    for key, old_tensor in state.items():
        if key not in model_sd:
            skipped += 1
            continue

        target_shape = model_sd[key].shape
        if old_tensor.shape == target_shape:
            resized[key] = old_tensor
            continue

        key_lower = key.lower()
        is_lora_a = key_lower.endswith("lora_a.weight") or ".lora_a." in key_lower
        is_lora_b = key_lower.endswith("lora_b.weight") or ".lora_b." in key_lower

        if is_lora_a:
            if old_tensor.ndim != 2 or target_shape[1] != old_tensor.shape[1]:
                raise RuntimeError(
                    f"[_resize_lora_state_to_model] lora_A key '{key}': "
                    f"unexpected shape change {old_tensor.shape} -> {target_shape}. "
                    f"Only rank dim (dim 0) may differ."
                )
            new_rank, old_rank = target_shape[0], old_tensor.shape[0]
            if old_rank < new_rank:
                pad = torch.zeros(
                    new_rank - old_rank,
                    old_tensor.shape[1],
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
                resized[key] = torch.cat([old_tensor, pad], dim=0)
            else:
                resized[key] = old_tensor[:new_rank, :]

        elif is_lora_b:
            if old_tensor.ndim != 2 or target_shape[0] != old_tensor.shape[0]:
                raise RuntimeError(
                    f"[_resize_lora_state_to_model] lora_B key '{key}': "
                    f"unexpected shape change {old_tensor.shape} -> {target_shape}. "
                    f"Only rank dim (dim 1) may differ."
                )
            new_rank, old_rank = target_shape[1], old_tensor.shape[1]
            if old_rank < new_rank:
                pad = torch.zeros(
                    old_tensor.shape[0],
                    new_rank - old_rank,
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
                resized[key] = torch.cat([old_tensor, pad], dim=1)
            else:
                resized[key] = old_tensor[:, :new_rank]

        else:
            print(
                f"[_resize_lora_state_to_model] WARNING: key '{key}' has shape mismatch "
                f"{old_tensor.shape} vs {target_shape} and is not lora_A/lora_B; skipping."
            )
            skipped += 1

    if skipped:
        print(f"[_resize_lora_state_to_model] skipped {skipped} key(s) not present in current model.")

    return resized


def load_lora_state(model: nn.Module, state: Optional[Dict[str, torch.Tensor]]) -> None:
    if not state:
        return
    aligned_state = _resize_lora_state_to_model(model, state)
    incompatible = set_peft_model_state_dict(model, aligned_state)
    missing = incompatible.missing_keys if hasattr(incompatible, "missing_keys") else []
    unexpected = incompatible.unexpected_keys if hasattr(incompatible, "unexpected_keys") else []
    if missing:
        raise RuntimeError(
            f"LoRA load failed, missing keys ({len(missing)}): {missing[:5]}"
        )
    if unexpected:
        print(f"[load_lora_state] WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
