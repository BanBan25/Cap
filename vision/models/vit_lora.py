from __future__ import annotations

import os
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import ViTForImageClassification

from vision.config import VisionFedConfig

# ---------- classifier key prefix (works for ViTForImageClassification) ----------
_CLASSIFIER_PREFIX = "classifier."


def build_vit_lora(cfg: VisionFedConfig, rank: int) -> ViTForImageClassification:
    """ViT-Base for image classification with LoRA on attention projections only."""
    quiet = os.environ.get("FLORA_QUIET_VIT_LOAD", "").lower() in ("1", "true", "yes")
    # Rank-sweep builds many ViTs; silence HF/tqdm "LOAD REPORT" / progress spam on stdout.
    out_ctx = redirect_stdout(StringIO()) if quiet else nullcontext()
    err_ctx = redirect_stderr(StringIO()) if quiet else nullcontext()
    with out_ctx, err_ctx:
        model = ViTForImageClassification.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_classes,
            ignore_mismatched_sizes=True,
        )
    lora_alpha = rank  # paper: alpha = rank
    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(cfg.lora_target_modules),
    )
    model = get_peft_model(model, peft_cfg)
    # Only LoRA parameters + classifier head are trainable
    for name, param in model.named_parameters():
        if "lora_" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# ---------- LoRA state helpers ----------

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return get_peft_model_state_dict(model)


def _resize_lora_state_to_model(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Align LoRA state dict shapes to the current model's expected shapes.

    Needed for rank-varying policies (e.g. random / adaptive) across FL rounds,
    where a client's saved LoRA state from round N may have a different rank
    than the model built for round N+1.

    - lora_A shape: [rank, in_dim]  — rank axis is dim 0
      * rank expansion: zero-pad on dim 0
      * rank shrink:    truncate on dim 0

    - lora_B shape: [out_dim, rank] — rank axis is dim 1
      * rank expansion: zero-pad on dim 1
      * rank shrink:    truncate on dim 1

    - other tensors: kept only when shape matches exactly; skipped with a warning otherwise.
    """
    model_sd = get_peft_model_state_dict(model)
    resized: Dict[str, torch.Tensor] = {}
    skipped = 0

    for key, old_tensor in state.items():
        if key not in model_sd:
            # Key no longer exists in this model — skip silently (aggregator may
            # have sent keys for modules not present in this client's config).
            skipped += 1
            continue

        target_shape = model_sd[key].shape

        if old_tensor.shape == target_shape:
            resized[key] = old_tensor
            continue

        # Determine whether this key is lora_A or lora_B by name suffix.
        key_lower = key.lower()
        is_lora_a = key_lower.endswith("lora_a.weight") or ".lora_a." in key_lower
        is_lora_b = key_lower.endswith("lora_b.weight") or ".lora_b." in key_lower

        if is_lora_a:
            # old shape: [old_rank, in_dim]  →  target: [new_rank, in_dim]
            if old_tensor.ndim != 2 or target_shape[1] != old_tensor.shape[1]:
                raise RuntimeError(
                    f"[_resize_lora_state_to_model] lora_A key '{key}': "
                    f"unexpected shape change {old_tensor.shape} → {target_shape}. "
                    f"Only rank dim (dim 0) may differ."
                )
            new_rank = target_shape[0]
            old_rank = old_tensor.shape[0]
            if old_rank < new_rank:
                # rank expansion — zero-pad trailing rows
                pad = torch.zeros(new_rank - old_rank, old_tensor.shape[1],
                                  dtype=old_tensor.dtype, device=old_tensor.device)
                resized[key] = torch.cat([old_tensor, pad], dim=0)
            else:
                # rank shrink — truncate to first new_rank rows
                resized[key] = old_tensor[:new_rank, :]

        elif is_lora_b:
            # old shape: [out_dim, old_rank]  →  target: [out_dim, new_rank]
            if old_tensor.ndim != 2 or target_shape[0] != old_tensor.shape[0]:
                raise RuntimeError(
                    f"[_resize_lora_state_to_model] lora_B key '{key}': "
                    f"unexpected shape change {old_tensor.shape} → {target_shape}. "
                    f"Only rank dim (dim 1) may differ."
                )
            new_rank = target_shape[1]
            old_rank = old_tensor.shape[1]
            if old_rank < new_rank:
                # rank expansion — zero-pad trailing columns
                pad = torch.zeros(old_tensor.shape[0], new_rank - old_rank,
                                  dtype=old_tensor.dtype, device=old_tensor.device)
                resized[key] = torch.cat([old_tensor, pad], dim=1)
            else:
                # rank shrink — truncate to first new_rank columns
                resized[key] = old_tensor[:, :new_rank]

        else:
            # Unknown LoRA tensor type — only keep if shape matches exactly.
            print(
                f"[_resize_lora_state_to_model] WARNING: key '{key}' has shape mismatch "
                f"{old_tensor.shape} vs {target_shape} and is not lora_A/lora_B — skipping."
            )
            skipped += 1
            continue

    if skipped:
        print(f"[_resize_lora_state_to_model] skipped {skipped} key(s) not present in current model.")

    return resized


def load_lora_state(model: nn.Module, state: Optional[Dict[str, torch.Tensor]]) -> None:
    if not state:
        return
    # Align ranks before loading — handles cross-round rank changes under
    # random / adaptive rank policies without crashing on size mismatch.
    aligned_state = _resize_lora_state_to_model(model, state)
    incompatible = set_peft_model_state_dict(model, aligned_state)
    missing = incompatible.missing_keys if hasattr(incompatible, "missing_keys") else []
    unexpected = incompatible.unexpected_keys if hasattr(incompatible, "unexpected_keys") else []
    if missing:
        raise RuntimeError(
            f"LoRA load failed — missing keys ({len(missing)}): {missing[:5]}"
        )
    if unexpected:
        print(f"[load_lora_state] WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")


# ---------- Classifier state helpers ----------

def _resolve_classifier(model: nn.Module) -> nn.Module:
    """Navigate through PeftModel wrapper to find the classifier head."""
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "base_model"):
        bm = model.base_model
        if hasattr(bm, "model") and hasattr(bm.model, "classifier"):
            return bm.model.classifier
        if hasattr(bm, "classifier"):
            return bm.classifier
    raise AttributeError("Cannot locate classifier head in the model")


def get_classifier_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    clf = _resolve_classifier(model)
    return {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}


def load_classifier_state(model: nn.Module, state: Optional[Dict[str, torch.Tensor]]) -> None:
    if not state:
        return
    clf = _resolve_classifier(model)
    clf.load_state_dict(state, strict=True)


# ---------- Combined load (LoRA + classifier) ----------

def load_federated_state(
    model: nn.Module,
    lora_state: Optional[Dict[str, torch.Tensor]],
    classifier_state: Optional[Dict[str, torch.Tensor]],
) -> None:
    """Load both LoRA adapter weights and classifier head into the model."""
    load_lora_state(model, lora_state)
    load_classifier_state(model, classifier_state)
