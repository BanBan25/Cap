"""
Commonsense benchmark evaluators: HellaSwag, WinoGrande, ARC-Challenge, PIQA.

All evaluated via lm-evaluation-harness (official), **0-shot** (no few-shot
exemplars), following the standard protocol for fine-tuned models
(Hu et al., 2023 — LLM-Adapters; Wu et al., 2025 — raFLoRA).

Primary metrics per task (lm-eval YAML defaults):
  - hellaswag:      acc_norm  (+ acc)
  - winogrande:     acc
  - arc_challenge:  acc_norm  (+ acc)
  - piqa:           acc_norm  (+ acc)

Requires: pip install lm-eval
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


# ===================================================================
# Shared lm-eval helpers
# ===================================================================

def _get_lm_eval_imports():
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator as lm_evaluator
        return lm_eval, HFLM, lm_evaluator
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness is required for commonsense evaluation. "
            "Install with: pip install lm-eval"
        )


def _extract_metric(results: dict, task: str, metric_key: str) -> float:
    """Extract a specific metric from lm-eval results for a single task."""
    res = results.get("results", {})
    if task in res:
        val = res[task].get(metric_key)
        if val is not None:
            return float(val)
    # Try without suffix
    for k, v in res.items():
        if k == task or task in k:
            val = v.get(metric_key)
            if val is not None:
                return float(val)
    raise ValueError(
        f"Metric '{metric_key}' not found for task '{task}' "
        f"in lm_eval results: {list(res.keys())}"
    )


# ===================================================================
# Per-benchmark evaluators (each uses task's default num_fewshot)
# ===================================================================

# Task config: task_name, primary metric key, secondary metric key (if any)
_COMMONSENSE_TASKS = {
    "hellaswag":     {"primary": "acc_norm,none", "secondary": "acc,none"},
    "winogrande":    {"primary": "acc,none",      "secondary": None},
    "arc_challenge": {"primary": "acc_norm,none", "secondary": "acc,none"},
    "piqa":          {"primary": "acc_norm,none", "secondary": "acc,none"},
}


@torch.no_grad()
def _evaluate_single_task(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    task_name: str,
    max_samples: int = 0,
    vllm_engine=None,
) -> Dict[str, float]:
    """Run a single commonsense task via lm-eval. Returns dict of metrics."""
    _, HFLM, lm_evaluator = _get_lm_eval_imports()

    if vllm_engine is not None:
        lm = vllm_engine.get_lm_eval_model()
        print(f"[{task_name}] Using vLLM backend.")
    else:
        model.eval()
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))

    limit = max_samples if max_samples > 0 else None
    if limit is not None:
        print(
            f"[{task_name}] WARNING: running with limit={limit}. "
            f"For paper main tables, use max_samples=0."
        )

    # Don't pass num_fewshot — let each task use its YAML default
    results = lm_evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        limit=limit,
        log_samples=False,
    )

    cfg = _COMMONSENSE_TASKS[task_name]
    out: Dict[str, float] = {}

    primary = _extract_metric(results, task_name, cfg["primary"])
    out[f"{task_name}_acc_norm" if "norm" in cfg["primary"] else f"{task_name}_acc"] = primary

    if cfg["secondary"]:
        secondary = _extract_metric(results, task_name, cfg["secondary"])
        out[f"{task_name}_acc" if "norm" not in cfg["secondary"] else f"{task_name}_acc_norm"] = secondary

    return out


def evaluate_hellaswag(
    model: nn.Module, tokenizer, device: torch.device, max_samples: int = 0,
    vllm_engine=None,
) -> Dict[str, float]:
    """HellaSwag 0-shot via lm-eval. Returns acc + acc_norm."""
    return _evaluate_single_task(model, tokenizer, device, "hellaswag", max_samples, vllm_engine)


def evaluate_winogrande(
    model: nn.Module, tokenizer, device: torch.device, max_samples: int = 0,
    vllm_engine=None,
) -> Dict[str, float]:
    """WinoGrande 0-shot via lm-eval. Returns acc."""
    return _evaluate_single_task(model, tokenizer, device, "winogrande", max_samples, vllm_engine)


def evaluate_arc_challenge(
    model: nn.Module, tokenizer, device: torch.device, max_samples: int = 0,
    vllm_engine=None,
) -> Dict[str, float]:
    """ARC-Challenge 0-shot via lm-eval. Returns acc + acc_norm."""
    return _evaluate_single_task(model, tokenizer, device, "arc_challenge", max_samples, vllm_engine)


def evaluate_piqa(
    model: nn.Module, tokenizer, device: torch.device, max_samples: int = 0,
    vllm_engine=None,
) -> Dict[str, float]:
    """PIQA 0-shot via lm-eval. Returns acc + acc_norm."""
    return _evaluate_single_task(model, tokenizer, device, "piqa", max_samples, vllm_engine)


@torch.no_grad()
def evaluate_commonsense_all(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    max_samples: int = 0,
    vllm_engine=None,
) -> Dict[str, float]:
    """Run all 4 commonsense benchmarks in a single lm-eval call.

    Returns dict with keys like hellaswag_acc_norm, winogrande_acc, etc.
    If vllm_engine is provided, uses vLLM backend for accelerated inference.
    """
    _, HFLM, lm_evaluator = _get_lm_eval_imports()

    if vllm_engine is not None:
        lm = vllm_engine.get_lm_eval_model()
        print("[commonsense] Using vLLM backend.")
    else:
        model.eval()
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))

    limit = max_samples if max_samples > 0 else None
    tasks = list(_COMMONSENSE_TASKS.keys())

    results = lm_evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        log_samples=False,
    )

    out: Dict[str, float] = {}
    for task_name, cfg in _COMMONSENSE_TASKS.items():
        primary = _extract_metric(results, task_name, cfg["primary"])
        if "norm" in cfg["primary"]:
            out[f"{task_name}_acc_norm"] = primary
        else:
            out[f"{task_name}_acc"] = primary

        if cfg["secondary"]:
            secondary = _extract_metric(results, task_name, cfg["secondary"])
            if "norm" not in cfg["secondary"]:
                out[f"{task_name}_acc"] = secondary
            else:
                out[f"{task_name}_acc_norm"] = secondary

    return out
