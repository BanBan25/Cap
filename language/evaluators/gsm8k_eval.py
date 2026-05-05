"""
GSM8K + MATH benchmark evaluation via lm-evaluation-harness (official).

- em:   GSM8K 5-shot exact-match (task: "gsm8k", default in lm-eval)
- math: MATH 4-shot with sympy/math_verify equivalence (task: "minerva_math")

Requires:
    pip install "lm-eval[math]"

Previous self-written extract/normalize helpers are kept below (marked
deprecated) for backward compatibility and debugging, but are no longer
used in the primary evaluation path.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


# ===================================================================
# Official evaluation via lm-evaluation-harness
# ===================================================================

def _get_lm_eval_imports():
    """Lazy import lm-eval; raise clear error if missing."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator as lm_evaluator
        return lm_eval, HFLM, lm_evaluator
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness is required for official GSM8K/MATH evaluation. "
            'Install with: pip install "lm-eval[math]"'
        )


def _extract_acc(results: dict, task_group: str) -> float:
    """Extract accuracy from lm-eval results dict.

    Tries group aggregate first, then manual macro-average over subtasks.
    """
    res = results.get("results", {})
    groups = results.get("groups", {})

    for source in (groups, res):
        if task_group in source:
            agg = source[task_group]
            acc = agg.get("exact_match,none",
                  agg.get("exact_match,flexible-extract",
                  agg.get("acc,none",
                  agg.get("acc", None))))
            if acc is not None:
                return float(acc)

    # Manual macro-average across subtasks
    acc_values = []
    for task_name, task_res in res.items():
        if task_name == task_group:
            continue
        if task_group.replace("_", "") in task_name.replace("_", ""):
            for key in ("exact_match,none", "exact_match,flexible-extract", "acc,none", "acc"):
                v = task_res.get(key)
                if v is not None:
                    acc_values.append(float(v))
                    break
    if not acc_values:
        raise ValueError(
            f"No accuracy found for '{task_group}' in lm_eval results: {list(res.keys())}"
        )
    return sum(acc_values) / len(acc_values)


@torch.no_grad()
def evaluate_gsm8k(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    max_samples: int = 0,
    vllm_engine=None,
) -> float:
    """GSM8K 5-shot exact-match via lm-evaluation-harness (official).

    task: "gsm8k" — 5-shot, loglikelihood-based, strict+flexible regex filters.
    Returns accuracy in [0, 1].

    If vllm_engine is provided, uses vLLM backend for accelerated inference.
    """
    _, HFLM, lm_evaluator = _get_lm_eval_imports()

    if vllm_engine is not None:
        lm = vllm_engine.get_lm_eval_model()
        print("[gsm8k] Using vLLM backend.")
    else:
        model.eval()
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))

    limit = max_samples if max_samples > 0 else None
    if limit is not None:
        print(
            f"[gsm8k] WARNING: running with limit={limit}. "
            f"For paper main tables, use max_samples=0 for the full test set."
        )

    results = lm_evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        num_fewshot=5,
        limit=limit,
        log_samples=False,
    )
    return _extract_acc(results, "gsm8k")


@torch.no_grad()
def evaluate_math(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    max_samples: int = 0,
    vllm_engine=None,
) -> float:
    """MATH 4-shot via lm-evaluation-harness minerva_math (official).

    task: "minerva_math" — 4-shot, generate_until, sympy/math_verify equivalence.
    Returns accuracy in [0, 1].

    If vllm_engine is provided, uses vLLM backend for accelerated inference.
    Requires: pip install "lm-eval[math]"
    """
    _, HFLM, lm_evaluator = _get_lm_eval_imports()

    if vllm_engine is not None:
        lm = vllm_engine.get_lm_eval_model()
        print("[math] Using vLLM backend.")
    else:
        model.eval()
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))

    limit = max_samples if max_samples > 0 else None
    if limit is not None:
        print(
            f"[math] WARNING: running with limit={limit} per subtask. "
            f"For paper main tables, use max_samples=0 for the full test set."
        )

    results = lm_evaluator.simple_evaluate(
        model=lm,
        tasks=["minerva_math"],
        limit=limit,
        log_samples=False,
    )
    return _extract_acc(results, "minerva_math")


# ===================================================================
# Deprecated: self-written helpers (kept for backward compat / debug)
# ===================================================================

def _resolve_benchmark_root(cfg) -> str:
    """Deprecated: only used by legacy load_* functions below."""
    if getattr(cfg, "eval_data_root", ""):
        return cfg.eval_data_root
    return os.path.join(cfg.data_root, "benchmarks")


def load_gsm8k_test(cfg) -> List[dict]:
    """Deprecated: legacy data loader. Official path now uses lm-eval."""
    local = os.path.join(_resolve_benchmark_root(cfg), "gsm8k", "test.jsonl")
    if os.path.isfile(local):
        with open(local, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    return [dict(row) for row in ds]


def load_math_test(cfg) -> List[dict]:
    """Deprecated: legacy data loader. Official path now uses lm-eval."""
    local = os.path.join(_resolve_benchmark_root(cfg), "math", "test.jsonl")
    if os.path.isfile(local):
        with open(local, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    return [dict(row) for row in ds]


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Deprecated: legacy answer extraction."""
    match = re.findall(r"####\s*([\-\d,\.]+)", text)
    if match:
        num_str = match[-1].replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Deprecated: legacy \\boxed{} extraction."""
    results: List[str] = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + len("\\boxed{")
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j].strip())
                    break
                depth -= 1
            j += 1
        i = j + 1
    return results[-1] if results else None
