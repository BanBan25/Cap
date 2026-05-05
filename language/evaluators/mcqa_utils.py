"""
DEPRECATED — Shared utilities for multiple-choice QA evaluation via log-likelihood scoring.

This module is NO LONGER used in the primary evaluation path.
All MCQA benchmarks (MMLU, HellaSwag, WinoGrande, ARC-Challenge, PIQA)
now use lm-evaluation-harness (official) via commonsense_eval.py and
alpaca_benchmarks.py.

Kept for backward compatibility and debugging only.

Original implementation: **Boundary-safe** conditional log-likelihood approach.
Prompt and choice are tokenized SEPARATELY, then explicitly concatenated as
token IDs.  This avoids BPE / SentencePiece re-tokenization mismatches that
occur when inferring the prompt / choice split from a jointly-tokenized string.

Truncation policy: choice tokens are always preserved first.  If the combined
sequence exceeds ``max_length``, the prompt is shortened from the left (keeping
BOS + rightmost context).  If choice tokens are still fully truncated in an
extreme edge case, a large penalty score is returned instead of a silent zero.

NOTE: Minimal implementation for paper experiments. Official benchmarks
(e.g. lm-evaluation-harness) may use slightly different normalization
strategies. Results are directionally correct but may differ from
official leaderboard numbers by a few points.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

_TRUNCATION_PENALTY: float = -1e9


# ------------------------------------------------------------------
# Internal: truncation helper
# ------------------------------------------------------------------

def _truncate_for_budget(
    prompt_ids: List[int],
    choice_ids: List[int],
    max_length: int,
    bos_token_id: Optional[int],
) -> Tuple[List[int], List[int]]:
    """Shorten *prompt_ids* so that ``len(prompt) + len(choice) <= max_length``.

    Choice tokens are never shortened unless the choice alone exceeds
    ``max_length``.  When the prompt must be cut, we keep BOS (if present)
    plus the rightmost tokens (closest context to the choice).
    """
    p = list(prompt_ids)
    c = list(choice_ids)

    if len(p) + len(c) <= max_length:
        return p, c

    budget = max_length - len(c)

    if budget < 1:
        # Choice alone overflows — hard-truncate choice, keep 1 prompt token
        c = c[: max(1, max_length - 1)]
        return p[:1] if p else [], c

    if len(p) > budget:
        has_bos = bos_token_id is not None and p and p[0] == bos_token_id
        if has_bos and budget >= 2:
            # [BOS] + tail of prompt
            p = [p[0]] + p[-(budget - 1) :]
        else:
            p = p[-budget:]

    return p, c


# ------------------------------------------------------------------
# Core scoring
# ------------------------------------------------------------------

@torch.no_grad()
def score_choices_loglikelihood(
    model: nn.Module,
    tokenizer,
    prompt: str,
    choices: List[str],
    device: torch.device,
    length_normalize: bool = True,
    max_length: int = 2048,
) -> List[float]:
    """Score each choice by mean log-likelihood of its tokens given the prompt.

    **Boundary-safe**: prompt and choice are tokenized independently
    (``add_special_tokens=True`` for prompt, ``False`` for choice) and
    concatenated as raw IDs.  The boundary between prompt and choice tokens
    is therefore exact — no heuristic ``prompt_len`` guessing.

    Returns a list of scores (higher = more likely).
    """
    prompt_ids: List[int] = tokenizer(
        prompt, add_special_tokens=True, truncation=False,
    )["input_ids"]

    scores: List[float] = []

    for choice_text in choices:
        choice_ids: List[int] = tokenizer(
            choice_text, add_special_tokens=False, truncation=False,
        )["input_ids"]

        if len(choice_ids) == 0:
            print("[mcqa] WARNING: choice tokenized to 0 tokens; penalty assigned.")
            scores.append(_TRUNCATION_PENALTY)
            continue

        p_ids, c_ids = _truncate_for_budget(
            prompt_ids, choice_ids, max_length, tokenizer.bos_token_id,
        )
        n_prompt = len(p_ids)
        n_choice = len(c_ids)

        if n_choice == 0:
            print("[mcqa] WARNING: choice fully truncated; penalty assigned.")
            scores.append(_TRUNCATION_PENALTY)
            continue

        full_ids = p_ids + c_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

        # Accumulate log P(choice_token_t | all preceding tokens).
        # Position (n_prompt - 1) predicts the first choice token at n_prompt;
        # position (n_prompt + n_choice - 2) predicts the last choice token.
        start = max(n_prompt - 1, 0)
        end = n_prompt + n_choice - 1
        choice_logp = 0.0
        n_scored = 0
        for i in range(start, end):
            choice_logp += log_probs[i, full_ids[i + 1]].item()
            n_scored += 1

        if length_normalize and n_scored > 0:
            choice_logp /= n_scored

        scores.append(choice_logp)

    return scores


# ------------------------------------------------------------------
# High-level accuracy helper
# ------------------------------------------------------------------

def mcqa_accuracy(
    model: nn.Module,
    tokenizer,
    samples: list,
    prompt_fn: Callable,
    choices_fn: Callable,
    label_fn: Callable,
    device: torch.device,
    max_samples: int = 200,
    desc: str = "eval-mcqa",
) -> float:
    """Generic MCQA evaluation via log-likelihood scoring.
    Returns accuracy in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0

    eval_samples = samples if max_samples <= 0 else samples[:max_samples]
    for sample in tqdm(eval_samples, desc=desc, leave=False):
        prompt = prompt_fn(sample)
        choices = choices_fn(sample)
        label = label_fn(sample)
        if not choices or label is None:
            continue

        scores = score_choices_loglikelihood(model, tokenizer, prompt, choices, device)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

    return correct / max(1, total)


def mcqa_accuracy_pair(
    model: nn.Module,
    tokenizer,
    samples: list,
    prompt_fn: Callable,
    choices_fn: Callable,
    label_fn: Callable,
    device: torch.device,
    max_samples: int = 200,
    desc: str = "eval-mcqa",
) -> Tuple[float, float]:
    """Return both raw-LL accuracy (acc) and length-normalized accuracy (acc_norm)."""
    model.eval()
    correct_raw = 0
    correct_norm = 0
    total = 0

    eval_samples = samples if max_samples <= 0 else samples[:max_samples]
    for sample in tqdm(eval_samples, desc=desc, leave=False):
        prompt = prompt_fn(sample)
        choices = choices_fn(sample)
        label = label_fn(sample)
        if not choices or label is None:
            continue

        raw_scores = score_choices_loglikelihood(
            model, tokenizer, prompt, choices, device, length_normalize=False,
        )
        norm_scores = score_choices_loglikelihood(
            model, tokenizer, prompt, choices, device, length_normalize=True,
        )
        raw_pred = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        norm_pred = max(range(len(norm_scores)), key=lambda i: norm_scores[i])
        if raw_pred == label:
            correct_raw += 1
        if norm_pred == label:
            correct_norm += 1
        total += 1

    denom = max(1, total)
    return correct_raw / denom, correct_norm / denom
