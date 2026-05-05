"""
Alpaca-route evaluation: MMLU, MT-Bench, AlpacaEval.

Official implementations:
- MMLU:       lm-evaluation-harness  (pip install lm-eval)
- MT-Bench:   FastChat llm_judge     (pip install fschat[model_worker,llm_judge])
              Requires OPENAI_API_KEY for GPT-4 judge.
- AlpacaEval: tatsu-lab/alpaca_eval  (pip install alpaca-eval)
              Requires OPENAI_API_KEY for LLM judge.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn


def _resolve_benchmark_root(cfg) -> str:
    if getattr(cfg, "eval_data_root", ""):
        return cfg.eval_data_root
    return os.path.join(cfg.data_root, "benchmarks")


def _save_model_for_eval(model: nn.Module, tokenizer, cfg, tmp_dir: str) -> None:
    """Merge LoRA into base weights and save as HuggingFace model for external eval tools."""
    from peft import PeftModel

    print(f"[eval] Saving merged model to {tmp_dir} for external evaluation ...")
    if isinstance(model, PeftModel):
        merged = model.merge_and_unload()
        merged.save_pretrained(tmp_dir)
    else:
        model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)


# =====================================================================
# MMLU  — official: lm-evaluation-harness
# =====================================================================

@torch.no_grad()
def evaluate_mmlu(
    model: nn.Module, tokenizer, cfg, device: torch.device, max_samples: int = 200,
    _preloaded=None, vllm_engine=None,
) -> float:
    """
    5-shot MMLU via lm-evaluation-harness (official).
    Returns accuracy in [0, 1].

    If vllm_engine is provided, uses vLLM backend for accelerated inference.
    Requires: pip install lm-eval
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator as lm_evaluator
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness is required for official MMLU evaluation. "
            "Install with: pip install lm-eval"
        )

    if vllm_engine is not None:
        lm = vllm_engine.get_lm_eval_model()
        print("[mmlu] Using vLLM backend.")
    else:
        model.eval()
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))

    # limit=None → full test set; limit=N → N samples **per subtask** (not total)
    # To avoid confusion, only pass limit when explicitly restricting per-subtask samples.
    limit = max_samples if max_samples > 0 else None
    if limit is not None:
        print(
            f"[mmlu] WARNING: running MMLU with limit={limit} per subtask (MMLU-subset). "
            f"For paper main tables, use max_samples=0 to run the full MMLU test set."
        )
    results = lm_evaluator.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=5,
        limit=limit,
        log_samples=False,
    )
    # Prefer results["groups"]["mmlu"] — lm-eval's weight_by_size group aggregate.
    # Fall back to results["results"]["mmlu"] (some versions put it there),
    # then manual macro-average as last resort.
    res = results.get("results", {})
    groups = results.get("groups", {})

    for source in (groups, res):
        if "mmlu" in source:
            agg = source["mmlu"]
            acc = agg.get("acc,none", agg.get("acc", None))
            if acc is not None:
                return float(acc)

    # Manual macro-average across subtasks (excludes the group key itself)
    acc_values = []
    for task_name, task_res in res.items():
        if task_name == "mmlu":
            continue
        if "mmlu" in task_name:
            acc = task_res.get("acc,none", task_res.get("acc", None))
            if acc is not None:
                acc_values.append(float(acc))
    if not acc_values:
        raise ValueError(f"No MMLU accuracy found in lm_eval results: {list(res.keys())}")
    return sum(acc_values) / len(acc_values)


# =====================================================================
# MT-Bench  — official: FastChat llm_judge
# =====================================================================

# Official NEED_REF_CATS — these categories use "math" (single) / "math-mt" (multi) judge
_NEED_REF_CATS = {"math", "coding", "reasoning", "arena-hard-200"}


def _resolve_judge_prompts_file(bench_root: str) -> str:
    """
    FastChat stores judge_prompts.jsonl in two common layouts:
      (A) <bench_root>/mt_bench/judge_prompts.jsonl   (our convention)
      (B) <bench_root>/judge_prompts.jsonl            (FastChat default data/)
    Try both; raise with clear message if neither exists.
    """
    candidates = [
        os.path.join(bench_root, "mt_bench", "judge_prompts.jsonl"),
        os.path.join(bench_root, "judge_prompts.jsonl"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"MT-Bench judge_prompts.jsonl not found. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nDownload from https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data"
    )


def _get_conv_template(model_name: str):
    """Get FastChat conversation template; returns (conv, stop_str, stop_token_ids)."""
    from fastchat.model.model_adapter import get_conversation_template
    conv = get_conversation_template(model_name)
    stop_str = conv.stop_str if hasattr(conv, "stop_str") else None
    stop_token_ids = conv.stop_token_ids if hasattr(conv, "stop_token_ids") else []
    return conv, stop_str, stop_token_ids or []


def _clean_response(text: str, stop_str: str | list[str] | None) -> str:
    """Strip stop strings and trailing whitespace from generated text."""
    if isinstance(stop_str, list):
        cut_positions = [text.index(s) for s in stop_str if s and s in text]
        if cut_positions:
            text = text[:min(cut_positions)]
    elif stop_str and stop_str in text:
        text = text[:text.index(stop_str)]
    return text.strip()


# Official FastChat temperature config per category
_MT_BENCH_TEMPERATURE = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


@torch.no_grad()
def evaluate_mt_bench(
    model: nn.Module, tokenizer, cfg, device: torch.device, max_samples: int = 80,
    _preloaded=None, vllm_engine=None,
) -> float:
    """
    MT-Bench via FastChat llm_judge (official).
    Single-turn uses "default" / "math" judge prompts.
    Multi-turn uses "default-mt" / "math-mt" judge prompts.
    Returns mean score in [1, 10].

    If vllm_engine is provided, uses vLLM batched generation for speed.

    Requires:
      pip install fschat[model_worker,llm_judge]
      OPENAI_API_KEY env var set for GPT-4 judge.
    """
    try:
        from fastchat.llm_judge.gen_judgment import (
            make_judge_single,
            play_a_match_single,
        )
        from fastchat.llm_judge.common import (
            load_questions,
            load_judge_prompts,
            MatchSingle,
        )
    except ImportError:
        raise ImportError(
            "FastChat is required for official MT-Bench evaluation. "
            "Install with: pip install fschat[model_worker,llm_judge]"
        )

    bench_root = _resolve_benchmark_root(cfg)
    question_file = os.path.join(bench_root, "mt_bench", "question.jsonl")
    if not os.path.isfile(question_file):
        raise FileNotFoundError(
            f"MT-Bench question file not found: {question_file}\n"
            "Download: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/mt_bench"
        )
    judge_prompts_file = _resolve_judge_prompts_file(bench_root)

    # ---- Validate reference answers ----
    ref_answer_dir = os.path.join(bench_root, "mt_bench", "reference_answer")
    ref_answers: dict = {}
    if os.path.isdir(ref_answer_dir):
        for ref_file in os.listdir(ref_answer_dir):
            if not ref_file.endswith(".jsonl"):
                continue
            fpath = os.path.join(ref_answer_dir, ref_file)
            with open(fpath, "r", encoding="utf-8") as rf:
                for line in rf:
                    row = json.loads(line.strip())
                    ref_answers[row["question_id"]] = row

    if model is not None:
        model.eval()
    questions = load_questions(question_file, None, None)
    if max_samples > 0:
        questions = questions[:max_samples]
    # Seed is set per-question below (matching FastChat's per-choice seed logic)

    # Pre-check: any NEED_REF question missing ref → hard fail
    for q in questions:
        cat = q.get("category", "")
        if cat in _NEED_REF_CATS and q["question_id"] not in ref_answers:
            raise FileNotFoundError(
                f"MT-Bench question {q['question_id']} (category={cat}) requires "
                f"reference answer but none found in {ref_answer_dir}. "
                f"Download official reference_answer/ from FastChat repo."
            )

    # ---- Generate answers using FastChat conv template ----
    model_name = getattr(cfg, "model_name", "llama-3")
    conv_template, stop_str, stop_token_ids = _get_conv_template(model_name)

    answers = []
    base_seed = int(getattr(cfg, "seed", 42))

    if vllm_engine is not None:
        # ---- vLLM batched generation (multi-turn requires sequential turns) ----
        print("[mt-bench] Using vLLM batched generation.")
        stop_list = []
        if isinstance(stop_str, list):
            stop_list = [s for s in stop_str if s]
        elif stop_str:
            stop_list = [stop_str]

        # Determine max turns
        max_turns = max(len(q.get("turns", [q.get("prompt", "")])) for q in questions)
        # Accumulate answers per question across turns
        q_answers_map: Dict[int, List[str]] = {i: [] for i in range(len(questions))}

        for turn_idx in range(max_turns):
            # Build prompts for all questions that have this turn
            batch_indices = []
            batch_prompts = []
            batch_temps = []
            batch_seeds = []
            for q_idx, q in enumerate(questions):
                turns = q.get("turns", [q.get("prompt", "")])
                if turn_idx >= len(turns):
                    continue
                category = q.get("category", "")
                temperature = _MT_BENCH_TEMPERATURE.get(category, 0.7)

                import copy
                conv = copy.deepcopy(conv_template)
                for h, a_text in zip(turns[:turn_idx], q_answers_map[q_idx][:turn_idx]):
                    conv.append_message(conv.roles[0], h)
                    conv.append_message(conv.roles[1], a_text)
                conv.append_message(conv.roles[0], turns[turn_idx])
                conv.append_message(conv.roles[1], None)

                batch_indices.append(q_idx)
                batch_prompts.append(conv.get_prompt())
                batch_temps.append(temperature)
                # Per-question/turn seed matching HF path
                batch_seeds.append(base_seed + q_idx * 100 + turn_idx)

            if not batch_prompts:
                continue

            # Group by temperature for correctness (vLLM SamplingParams is per-batch)
            temp_groups: Dict[float, List[int]] = {}
            for i, t in enumerate(batch_temps):
                temp_groups.setdefault(t, []).append(i)

            for temp, indices in temp_groups.items():
                sub_prompts = [batch_prompts[i] for i in indices]
                sub_q_indices = [batch_indices[i] for i in indices]
                # Per-prompt SamplingParams for correct per-question seed
                from vllm import SamplingParams
                per_prompt_params = []
                for i in indices:
                    seed_val = batch_seeds[i] if temp > 0.0 else None
                    kwargs = dict(
                        max_tokens=1024,
                        temperature=temp,
                        seed=seed_val,
                    )
                    if stop_list:
                        kwargs["stop"] = stop_list
                    if stop_token_ids:
                        kwargs["stop_token_ids"] = stop_token_ids
                    per_prompt_params.append(SamplingParams(**kwargs))

                raw_outputs = vllm_engine.generate_with_params(
                    sub_prompts, per_prompt_params,
                )
                for qi, out in zip(sub_q_indices, raw_outputs):
                    cleaned = _clean_response(
                        out.outputs[0].text.strip(), stop_str,
                    )
                    q_answers_map[qi].append(cleaned)

        for q_idx, q in enumerate(questions):
            answers.append({
                "question_id": q["question_id"],
                "model_id": "fed_model",
                "choices": [{"index": 0, "turns": q_answers_map[q_idx]}],
            })
    else:
        # ---- Original HF generate path (sequential) ----
        for q_idx, q in enumerate(questions):
            turns = q.get("turns", [q.get("prompt", "")])
            category = q.get("category", "")
            temperature = _MT_BENCH_TEMPERATURE.get(category, 0.7)
            q_answers: List[str] = []

            for i, turn_text in enumerate(turns):
                torch.manual_seed(base_seed + q_idx * 100 + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(base_seed + q_idx * 100 + i)
                import copy
                conv = copy.deepcopy(conv_template)
                for h, a_text in zip(turns[:i], q_answers[:i]):
                    conv.append_message(conv.roles[0], h)
                    conv.append_message(conv.roles[1], a_text)
                conv.append_message(conv.roles[0], turn_text)
                conv.append_message(conv.roles[1], None)
                prompt_str = conv.get_prompt()

                inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                do_sample = temperature > 0.0
                gen_kwargs = dict(
                    max_new_tokens=1024,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                if stop_token_ids:
                    eos_ids = list(set([tokenizer.eos_token_id] + stop_token_ids))
                    gen_kwargs["eos_token_id"] = eos_ids

                generated = model.generate(**inputs, **gen_kwargs)
                gen_ids = generated[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                response = _clean_response(response, stop_str)
                q_answers.append(response)

            answers.append({
                "question_id": q["question_id"],
                "model_id": "fed_model",
                "choices": [{"index": 0, "turns": q_answers}],
            })

    # ---- Load judge config ----
    judge_prompts = load_judge_prompts(judge_prompts_file)
    judges = make_judge_single("gpt-4", judge_prompts)

    # ---- Judge every turn with correct prompt family ----
    # Official keys: "default" (single), "default-mt" (multi),
    #                "math" (single ref), "math-mt" (multi ref)
    scores: List[float] = []
    judgments: List[dict] = []
    q_map = {q["question_id"]: q for q in questions}
    for a in answers:
        q_id = a["question_id"]
        q = q_map.get(q_id)
        if q is None:
            continue
        ref = ref_answers.get(q_id)
        category = q.get("category", "")
        use_ref = category in _NEED_REF_CATS

        n_turns = len(a["choices"][0]["turns"])
        for turn_idx in range(n_turns):
            is_multi = turn_idx > 0
            # Select correct judge prompt family
            if use_ref:
                jkey = "math-mt" if is_multi else "math"
            else:
                jkey = "default-mt" if is_multi else "default"
            if jkey not in judges:
                raise KeyError(
                    f"Missing required MT-Bench judge key '{jkey}' in judge prompts. "
                    f"Available keys: {sorted(judges.keys())}"
                )
            judge = judges[jkey]

            match = MatchSingle(
                question=q,
                model="fed_model",
                answer=a,
                judge=judge,
                ref_answer=ref if use_ref else None,
                multi_turn=is_multi,
            )
            result = play_a_match_single(match, output_file=None)
            if result is not None:
                sc = result.score if hasattr(result, "score") else result.get("score")
                judgments.append({
                    "question_id": q_id,
                    "turn": turn_idx,
                    "judge_key": jkey,
                    "score": sc,
                    "category": category,
                })
                if sc is not None and float(sc) != -1.0:
                    scores.append(float(sc))

    # ---- Persist MT-Bench artifacts for paper reproducibility ----
    out_root = cfg.resolved_output_dir() if hasattr(cfg, "resolved_output_dir") else cfg.output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(out_root, "eval_artifacts", f"mt_bench_{stamp}")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "model_answers.jsonl"), "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(artifact_dir, "judgments.jsonl"), "w", encoding="utf-8") as f:
        for j in judgments:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    # Paper reproducibility: record judge model, date, ref answer source
    with open(os.path.join(artifact_dir, "judge_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "judge_model": "gpt-4",
            "eval_date": stamp,
            "num_questions": len(questions),
            "num_judgments": len(judgments),
            "ref_answer_dir": ref_answer_dir if ref_answers else None,
            "ref_answer_count": len(ref_answers),
            "need_ref_categories": sorted(_NEED_REF_CATS),
        }, f, ensure_ascii=False, indent=2)
    print(f"[mt-bench] Artifacts saved to {artifact_dir}")

    return sum(scores) / max(1, len(scores)) if scores else float("nan")


# =====================================================================
# AlpacaEval  — official: tatsu-lab/alpaca_eval
# =====================================================================

def load_alpaca_eval_prompts(cfg) -> List[dict]:
    """Load AlpacaEval instructions (used for generation step)."""
    local = os.path.join(_resolve_benchmark_root(cfg), "alpaca_eval", "prompts.jsonl")
    if os.path.isfile(local):
        with open(local, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
        return [dict(row) for row in ds]
    except Exception:
        print("[alpaca-eval] Could not load prompts. Returning empty list.")
        return []


@torch.no_grad()
def evaluate_alpaca_eval(
    model: nn.Module, tokenizer, cfg, device: torch.device, max_samples: int = 0,
    _preloaded=None, vllm_engine=None,
) -> float:
    """
    AlpacaEval via tatsu-lab/alpaca_eval (official).
    Returns win-rate in [0, 100].

    If vllm_engine is provided, uses vLLM batched generation for speed.

    Requires:
      pip install alpaca-eval
      OPENAI_API_KEY env var set for LLM judge.
    """
    try:
        import alpaca_eval
        from alpaca_eval import evaluate as ae_evaluate
    except ImportError:
        raise ImportError(
            "alpaca-eval is required for official AlpacaEval evaluation. "
            "Install with: pip install alpaca-eval"
        )

    if model is not None:
        model.eval()
    prompts = _preloaded if _preloaded is not None else load_alpaca_eval_prompts(cfg)
    if not prompts:
        raise ValueError("[alpaca-eval] No prompts available.")
    if max_samples > 0:
        prompts = prompts[:max_samples]

    # Generate responses
    outputs = []
    if vllm_engine is not None:
        # ---- vLLM batched generation ----
        print("[alpaca-eval] Using vLLM batched generation.")
        batch_prompts = []
        batch_instructions = []
        for item in prompts:
            instruction = item.get("instruction", item.get("prompt", ""))
            if not instruction:
                continue
            prompt_text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )
            batch_prompts.append(prompt_text)
            batch_instructions.append(instruction)

        responses = vllm_engine.batch_generate(
            batch_prompts, max_tokens=512, temperature=0.0,
        )
        for instruction, gen_text in zip(batch_instructions, responses):
            outputs.append({
                "instruction": instruction,
                "output": gen_text,
                "generator": "fed_model",
            })
    else:
        # ---- Original HF generate path (sequential) ----
        for item in prompts:
            instruction = item.get("instruction", item.get("prompt", ""))
            if not instruction:
                continue
            prompt_text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_ids = generated[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            outputs.append({
                "instruction": instruction,
                "output": gen_text,
                "generator": "fed_model",
            })

    with tempfile.TemporaryDirectory() as tmp_dir:
        outputs_file = os.path.join(tmp_dir, "outputs.json")
        with open(outputs_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False)

        df_leaderboard, annotations = ae_evaluate(
            model_outputs=outputs_file,
            annotators_config="alpaca_eval_gpt4_turbo_fn",
            is_return_instead_of_print=True,
        )

        # Persist raw artifacts for paper reproducibility.
        out_root = cfg.resolved_output_dir() if hasattr(cfg, "resolved_output_dir") else cfg.output_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = os.path.join(out_root, "eval_artifacts", f"alpaca_eval_{stamp}")
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "model_outputs.json"), "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        with open(os.path.join(artifact_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump(df_leaderboard.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(artifact_dir, "judge_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpaca_eval_version": getattr(alpaca_eval, "__version__", "unknown"),
                    "leaderboard_columns": list(df_leaderboard.columns),
                    "annotations_type": str(type(annotations)),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # AlpacaEval 2.0 主推 length-controlled win rate (LC Win Rate).
    # 优先取 lc_win_rate；旧版本 fallback 到 win_rate.
    row = df_leaderboard.loc["fed_model"]
    lc = row.get("length_controlled_winrate", row.get("lc_win_rate", None))
    wr = row.get("win_rate", None)

    def _to_pct(v):
        if v is None:
            return None
        v = float(v)
        return v * 100.0 if v <= 1.0 else v

    lc_pct = _to_pct(lc)
    wr_pct = _to_pct(wr)

    return lc_pct if lc_pct is not None else (wr_pct if wr_pct is not None else float("nan"))
