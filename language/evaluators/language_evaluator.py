"""
Language evaluator — dispatches to per-dataset benchmark suites.

Benchmark coverage by training dataset:

  Alpaca          -> mmlu, mt_bench, alpaca_eval
  GSM8K           -> em (GSM8K 5-shot EM)
  Commonsense15K  -> hellaswag, winogrande, arc_challenge, piqa

All MCQA / EM benchmarks use lm-evaluation-harness (official).
MT-Bench uses FastChat llm_judge (official).
AlpacaEval uses tatsu-lab/alpaca_eval (official).
"""
from __future__ import annotations

import time
import traceback
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

from language.models.causal_lm_lora import build_causal_lm_lora, build_tokenizer, load_lora_state
from paper_config import PaperFedConfig
from shared.lora_ops import infer_lora_rank
from shared.types import ClientInitState


class LanguageEvaluator:
    """Multi-benchmark evaluator for all paper language tasks."""

    def __init__(self, cfg: PaperFedConfig, tokenizer, use_vllm: bool = False):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self._device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self._max_samples = getattr(cfg, "eval_max_samples", 200)
        self._use_vllm = use_vllm
        self._vllm_engine = None

    # ------------------------------------------------------------------
    # Core: evaluate a ready-to-use model on all benchmarks for this dataset
    # ------------------------------------------------------------------

    def evaluate_model(self, model: nn.Module, vllm_engine=None) -> Dict[str, float]:
        """
        Run all paper benchmarks for ``self.cfg.dataset``.
        Returns a dict whose keys are the paper metric names.

        If ``vllm_engine`` is provided externally, uses it directly.
        Otherwise falls back to HF in-memory inference (no vLLM).

        vLLM engine creation is handled by callers (evaluate_flora_global,
        evaluate_personalized) who control the model lifecycle.
        """
        if vllm_engine is None:
            # HF path: model must be on device
            model.to(self._device)
            model.eval()

        ds = self.cfg.dataset
        if ds == "gsm8k":
            return self._eval_gsm8k_suite(model, vllm_engine)
        elif ds == "commonsense15k":
            return self._eval_commonsense_suite(model, vllm_engine)
        elif ds == "alpaca":
            return self._eval_alpaca_suite(model, vllm_engine)

        return {}

    # ------------------------------------------------------------------
    # Per-dataset benchmark suites
    # ------------------------------------------------------------------

    def _eval_gsm8k_suite(self, model: nn.Module, vllm_engine=None) -> Dict[str, float]:
        """GSM8K route -> em only (lm-eval gsm8k 5-shot)."""
        from language.evaluators.gsm8k_eval import evaluate_gsm8k

        results: Dict[str, float] = {}

        # ---- em (GSM8K 5-shot exact-match via lm-eval) ----
        try:
            results["em"] = evaluate_gsm8k(
                model, self.tokenizer, self._device,
                max_samples=self._max_samples,
                vllm_engine=vllm_engine,
            )
        except Exception as e:
            print(f"[eval] GSM8K EM failed: {e}")
            results["em"] = float("nan")

        # NOTE:
        # GSM8K evaluation is intentionally limited to EM only.
        # Keep the old minerva_math path commented here for quick restore if
        # the paper setup later needs the auxiliary MATH benchmark again.
        #
        # try:
        #     results["math"] = evaluate_math(
        #         model, self.tokenizer, self._device,
        #         max_samples=self._max_samples,
        #         vllm_engine=vllm_engine,
        #     )
        # except Exception as e:
        #     print(f"[eval] MATH failed: {e}")
        #     results["math"] = float("nan")

        return results

    def _eval_commonsense_suite(self, model: nn.Module, vllm_engine=None) -> Dict[str, float]:
        """Commonsense15K route -> hellaswag + winogrande + arc_challenge + piqa

        All via lm-evaluation-harness (official). Runs all 4 tasks in a single
        lm-eval call for efficiency.
        """
        from language.evaluators.commonsense_eval import evaluate_commonsense_all

        results: Dict[str, float] = {}
        try:
            results = evaluate_commonsense_all(
                model, self.tokenizer, self._device,
                max_samples=self._max_samples,
                vllm_engine=vllm_engine,
            )
        except Exception as e:
            print(f"[eval] commonsense suite failed: {e}")
            traceback.print_exc()
            for key in (
                "hellaswag_acc", "hellaswag_acc_norm",
                "winogrande_acc",
                "arc_challenge_acc", "arc_challenge_acc_norm",
                "piqa_acc", "piqa_acc_norm",
            ):
                results[key] = float("nan")

        avg_keys = ["hellaswag_acc", "winogrande_acc", "arc_challenge_acc", "piqa_acc"]
        avg_vals = [results[k] for k in avg_keys if isinstance(results.get(k), (int, float))]
        if avg_vals:
            results["commonsense_avg"] = float(sum(avg_vals) / len(avg_vals))

        return results

    def _eval_alpaca_suite(self, model: nn.Module, vllm_engine=None) -> Dict[str, float]:
        """Alpaca route -> mmlu + mt_bench + alpaca_eval"""
        from language.evaluators.alpaca_benchmarks import (
            evaluate_alpaca_eval,
            evaluate_mmlu,
            evaluate_mt_bench,
            load_alpaca_eval_prompts,
        )

        results: Dict[str, float] = {}

        # ---- mmlu ----
        try:
            _t1 = time.time()
            results["mmlu"] = evaluate_mmlu(
                model, self.tokenizer, self.cfg,
                self._device, max_samples=self._max_samples,
                vllm_engine=vllm_engine,
            )
            infer_s = time.time() - _t1
            print(f"[eval]     mmlu        inference={infer_s:.1f}s")
            results["__timing_mmlu__"] = {"data_load_s": 0.0, "inference_s": round(infer_s, 2)}  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] MMLU failed: {e}")
            results["mmlu"] = float("nan")

        # ---- mt_bench ----
        try:
            _t1 = time.time()
            results["mt_bench"] = evaluate_mt_bench(
                model, self.tokenizer, self.cfg,
                self._device, max_samples=self._max_samples,
                vllm_engine=vllm_engine,
            )
            infer_s = time.time() - _t1
            print(f"[eval]     mt_bench    inference={infer_s:.1f}s")
            results["__timing_mt_bench__"] = {"data_load_s": 0.0, "inference_s": round(infer_s, 2)}  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] MT-Bench failed: {e}")
            results["mt_bench"] = float("nan")

        # ---- alpaca_eval ----
        try:
            _t0 = time.time()
            prompts_ae = load_alpaca_eval_prompts(self.cfg)
            load_s = time.time() - _t0
            _t1 = time.time()
            results["alpaca_eval"] = evaluate_alpaca_eval(
                model, self.tokenizer, self.cfg,
                self._device, max_samples=self._max_samples,
                _preloaded=prompts_ae,
                vllm_engine=vllm_engine,
            )
            infer_s = time.time() - _t1
            print(f"[eval]     alpaca_eval data_load={load_s:.1f}s  inference={infer_s:.1f}s")
            results["__timing_alpaca_eval__"] = {"data_load_s": round(load_s, 2), "inference_s": round(infer_s, 2)}  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] AlpacaEval failed: {e}")
            results["alpaca_eval"] = float("nan")

        return results

    # ------------------------------------------------------------------
    # Federated evaluation entry points
    # ------------------------------------------------------------------

    def evaluate_flora_global(
        self,
        lora_state: dict,
        test_loader=None,        # kept for backward compat, ignored for new benchmarks
        test_samples=None,       # kept for backward compat, ignored for new benchmarks
    ) -> Dict[str, float]:
        """Evaluate the FLoRA global model on paper benchmarks.

        When use_vllm=True:
          1. Build PeftModel + load LoRA state
          2. Deep-copy + merge → save to temp dir (original model untouched)
          3. Delete HF model from GPU to free VRAM
          4. Start vLLM engine from saved checkpoint
          5. Run benchmarks via vLLM
          6. Shut down vLLM engine
        """
        rank = infer_lora_rank(lora_state)
        model, _, _ = build_causal_lm_lora(self.cfg, rank, tokenizer=self.tokenizer)
        load_lora_state(model, lora_state)

        t0 = time.time()
        vllm_engine = None
        if self._use_vllm:
            from language.evaluators.vllm_engine import VLLMEngine
            # from_model does: deepcopy → merge → save → free copies
            vllm_engine = VLLMEngine.from_model(
                model, self.tokenizer,
                tensor_parallel_size=self.cfg.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.cfg.vllm_gpu_memory_utilization,
                max_model_len=self.cfg.vllm_max_model_len,
                dtype=self.cfg.vllm_dtype,
            )
            # Now free the original HF model before vLLM claims GPU
            del model
            torch.cuda.empty_cache()
            model = None  # type: ignore[assignment]

        try:
            metrics = self.evaluate_model(model, vllm_engine=vllm_engine)
        finally:
            if vllm_engine is not None:
                vllm_engine.shutdown()
            if model is not None:
                del model
                torch.cuda.empty_cache()

        print(f"[eval] global  benchmark_wall_time={time.time() - t0:.1f}s")
        return metrics

    def evaluate_personalized(
        self,
        per_client_init: Dict[int, ClientInitState],
        client_ranks: Dict[int, int],
        test_loader=None,
        client_sample_counts: Optional[Dict[int, int]] = None,
        aggregation: Literal["uniform", "sample_weighted"] = "uniform",
        test_samples=None,
        on_client_done=None,
    ) -> Dict[str, float]:
        """Evaluate personalized models (EGWSA / FlexLoRA) — weighted average across clients.

        When ``use_vllm=True``, a single vLLM base model is loaded once with
        ``enable_lora=True``, and each client's LoRA adapter is hot-swapped
        via ``LoRARequest`` — no per-client model reload needed.

        When ``use_vllm=False``, falls back to HF inference with per-rank
        model caching (original behavior).
        """
        if self._use_vllm:
            return self._evaluate_personalized_vllm(
                per_client_init, client_ranks,
                client_sample_counts=client_sample_counts,
                aggregation=aggregation,
                on_client_done=on_client_done,
            )
        return self._evaluate_personalized_hf(
            per_client_init, client_ranks,
            client_sample_counts=client_sample_counts,
            aggregation=aggregation,
            on_client_done=on_client_done,
        )

    # ------------------------------------------------------------------
    # Shared helpers for personalized evaluation
    # ------------------------------------------------------------------

    def _select_eval_clients(self, per_client_init):
        max_clients = getattr(self.cfg, "eval_max_clients", -1)
        all_cids = sorted(per_client_init.keys())
        if max_clients > 0 and len(all_cids) > max_clients:
            step = len(all_cids) / max_clients
            eval_cids = {all_cids[int(i * step)] for i in range(max_clients)}
            print(f"[eval] eval_max_clients={max_clients}: evaluating "
                  f"{sorted(eval_cids)} (out of {len(all_cids)} total clients)")
        else:
            eval_cids = set(all_cids)
        return eval_cids

    @staticmethod
    def _log_client(cid, cwall, bm_timing):
        print(
            f"[eval]   client {cid}  total={cwall:.1f}s  "
            + "  ".join(
                f"{bm}(load={v['data_load_s']:.1f}s infer={v['inference_s']:.1f}s)"
                for bm, v in bm_timing.items()
            )
        )

    @staticmethod
    def _strip_timing(raw):
        return {
            k.strip("_").replace("timing_", ""): raw.pop(k)
            for k in list(raw.keys()) if k.startswith("__timing_")
        }

    @staticmethod
    def _aggregate_metrics(
        per_client_metrics, client_sample_counts, aggregation,
        per_client_wall_s, total_wall,
    ):
        if client_sample_counts is None:
            client_sample_counts = {cid: 1 for cid in per_client_metrics}
        if aggregation == "sample_weighted":
            total_n = sum(client_sample_counts[cid] for cid in per_client_metrics)
            w = {cid: client_sample_counts[cid] / total_n for cid in per_client_metrics}
        else:
            n = len(per_client_metrics)
            w = {cid: 1.0 / n for cid in per_client_metrics}
        all_keys: set = set()
        for m in per_client_metrics.values():
            all_keys.update(m.keys())
        result: Dict[str, float] = {}
        for mk in sorted(all_keys):
            result[mk] = sum(
                w[cid] * m.get(mk, 0.0) for cid, m in per_client_metrics.items()
            )
        result["__per_client_wall_s__"] = per_client_wall_s   # type: ignore[assignment]
        result["__total_eval_wall_s__"] = total_wall           # type: ignore[assignment]
        result["__per_client_metrics__"] = per_client_metrics  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------
    # HF path (original behavior, no vLLM)
    # ------------------------------------------------------------------

    def _evaluate_personalized_hf(
        self,
        per_client_init,
        client_ranks,
        client_sample_counts=None,
        aggregation="uniform",
        on_client_done=None,
    ):
        eval_cids = self._select_eval_clients(per_client_init)
        per_client_metrics: Dict[int, Dict[str, float]] = {}
        per_client_wall_s = {}

        models_by_rank: Dict[int, torch.nn.Module] = {}
        t0 = time.time()
        for cid, init in sorted(per_client_init.items()):
            if cid not in eval_cids:
                continue
            rank = client_ranks[cid]
            if rank not in models_by_rank:
                models_by_rank[rank], _, _ = build_causal_lm_lora(
                    self.cfg, rank, tokenizer=self.tokenizer,
                )
            model = models_by_rank[rank]
            load_lora_state(model, init.lora_state)
            t_c0 = time.time()
            raw = self.evaluate_model(model)
            cwall = round(time.time() - t_c0, 2)
            bm_timing = self._strip_timing(raw)
            per_client_metrics[cid] = raw
            per_client_wall_s[cid] = {"total_s": cwall, "benchmarks": bm_timing}
            self._log_client(cid, cwall, bm_timing)
            if on_client_done is not None:
                on_client_done(cid, raw, per_client_wall_s[cid])

        total_wall = round(time.time() - t0, 2)
        print(f"[eval] personalized(HF)  wall={total_wall:.1f}s  "
              f"clients={len(per_client_metrics)}")

        for _m in models_by_rank.values():
            del _m
        del models_by_rank
        torch.cuda.empty_cache()

        return self._aggregate_metrics(
            per_client_metrics, client_sample_counts,
            aggregation, per_client_wall_s, total_wall,
        )

    # ------------------------------------------------------------------
    # vLLM path (dynamic LoRA adapter switching)
    # ------------------------------------------------------------------

    def _evaluate_personalized_vllm(
        self,
        per_client_init,
        client_ranks,
        client_sample_counts=None,
        aggregation="uniform",
        on_client_done=None,
    ):
        from language.evaluators.personalized_vllm_engine import (
            PersonalizedVLLMEngine,
        )

        max_rank = max(client_ranks.values())
        engine = PersonalizedVLLMEngine(
            self.cfg,
            max_lora_rank=max_rank,
            tensor_parallel_size=self.cfg.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.cfg.vllm_gpu_memory_utilization,
            max_model_len=self.cfg.vllm_max_model_len,
            dtype=self.cfg.vllm_dtype,
        )
        engine.start()

        eval_cids = self._select_eval_clients(per_client_init)
        per_client_metrics: Dict[int, Dict[str, float]] = {}
        per_client_wall_s = {}

        t0 = time.time()
        try:
            for cid, init in sorted(per_client_init.items()):
                if cid not in eval_cids:
                    continue
                engine.prepare_client(cid, init.lora_state)
                lm = engine.get_lm_eval_model(cid)

                t_c0 = time.time()
                raw = self._eval_with_lm_eval_model(lm)
                cwall = round(time.time() - t_c0, 2)
                bm_timing = self._strip_timing(raw)
                per_client_metrics[cid] = raw
                per_client_wall_s[cid] = {
                    "total_s": cwall, "benchmarks": bm_timing,
                }
                self._log_client(cid, cwall, bm_timing)
                if on_client_done is not None:
                    on_client_done(cid, raw, per_client_wall_s[cid])
        finally:
            engine.shutdown()

        total_wall = round(time.time() - t0, 2)
        print(f"[eval] personalized(vLLM)  wall={total_wall:.1f}s  "
              f"clients={len(per_client_metrics)}")

        return self._aggregate_metrics(
            per_client_metrics, client_sample_counts,
            aggregation, per_client_wall_s, total_wall,
        )

    # ------------------------------------------------------------------
    # Run benchmarks with a pre-built lm-eval model (vLLM personalized)
    # ------------------------------------------------------------------

    def _eval_with_lm_eval_model(self, lm) -> Dict[str, float]:
        """Run benchmarks using an lm-eval model wrapper directly."""
        ds = self.cfg.dataset
        if ds == "gsm8k":
            return self._eval_gsm8k_lm(lm)
        elif ds == "commonsense15k":
            return self._eval_commonsense_lm(lm)
        elif ds == "alpaca":
            return self._eval_alpaca_lm(lm)
        return {}

    def _eval_gsm8k_lm(self, lm) -> Dict[str, float]:
        from lm_eval import evaluator as lm_evaluator
        from language.evaluators.gsm8k_eval import _extract_acc
        results: Dict[str, float] = {}
        limit = self._max_samples if self._max_samples > 0 else None
        try:
            _t = time.time()
            r = lm_evaluator.simple_evaluate(
                model=lm, tasks=["gsm8k"], num_fewshot=5,
                limit=limit, log_samples=False,
            )
            results["em"] = _extract_acc(r, "gsm8k")
            results["__timing_em__"] = {
                "data_load_s": 0.0,
                "inference_s": round(time.time() - _t, 2),
            }  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] GSM8K failed: {e}")
            results["em"] = float("nan")
        # NOTE:
        # Personalized GSM8K evaluation also stays EM-only. The old
        # minerva_math block is intentionally left out to avoid running a
        # second benchmark during GSM8K evaluation.
        return results

    def _eval_commonsense_lm(self, lm) -> Dict[str, float]:
        from lm_eval import evaluator as lm_evaluator
        from language.evaluators.commonsense_eval import (
            _COMMONSENSE_TASKS, _extract_metric,
        )
        limit = self._max_samples if self._max_samples > 0 else None
        tasks = list(_COMMONSENSE_TASKS.keys())
        out: Dict[str, float] = {}
        try:
            _t = time.time()
            r = lm_evaluator.simple_evaluate(
                model=lm, tasks=tasks, limit=limit, log_samples=False,
            )
            _elapsed = round(time.time() - _t, 2)
            for task_name, cfg in _COMMONSENSE_TASKS.items():
                primary = _extract_metric(r, task_name, cfg["primary"])
                k = f"{task_name}_acc_norm" if "norm" in cfg["primary"] else f"{task_name}_acc"
                out[k] = primary
                if cfg["secondary"]:
                    sec = _extract_metric(r, task_name, cfg["secondary"])
                    k2 = f"{task_name}_acc" if "norm" not in cfg["secondary"] else f"{task_name}_acc_norm"
                    out[k2] = sec
            out["__timing_commonsense__"] = {
                "data_load_s": 0.0,
                "inference_s": _elapsed,
            }  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] Commonsense eval failed: {e}")
            for task_name, cfg in _COMMONSENSE_TASKS.items():
                k = f"{task_name}_acc_norm" if "norm" in cfg["primary"] else f"{task_name}_acc"
                out[k] = float("nan")
                if cfg["secondary"]:
                    k2 = f"{task_name}_acc" if "norm" not in cfg["secondary"] else f"{task_name}_acc_norm"
                    out[k2] = float("nan")
        avg_keys = ["hellaswag_acc", "winogrande_acc", "arc_challenge_acc", "piqa_acc"]
        avg_vals = [out[k] for k in avg_keys if isinstance(out.get(k), (int, float))]
        if avg_vals:
            out["commonsense_avg"] = float(sum(avg_vals) / len(avg_vals))
        return out

    def _eval_alpaca_lm(self, lm) -> Dict[str, float]:
        """Alpaca suite: MMLU via lm-eval, MT-Bench/AlpacaEval via generation."""
        from lm_eval import evaluator as lm_evaluator
        from language.evaluators.alpaca_benchmarks import (
            evaluate_mt_bench,
            evaluate_alpaca_eval, load_alpaca_eval_prompts,
        )
        results: Dict[str, float] = {}
        limit = self._max_samples if self._max_samples > 0 else None

        # MMLU via lm-eval (loglikelihood)
        try:
            _t_mmlu = time.time()
            r = lm_evaluator.simple_evaluate(
                model=lm, tasks=["mmlu"], num_fewshot=5,
                limit=limit, log_samples=False,
            )
            _mmlu_elapsed = round(time.time() - _t_mmlu, 2)
            res = r.get("results", {})
            groups = r.get("groups", {})
            mmlu_found = False
            for source in (groups, res):
                if "mmlu" in source:
                    agg = source["mmlu"]
                    acc = agg.get("acc,none", agg.get("acc", None))
                    if acc is not None:
                        results["mmlu"] = float(acc)
                        mmlu_found = True
                        break
            if not mmlu_found:
                acc_values = []
                for task_name, task_res in res.items():
                    if task_name == "mmlu":
                        continue
                    if "mmlu" in task_name:
                        acc = task_res.get("acc,none", task_res.get("acc", None))
                        if acc is not None:
                            acc_values.append(float(acc))
                results["mmlu"] = (sum(acc_values) / len(acc_values)) if acc_values else float("nan")
            results["__timing_mmlu__"] = {
                "data_load_s": 0.0,
                "inference_s": _mmlu_elapsed,
            }  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] MMLU failed: {e}")
            results["mmlu"] = float("nan")

        # MT-Bench and AlpacaEval via generation adapter
        vllm_adapter = _LmEvalVLLMAdapter(lm)
        try:
            _t1 = time.time()
            results["mt_bench"] = evaluate_mt_bench(
                None, self.tokenizer, self.cfg, self._device,
                max_samples=self._max_samples, vllm_engine=vllm_adapter,
            )
            results["__timing_mt_bench__"] = {
                "data_load_s": 0.0,
                "inference_s": round(time.time() - _t1, 2),
            }  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] MT-Bench failed: {e}")
            results["mt_bench"] = float("nan")

        try:
            _t0 = time.time()
            prompts_ae = load_alpaca_eval_prompts(self.cfg)
            _t1 = time.time()
            results["alpaca_eval"] = evaluate_alpaca_eval(
                None, self.tokenizer, self.cfg, self._device,
                max_samples=self._max_samples,
                _preloaded=prompts_ae, vllm_engine=vllm_adapter,
            )
            results["__timing_alpaca_eval__"] = {
                "data_load_s": round(_t1 - _t0, 2),
                "inference_s": round(time.time() - _t1, 2),
            }  # type: ignore[assignment]
        except Exception as e:
            print(f"[eval] AlpacaEval failed: {e}")
            results["alpaca_eval"] = float("nan")

        return results


class _LmEvalVLLMAdapter:
    """Adapter exposing batch_generate / get_lm_eval_model from lm-eval VLLM wrapper.

    Lets evaluate_mt_bench / evaluate_alpaca_eval use the same vllm_engine
    interface they expect, backed by the personalized lm-eval model.
    """

    def __init__(self, lm_eval_model):
        self._lm = lm_eval_model

    def get_lm_eval_model(self):
        return self._lm

    def _ensure_lm_eval_model(self):
        pass

    def batch_generate(self, prompts, *, max_tokens=512, temperature=0.0,
                       top_p=1.0, stop=None, stop_token_ids=None, seed=None):
        from vllm import SamplingParams
        kwargs = dict(max_tokens=max_tokens, temperature=temperature,
                      top_p=top_p, seed=seed)
        if stop:
            kwargs["stop"] = stop
        if stop_token_ids:
            kwargs["stop_token_ids"] = stop_token_ids
        params = SamplingParams(**kwargs)
        llm = self._lm.model
        lora_req = self._lm.lora_request
        outputs = llm.generate(
            prompts, params, lora_request=lora_req, use_tqdm=True,
        )
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_with_params(self, prompts, sampling_params_list):
        """Per-prompt SamplingParams generation with LoRA request."""
        llm = self._lm.model
        lora_req = self._lm.lora_request
        return llm.generate(
            prompts, sampling_params_list,
            lora_request=lora_req, use_tqdm=True,
        )
