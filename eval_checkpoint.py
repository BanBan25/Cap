"""
Standalone checkpoint evaluator for language federated experiments.

Usage
-----
# Evaluate a global-model checkpoint (FLoRA / HETLoRA / raFLoRA):
  python eval_checkpoint.py \\
      --config  outputs/language_alpaca_flora_fixed_c10/config.json \\
      --checkpoint outputs/language_alpaca_flora_fixed_c10/checkpoints/round_0009_global.pt

# Evaluate a personalized checkpoint (FlexLoRA / EGWSA) — pass the directory:
  python eval_checkpoint.py \\
      --config  outputs/language_alpaca_flexlora_fixed_c10/config.json \\
      --checkpoint outputs/language_alpaca_flexlora_fixed_c10/checkpoints/round_0007

# Evaluate the best round by loss (reads metrics.jsonl automatically):
  python eval_checkpoint.py \\
      --config  outputs/language_alpaca_flexlora_fixed_c10/config.json \\
      --best-loss

# Limit benchmark samples to speed up evaluation:
  python eval_checkpoint.py ... --max-samples 50

Checkpoint naming conventions (written by language/engine.py):
  round_XXXX_global.pt          — single global LoRA  (flora/hetlora/raflora)
  round_XXXX_client_N.pt        — per-client LoRA     (flexlora/egwsa)
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import re
import sys
import time

# vLLM V1 engine forks a subprocess — CUDA context must not be initialized
# before fork.  We use multiple strategies:
# 1. Set multiprocessing start method to 'spawn'
# 2. Tell vLLM to use 'spawn' for its worker processes
# 3. Disable V1 engine (V0 doesn't fork EngineCore as a separate process)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V1", "0")
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set or context already used

import torch

from paper_config import LANGUAGE_DEFAULTS, PaperFedConfig, VISION_DEFAULTS
from language.models.causal_lm_lora import build_causal_lm_lora, build_tokenizer, load_lora_state
from language.evaluators import LanguageEvaluator
from shared.lora_ops import infer_lora_rank
from shared.types import ClientInitState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> PaperFedConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cfg = PaperFedConfig(**{k: v for k, v in d.items() if hasattr(PaperFedConfig, k)})
    # If ``lora_target_modules`` was omitted from JSON, the dataclass default is ViT
    # names — language runs then crash on LLaMA/Qwen. Align with language defaults.
    if cfg.task_type == "language":
        if list(cfg.lora_target_modules) == list(VISION_DEFAULTS["lora_target_modules"]):
            cfg.lora_target_modules = list(LANGUAGE_DEFAULTS["lora_target_modules"])
    return cfg


def best_loss_round(out_dir: str) -> int:
    """Return the round index with the lowest avg_loss from metrics.jsonl."""
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics.jsonl not found in {out_dir}")

    best_round, best_loss = -1, float("inf")
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            loss = row.get("avg_loss")
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_round = row["round"]

    if best_round < 0:
        raise ValueError("No avg_loss found in metrics.jsonl. "
                         "Re-run training with the updated engine to record losses.")
    print(f"[eval] Best loss round: {best_round}  avg_loss={best_loss:.6f}")
    return best_round


def resolve_checkpoint(args: argparse.Namespace, out_dir: str) -> tuple[str | None, str | None]:
    """
    Returns (global_pt_path, client_dir_prefix).
    Exactly one of the two will be non-None.
    """
    if args.best_loss:
        round_idx = best_loss_round(out_dir)
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        # Try global first
        global_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}_global.pt")
        if os.path.isfile(global_path):
            return global_path, None
        # Try personalized
        prefix = os.path.join(ckpt_dir, f"round_{round_idx:04d}")
        client_files = glob.glob(f"{prefix}_client_*.pt")
        if client_files:
            return None, prefix
        raise FileNotFoundError(
            f"No checkpoint found for round {round_idx} in {ckpt_dir}. "
            f"Make sure training was run with the updated engine."
        )

    ckpt = args.checkpoint
    if ckpt is None:
        raise ValueError("Provide --checkpoint <path> or use --best-loss.")

    # Explicit global file
    if ckpt.endswith(".pt") and os.path.isfile(ckpt):
        if "_global.pt" in ckpt:
            return ckpt, None
        if "_client_" in ckpt:
            # Single client file passed — derive prefix
            prefix = re.sub(r"_client_\d+\.pt$", "", ckpt)
            return None, prefix
        # Unknown suffix — treat as global
        return ckpt, None

    # Directory prefix (e.g. ".../checkpoints/round_0007")
    client_files = glob.glob(f"{ckpt}_client_*.pt")
    if client_files:
        return None, ckpt
    global_path = f"{ckpt}_global.pt"
    if os.path.isfile(global_path):
        return global_path, None

    raise FileNotFoundError(
        f"Cannot locate checkpoint at '{ckpt}'. "
        "Pass a .pt file or the round prefix (e.g. checkpoints/round_0007)."
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_global(cfg: PaperFedConfig, tokenizer, lora_path: str, max_samples: int, use_vllm: bool = False) -> dict:
    lora_state = torch.load(lora_path, map_location="cpu")
    rank = infer_lora_rank(lora_state)
    print(f"[eval] Global LoRA  rank={rank}  file={lora_path}")

    evaluator = LanguageEvaluator(cfg, tokenizer, use_vllm=use_vllm)
    evaluator._max_samples = max_samples
    metrics = evaluator.evaluate_flora_global(lora_state)
    return metrics


def eval_personalized(
    cfg: PaperFedConfig,
    tokenizer,
    prefix: str,
    max_samples: int,
    incremental_path: str | None = None,
    use_vllm: bool = False,
) -> dict:
    client_files = sorted(glob.glob(f"{prefix}_client_*.pt"))
    if not client_files:
        raise FileNotFoundError(f"No client checkpoint files found for prefix: {prefix}")

    # Parse client ids from filenames
    cid_re = re.compile(r"_client_(\d+)\.pt$")
    per_client_init: dict[int, ClientInitState] = {}
    client_ranks: dict[int, int] = {}

    for fpath in client_files:
        m = cid_re.search(fpath)
        if not m:
            continue
        cid = int(m.group(1))
        lora_state = torch.load(fpath, map_location="cpu")
        rank = infer_lora_rank(lora_state)
        per_client_init[cid] = ClientInitState(lora_state=lora_state)
        client_ranks[cid] = rank

    print(f"[eval] Personalized LoRA  clients={sorted(per_client_init.keys())}  "
          f"ranks={client_ranks}  prefix={prefix}")

    # Callback: write one JSONL line per client as soon as it finishes
    def _on_client_done(cid: int, client_metrics: dict, client_timing: dict) -> None:
        if incremental_path is None:
            return
        row = {
            "client_id": cid,
            "metrics": client_metrics,
            "timing": client_timing,
        }
        with open(incremental_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eval]   -> saved client {cid} to {incremental_path}")

    evaluator = LanguageEvaluator(cfg, tokenizer, use_vllm=use_vllm)
    evaluator._max_samples = max_samples
    metrics = evaluator.evaluate_personalized(
        per_client_init,
        client_ranks,
        aggregation="uniform",
        on_client_done=_on_client_done,
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a saved LoRA checkpoint.")

    p.add_argument("--config", required=True,
                   help="Path to config.json saved by the training run.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt file (global) or round prefix "
                        "(e.g. checkpoints/round_0007) for personalized models.")
    p.add_argument("--best-loss", action="store_true",
                   help="Automatically select the round with lowest avg_loss "
                        "from metrics.jsonl in the same output directory.")
    p.add_argument("--max-samples", type=int, default=200,
                   help="Max samples per benchmark (default: 200).")
    p.add_argument("--output", default=None,
                   help="Optional path to write eval results as JSON.")
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=None,
                   help="Use vLLM for accelerated inference. Defaults to enabled for GSM8K.")
    p.add_argument("--vllm-tp", type=int, default=None,
                   help="vLLM tensor_parallel_size (default: 1). Set >1 for multi-GPU.")
    p.add_argument("--vllm-gpu-util", type=float, default=None,
                   help="vLLM gpu_memory_utilization (default: 0.85).")
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                   help="vLLM max_model_len. None = auto from model config.")
    p.add_argument("--vllm-dtype", type=str, default=None,
                   choices=["auto", "float16", "bfloat16"],
                   help="vLLM dtype (default: auto). Use float16 for T4/V100.")

    args = p.parse_args()

    # ---- Load config ----
    cfg = load_config(args.config)
    out_dir = os.path.dirname(os.path.abspath(args.config))

    # Override eval max_samples
    cfg.eval_max_samples = args.max_samples

    use_vllm = (cfg.dataset == "gsm8k") if args.use_vllm is None else args.use_vllm

    # Override vLLM config from CLI if provided
    if args.vllm_tp is not None:
        cfg.vllm_tensor_parallel_size = args.vllm_tp
    if args.vllm_gpu_util is not None:
        cfg.vllm_gpu_memory_utilization = args.vllm_gpu_util
    if args.vllm_max_model_len is not None:
        cfg.vllm_max_model_len = args.vllm_max_model_len
    if args.vllm_dtype is not None:
        cfg.vllm_dtype = args.vllm_dtype

    # ---- Resolve checkpoint path ----
    global_pt, client_prefix = resolve_checkpoint(args, out_dir)

    # ---- Tokenizer ----
    print(f"[eval] Loading tokenizer from {cfg.model_name} ...")
    t_tok = time.time()
    tokenizer = build_tokenizer(cfg)
    print(f"[eval] tokenizer_wall_time={time.time() - t_tok:.2f}s")

    # ---- Incremental save path (per-client JSONL, written as each client finishes) ----
    incremental_path = None
    if client_prefix is not None:
        base = args.output if args.output else client_prefix.replace(os.sep, "_").replace("/", "_")
        incremental_path = re.sub(r"\.json$", "", base) + "_per_client.jsonl"
        print(f"[eval] Per-client results will be streamed to: {incremental_path}")

    # ---- Run evaluation ----
    t_ev = time.time()
    if global_pt is not None:
        metrics = eval_global(cfg, tokenizer, global_pt, args.max_samples, use_vllm=use_vllm)
        ckpt_label = global_pt
    else:
        metrics = eval_personalized(cfg, tokenizer, client_prefix, args.max_samples,
                                    incremental_path=incremental_path,
                                    use_vllm=use_vllm)
        ckpt_label = client_prefix
    eval_wall = time.time() - t_ev
    print(f"[eval] checkpoint_eval_wall_time={eval_wall:.1f}s")

    # ---- Extract side-channel timing keys injected by evaluate_personalized ----
    per_client_wall_s = metrics.pop("__per_client_wall_s__", {})
    total_eval_wall_s = metrics.pop("__total_eval_wall_s__", None)
    per_client_metrics = metrics.pop("__per_client_metrics__", {})

    # ---- Print results ----
    print("\n" + "=" * 50)
    print(f"[eval] Checkpoint : {ckpt_label}")
    print(f"[eval] max_samples: {args.max_samples}")
    print("[eval] Results (aggregated):")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    if per_client_wall_s:
        print("[eval] Per-client eval wall time:")
        for cid, ws in sorted(per_client_wall_s.items()):
            if isinstance(ws, dict):
                total_s = ws.get("total_s", None)
                if total_s is not None:
                    print(f"  client {cid}: {float(total_s):.1f}s")
                else:
                    print(f"  client {cid}: {ws}")
            else:
                print(f"  client {cid}: {float(ws):.1f}s")
    print("=" * 50)

    result = {
        "checkpoint": ckpt_label,
        "config": args.config,
        "max_samples": args.max_samples,
        "timing": {
            "checkpoint_eval_wall_s": round(eval_wall, 2),
            "total_benchmark_wall_s": total_eval_wall_s,
            "per_client_wall_s": {str(k): v for k, v in per_client_wall_s.items()},
        },
        "metrics": metrics,
        "per_client_metrics": {str(k): v for k, v in per_client_metrics.items()},
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[eval] Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
