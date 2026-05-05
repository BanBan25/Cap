"""
Convert Flora experiment logs into plot-ready JSON payloads for plot/*.py.

The plotting code has changed over time, so this extractor now normalizes
method aliases and emits shapes that match the latest figure semantics while
remaining backward-compatible where practical.

Important alias:
  EGWSA == CAP == ours
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_by_path(obj: Any, path: str | None) -> Any:
    if path is None or path == "":
        return obj
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def final_metric(rows: List[dict], metric_key: str) -> float | None:
    for row in reversed(rows):
        val = get_by_path(row, metric_key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _method_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_method_id(name: str) -> str:
    token = _method_token(name)
    if token in {"egwsa", "egwsaours", "cap", "capours", "ours"}:
        return "ours"
    if token in {"raflora"}:
        return "raflora"
    if token in {"flexlora"}:
        return "flexlora"
    if token in {"hetlora"}:
        return "hetlora"
    if token in {"flora", "floratrunc"}:
        return "flora"
    return name


def display_method_name(name: str, kind: str) -> str:
    canonical = canonical_method_id(name)
    if canonical == "ours":
        if kind == "pareto":
            return "EGWSA"
        if kind == "compute":
            return "EGWSA (Ours)"
        if kind in {"energy", "hyper"}:
            return "CAP (Ours)"
        return "CAP (Ours)"
    if canonical == "raflora":
        return "raFLoRA"
    if canonical == "flexlora":
        return "FlexLoRA"
    if canonical == "hetlora":
        return "HetLoRA"
    if canonical == "flora":
        return "FLoRA"
    return name


def coerce_numeric_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(float(item))
    return out


def pick_round(rows: List[dict], round_idx: int | None) -> dict | None:
    if round_idx is None:
        return rows[-1] if rows else None
    for row in rows:
        if row.get("round") == round_idx:
            return row
    return None


def prepare_pareto(spec: dict) -> dict:
    out = {}
    for method, info in spec["methods"].items():
        rows = load_jsonl(os.path.join(info["run_dir"], "metrics.jsonl"))
        xs, ys = [], []
        metric_path = info.get("metric_key") or f"eval.{info['metric']}"
        for row in rows:
            x = get_by_path(row, info.get("comm_key", "comm.cumulative_total_bytes"))
            metric = get_by_path(row, metric_path)
            if isinstance(x, (int, float)) and isinstance(metric, (int, float)):
                xs.append(float(x) / (1024 ** 3))
                ys.append(float(metric) * 100.0)
        out[display_method_name(method, "pareto")] = {"x": xs, "y": ys}
    return out


def prepare_compute(spec: dict) -> dict:
    out = {}
    for method, info in spec["methods"].items():
        rows = load_jsonl(os.path.join(info["run_dir"], "metrics.jsonl"))
        train_s = sum(float(get_by_path(r, info.get("train_key", "timing.train_s")) or 0.0) for r in rows)
        agg_s = sum(float(get_by_path(r, info.get("agg_key", "timing.agg_s")) or 0.0) for r in rows)
        dist_s = sum(float(get_by_path(r, info.get("dist_key", "timing.dist_s")) or 0.0) for r in rows)
        display = display_method_name(method, "compute")
        out[display] = {
            "local_train_min": train_s / 60.0,
            "agg_min": agg_s / 60.0,
            "dist_min": dist_s / 60.0,
            "total_min": (train_s + agg_s + dist_s) / 60.0,
        }
    return out


def prepare_rank_behavior(spec: dict) -> dict:
    with open(spec["rank_sweep_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {
        "base_ranks": data["base_ranks"],
        "sample_counts": data["sample_counts"],
        "normalized_sample_counts": data.get("normalized_sample_counts", {}),
        "theoretical_ranks": data["theoretical_ranks"],
        "empirical_inflection_ranks": data.get("empirical_inflection_ranks", {}),
        "sweep_ranks": data.get("sweep_ranks", []),
        "sweep_results": data["sweep_results"],
    }
    if "rq3_behavior_aggregate" in data:
        out["rq3_behavior_aggregate"] = data["rq3_behavior_aggregate"]
    return out


def prepare_energy(spec: dict) -> dict:
    curves = {}
    retention = {}

    for method, info in spec["methods"].items():
        rows = load_jsonl(os.path.join(info["run_dir"], "metrics.jsonl"))
        display = display_method_name(method, "energy")

        xs, ys = [], []
        curve_key = info.get("energy_key", "energy_ratio_q")
        curve_scale = float(info.get("energy_scale", 1.0))
        for row in rows:
            val = get_by_path(row, curve_key)
            if val is None and curve_key == "energy_ratio_q":
                val = row.get("energy_ratio_mean")
            if isinstance(row.get("round"), int) and isinstance(val, (int, float)):
                xs.append(int(row["round"]))
                ys.append(float(val) * curve_scale)
        if xs or ys:
            curves[display] = {"x": xs, "y": ys}

        if "retention_values" in info:
            retention_vals = coerce_numeric_list(info["retention_values"])
        else:
            target = pick_round(rows, info.get("retention_round", spec.get("retention_round")))
            retention_key = info.get("retention_key", "signal_retention_values")
            retention_vals = coerce_numeric_list(get_by_path(target, retention_key)) if target else []
        if retention_vals:
            retention[display] = retention_vals

    out = {
        "higher_rank_energy_ratio": curves,
        "signal_retention": retention,
        "series": curves,
        "retention": retention,
    }
    return out


def prepare_violin(spec: dict) -> dict:
    out = {"before": None, "after": {}}
    before_sources = []
    for method, info in spec["methods"].items():
        rows = load_jsonl(os.path.join(info["run_dir"], "metrics.jsonl"))
        target = pick_round(rows, spec.get("round"))
        if target is None:
            continue
        sv_after = target.get("sv_after") or target.get("sv_snapshot") or {}
        sv_before = target.get("sv_before") or {}
        if not sv_after:
            continue
        layer_key = next(iter(sv_after))
        out["after"][display_method_name(method, "energy")] = sv_after[layer_key]
        if layer_key in sv_before:
            before_sources.extend(sv_before[layer_key])
    out["before"] = before_sources
    return out


def prepare_hyper(spec: dict) -> dict:
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run in spec["runs"]:
        rows = load_jsonl(os.path.join(run["run_dir"], "metrics.jsonl"))
        value = final_metric(rows, run.get("metric_key") or f"eval.{run['metric']}")
        if value is None:
            continue
        panel = run["panel"]
        x_value = str(run["x_value"])
        method = display_method_name(run["method"], "hyper")
        grouped[panel][x_value][method].append(float(value) * 100.0)

    legacy = {}
    panels = {}
    panel_order = spec.get("panel_order")

    panel_names: Iterable[str]
    if panel_order:
        panel_names = panel_order
    else:
        panel_names = grouped.keys()

    for panel in panel_names:
        x_map = grouped.get(panel, {})
        if not x_map:
            continue

        legacy[panel] = {}
        x_labels = spec.get("x_labels", {}).get(panel)
        if not x_labels:
            x_labels = sorted(
                x_map.keys(),
                key=lambda v: float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v,
            )

        methods = sorted({m for method_map in x_map.values() for m in method_map})
        method_means = {m: [] for m in methods}
        method_stds = {m: [] for m in methods}
        method_ns = {m: [] for m in methods}

        for x_value in x_labels:
            legacy[panel][x_value] = {}
            for method in methods:
                vals = x_map.get(x_value, {}).get(method, [])
                stats = {
                    "mean": mean(vals) if vals else None,
                    "std": pstdev(vals) if len(vals) > 1 else 0.0 if vals else None,
                    "n": len(vals),
                }
                legacy[panel][x_value][method] = stats
                method_means[method].append(stats["mean"])
                method_stds[method].append(stats["std"])
                method_ns[method].append(stats["n"])

        panels[panel] = {
            "x_values": x_labels,
            "methods": {
                method: {
                    "mean": method_means[method],
                    "std": method_stds[method],
                    "n": method_ns[method],
                }
                for method in methods
            },
        }

    return {
        "panels": panels,
        "legacy": legacy,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare plot-ready JSON from Flora logs")
    p.add_argument("--spec", required=True, help="Path to a JSON spec describing the extraction task.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    args = p.parse_args()

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    kind = spec["kind"]
    if kind == "pareto":
        out = prepare_pareto(spec)
    elif kind == "compute":
        out = prepare_compute(spec)
    elif kind == "rank_behavior":
        out = prepare_rank_behavior(spec)
    elif kind == "energy":
        out = prepare_energy(spec)
    elif kind == "violin":
        out = prepare_violin(spec)
    elif kind == "hyper":
        out = prepare_hyper(spec)
    else:
        raise ValueError(f"Unknown spec kind: {kind}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[prepare_plot_data] Saved to {args.output}")


if __name__ == "__main__":
    main()
