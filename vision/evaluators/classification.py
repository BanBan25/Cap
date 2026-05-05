"""
Vision classification evaluator.

Supports two evaluation paradigms:
  1. **Global model** (FLoRA): one shared LoRA + one shared classifier → single
     evaluation pass on the test set.
  2. **Personalized models** (EGWSA / FlexLoRA): each client has its own LoRA init + a
     shared classifier → each client is evaluated independently on the *same*
     global test set, then per-client metrics are aggregated.

For personalized evaluation, the metric aggregation mode is configurable:
  - ``"uniform"``: arithmetic mean across clients (each client counts equally).
  - ``"sample_weighted"``: weighted mean where the weight of client *i* is
    proportional to its local training sample count *N_i*.

This is purely an **evaluation reporting** choice and does not affect training
or aggregation logic.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vision.config import VisionFedConfig
from vision.federated.lora_ops import infer_lora_rank
from vision.federated.types import ClientInitState
from vision.models.vit_lora import build_vit_lora, load_federated_state


class VisionClassificationEvaluator:
    def __init__(self, cfg: VisionFedConfig):
        self.cfg = cfg

    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=pixel_values).logits
            pred1 = logits.argmax(dim=-1)
            correct1 += (pred1 == labels).sum().item()
            _, idx5 = logits.topk(5, dim=-1)
            correct5 += (idx5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            total += labels.numel()
        return {"top1": correct1 / total, "top5": correct5 / total}

    def evaluate_flora_global(
        self,
        lora_state: dict,
        classifier_state: dict,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate FLoRA global model (shared LoRA + shared classifier)."""
        rank = infer_lora_rank(lora_state)
        model = build_vit_lora(self.cfg, rank)
        load_federated_state(model, lora_state, classifier_state)
        return self.evaluate_model(model, test_loader)

    def evaluate_personalized(
        self,
        per_client_init: Dict[int, ClientInitState],
        client_ranks: Dict[int, int],
        test_loader: DataLoader,
        client_sample_counts: Dict[int, int],
        aggregation: Literal["uniform", "sample_weighted"] = "uniform",
    ) -> Dict[str, float]:
        """
        Evaluate personalized models (e.g. EGWSA / FlexLoRA) on a shared global test set.

        Each client's model (personalized LoRA + shared classifier) is evaluated
        independently; per-client Top-1 / Top-5 are then aggregated according to
        ``aggregation``:
          - ``"uniform"``: simple arithmetic mean.
          - ``"sample_weighted"``: weighted by client local sample count.

        Args:
            per_client_init: per-client ``ClientInitState`` (LoRA + classifier).
            client_ranks: per-client LoRA rank used for model construction.
            test_loader: global test ``DataLoader``.
            client_sample_counts: {client_id: num_local_train_samples}.
            aggregation: ``"uniform"`` or ``"sample_weighted"``.

        Returns:
            ``{"top1": ..., "top5": ...}``
        """
        per_client_metrics: Dict[int, Dict[str, float]] = {}
        models_by_rank: Dict[int, nn.Module] = {}
        for cid, init in sorted(per_client_init.items()):
            rank = client_ranks[cid]
            if rank not in models_by_rank:
                models_by_rank[rank] = build_vit_lora(self.cfg, rank)
            model = models_by_rank[rank]
            load_federated_state(model, init.lora_state, init.classifier_state)
            per_client_metrics[cid] = self.evaluate_model(model, test_loader)

        for _m in models_by_rank.values():
            del _m
        del models_by_rank
        torch.cuda.empty_cache()

        if aggregation == "sample_weighted":
            total_n = sum(client_sample_counts[cid] for cid in per_client_metrics)
            w = {cid: client_sample_counts[cid] / total_n for cid in per_client_metrics}
        else:
            n = len(per_client_metrics)
            w = {cid: 1.0 / n for cid in per_client_metrics}

        top1 = sum(w[cid] * m["top1"] for cid, m in per_client_metrics.items())
        top5 = sum(w[cid] * m["top5"] for cid, m in per_client_metrics.items())
        return {"top1": top1, "top5": top5}
