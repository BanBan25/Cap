from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision.config import VisionFedConfig
from vision.models.vit_lora import get_classifier_state_dict, get_lora_state_dict


class VisionClientTrainer:
    """Single-client local training (vision); does not use HF Trainer text path."""

    def __init__(self, cfg: VisionFedConfig):
        self.cfg = cfg

    def train_one_round(
        self,
        model: nn.Module,
        loader: DataLoader,
        global_step_start: int,
        total_global_steps: int,
    ) -> Dict[str, object]:
        cfg = self.cfg
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        ce = nn.CrossEntropyLoss()
        running_loss = 0.0
        n_batches = 0

        for _ in range(cfg.local_epochs):
            for batch in tqdm(loader, desc="local", leave=False):
                global_step = global_step_start + n_batches
                lr_mult = max(0.0, 1.0 - float(global_step) / float(max(1, total_global_steps)))
                for g in opt.param_groups:
                    g["lr"] = cfg.learning_rate * lr_mult

                pixel_values = batch[0].to(device)
                labels = batch[1].to(device)
                opt.zero_grad(set_to_none=True)
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss if hasattr(out, "loss") and out.loss is not None else ce(
                    out.logits, labels
                )
                loss.backward()
                opt.step()
                running_loss += float(loss.detach().cpu())
                n_batches += 1

        lora_state = get_lora_state_dict(model)
        classifier_state = get_classifier_state_dict(model)
        return {
            "lora_state": lora_state,
            "classifier_state": classifier_state,
            "mean_loss": running_loss / max(1, n_batches),
            "num_steps": n_batches,
        }
