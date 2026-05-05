"""
Local trainer for language (causal LM) federated clients.
Performs standard causal LM training on tokenized instruction data.
Returns LoRA state for aggregation (no classifier head for language).
"""
from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from language.models.causal_lm_lora import get_lora_state_dict
from paper_config import PaperFedConfig


class LanguageClientTrainer:
    def __init__(self, cfg: PaperFedConfig):
        self.cfg = cfg

    def train_one_round(
        self,
        model: nn.Module,
        loader: DataLoader,
        global_step_start: int,
        total_global_steps: int,
    ) -> Dict[str, object]:
        cfg = self.cfg
        setup_t0 = time.time()
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if next(model.parameters()).device != device:
            model.to(device)
        model.train()

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        trainer_setup_s = time.time() - setup_t0

        running_loss = 0.0
        n_batches = 0
        grad_accum = cfg.gradient_accumulation_steps

        # AMP: use bf16 if available (Ampere+), else fp16
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

        train_loop_t0 = time.time()
        for _ in range(cfg.local_epochs):
            for batch_idx, batch in enumerate(tqdm(loader, desc="local-lm", leave=False)):
                global_step = global_step_start + n_batches
                lr_mult = max(0.0, 1.0 - float(global_step) / float(max(1, total_global_steps)))
                for g in opt.param_groups:
                    g["lr"] = cfg.learning_rate * lr_mult

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                running_loss += float(outputs.loss.detach().cpu())
                n_batches += 1
        train_loop_s = time.time() - train_loop_t0

        lora_state = get_lora_state_dict(model)
        return {
            "lora_state": lora_state,
            "classifier_state": None,
            "mean_loss": running_loss / max(1, n_batches),
            "num_steps": n_batches,
            "timing": {
                "trainer_setup_s": trainer_setup_s,
                "train_loop_s": train_loop_s,
            },
        }
