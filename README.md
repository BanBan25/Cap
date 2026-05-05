## Main Entry Points

- `paper_main.py`: unified experiment entry point for paper experiments.
- `vision_main.py`: vision-only entry point.
- `rank_sweep.py`: rank sweep experiments.
- `eval_checkpoint.py`: post-training evaluation.

## Core Modules

- `shared/aggregators/`: aggregation algorithms.
- `shared/rank_policies.py`: shared rank-policy utilities.
- `vision/`: CIFAR-100 data handling, ViT-LoRA model, vision trainers, and evaluators.
- `language/`: language data modules, causal-LM LoRA model, trainers, and evaluators.
- `tools/`: diagnostic scripts.

## Run

Install dependencies with:

```bash
pip install -r requirements.txt
```