#!/usr/bin/env bash
set -euo pipefail

accelerate launch --mixed_precision=bf16 src/train_difix.py \
  --dataset_path data/derain.json \
  --output_dir outputs/hhdifix \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --stage_a_steps 15000 \
  --stage_b_steps 5000 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 5 \
  --eval_freq 500 \
  --enable_xformers_memory_efficient_attention
