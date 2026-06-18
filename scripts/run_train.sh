#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"

python src/data/build_dataset.py --n_train 8000 --n_eval 800 --seed 7 --out data_out

# Smoke test FIRST (confirms TRL arg names on your installed version)
python src/train_grpo.py --model "$MODEL" --data data_out/train.jsonl \
  --out runs/smoke --max_steps 20

# Real run
python src/train_grpo.py --model "$MODEL" --data data_out/train.jsonl \
  --out runs/grpo-qwen15b --max_steps 1200
