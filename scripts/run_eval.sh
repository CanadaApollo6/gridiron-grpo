#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER="${2:-runs/grpo-qwen15b}"

python src/eval/evaluate.py --model "$MODEL" --data data_out/eval.jsonl --label baseline
python src/eval/evaluate.py --model "$MODEL" --adapter "$ADAPTER" \
  --data data_out/eval.jsonl --label grpo
python src/eval/make_chart.py results/baseline.json results/grpo.json
