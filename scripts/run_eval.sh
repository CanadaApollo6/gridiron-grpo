#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER="${2:-runs/grpo-qwen15b}"
MAXNEW="${3:-1024}"   # match the training completion budget (was silently 512)

python src/eval/evaluate.py --model "$MODEL" --data data_out/eval.jsonl --label baseline \
  --max_new_tokens "$MAXNEW"
python src/eval/evaluate.py --model "$MODEL" --adapter "$ADAPTER" \
  --data data_out/eval.jsonl --label grpo --max_new_tokens "$MAXNEW"
# paired base-vs-tuned with CIs + McNemar + floors, then the before/after chart
python src/eval/compare.py results/baseline.json results/grpo.json
python src/eval/make_chart.py results/baseline.json results/grpo.json
