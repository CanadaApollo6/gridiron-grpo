#!/usr/bin/env bash
# Base-model pass@k per task -- the Phase-1 learnability probe (see EXPERIMENTS.md).
# A task with base pass@8 ~ 0 is unlearnable by GRPO; this predicts which cells of
# the recipe matrix can move BEFORE you pay for them.
#   bash scripts/run_passk.sh <model> <label> [adapter]
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
LABEL="${2:-base}"
ADAPTER="${3:-}"
ARGS=""
[ -n "$ADAPTER" ] && ARGS="--adapter $ADAPTER"
python src/eval/pass_at_k.py --model "$MODEL" $ARGS --data data_out/eval.jsonl \
  --label "$LABEL" --n_samples "${N_SAMPLES:-64}" --temperature "${TEMP:-0.9}" \
  --max_new_tokens "${MAX_NEW:-1024}" --limit "${LIMIT:-120}"
