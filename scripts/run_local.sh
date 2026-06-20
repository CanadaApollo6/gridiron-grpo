#!/usr/bin/env bash
# run_local.sh — free local runner for the INFERENCE side on a small GPU (e.g. RTX
# 3080, 10GB). Eval / pass@k / compare / aggregate import no vLLM or trl, so they
# run on a plain CUDA PyTorch (WSL2 or native Windows). The RTX 3080 is Ampere, so
# bf16 works as-is. Only the optional `smoke` subcommand needs the full trl+vLLM
# stack. Tune VRAM with GEN_BATCH / BS (lower them if you hit OOM).
#
# Usage (run from the repo root in WSL):
#   bash scripts/run_local.sh data                                 # build train+eval jsonl
#   bash scripts/run_local.sh passk  <model> [label] [adapter]     # base pass@k (learnability)
#   bash scripts/run_local.sh eval   <model> [adapter] [label]     # single eval
#   bash scripts/run_local.sh basevs <base_model> <adapter_dir>    # base+tuned eval + compare
#   bash scripts/run_local.sh aggregate ["runs_dl/*/"]             # roll up downloaded runs
#   bash scripts/run_local.sh smoke                                # 0.5B GRPO smoke (needs WSL+vLLM)
#
# VRAM/scope knobs (defaults tuned for ~10GB):
#   N_EVAL=800 LIMIT=120 N_SAMPLES=64 GEN_BATCH=16 BS=8 MAXNEW=1024 PASSK_MAXNEW=512 PY=python3
# Examples:
#   bash scripts/run_local.sh passk HuggingFaceTB/SmolLM2-1.7B-Instruct base_smollm2
#   GEN_BATCH=8 LIMIT=60 bash scripts/run_local.sh passk Qwen/Qwen2.5-1.5B-Instruct base_qwen
#   bash scripts/run_local.sh basevs HuggingFaceTB/SmolLM2-1.7B-Instruct runs_dl/gg-smollm2-r1-s7
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"   # no flash-attn needed locally
PY="${PY:-python3}"

N_TRAIN="${N_TRAIN:-8000}"; N_EVAL="${N_EVAL:-800}"; LIMIT="${LIMIT:-120}"
N_SAMPLES="${N_SAMPLES:-64}"; GEN_BATCH="${GEN_BATCH:-16}"; BS="${BS:-8}"; MAXNEW="${MAXNEW:-1024}"

ensure_data() {
  [ -f data_out/eval.jsonl ] || {
    echo "[data] building eval set (n=$N_EVAL, seed 7)"
    $PY src/data/build_dataset.py --n_train 1000 --n_eval "$N_EVAL" --seed 7 --out data_out
  }
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  data)
    $PY src/data/build_dataset.py --n_train "$N_TRAIN" --n_eval "$N_EVAL" --seed 7 --out data_out ;;
  passk)
    ensure_data
    MODEL="${1:?usage: passk <model> [label] [adapter]}"; LABEL="${2:-base}"; ADAPTER="${3:-}"
    A=""; [ -n "$ADAPTER" ] && A="--adapter $ADAPTER"
    echo "[passk] $MODEL  n_samples=$N_SAMPLES gen_batch=$GEN_BATCH limit=$LIMIT"
    $PY src/eval/pass_at_k.py --model "$MODEL" $A --data data_out/eval.jsonl --label "$LABEL" \
        --n_samples "$N_SAMPLES" --gen_batch "$GEN_BATCH" --max_new_tokens "${PASSK_MAXNEW:-512}" --limit "$LIMIT" ;;
  eval)
    ensure_data
    MODEL="${1:?usage: eval <model> [adapter] [label]}"; ADAPTER="${2:-}"; LABEL="${3:-eval}"
    A=""; [ -n "$ADAPTER" ] && A="--adapter $ADAPTER"
    $PY src/eval/evaluate.py --model "$MODEL" $A --data data_out/eval.jsonl --label "$LABEL" \
        --bs "$BS" --max_new_tokens "$MAXNEW" ;;
  basevs)
    ensure_data
    BASE="${1:?usage: basevs <base_model> <adapter_dir>}"; ADAPTER="${2:?need adapter dir}"
    $PY src/eval/evaluate.py --model "$BASE" --data data_out/eval.jsonl --label baseline \
        --bs "$BS" --max_new_tokens "$MAXNEW"
    $PY src/eval/evaluate.py --model "$BASE" --adapter "$ADAPTER" --data data_out/eval.jsonl --label grpo \
        --bs "$BS" --max_new_tokens "$MAXNEW"
    $PY src/eval/compare.py results/baseline.json results/grpo.json ;;
  aggregate)
    GLOB="${1:-runs_dl/*/}"
    $PY src/eval/aggregate.py --out results $GLOB ;;
  smoke)
    bash scripts/smoke_local.sh ;;
  *)
    sed -n '2,28p' "$0" ;;
esac
