#!/usr/bin/env bash
# =============================================================================
# gridiron-grpo — full pipeline for a Hugging Face Job (ephemeral GPU container).
#
# This is what runs *inside* the rented GPU container. It is self-contained:
# it clones the repo, installs deps, builds data, runs a 20-step smoke test,
# trains for real, evaluates base vs. tuned, charts the result, and — crucially —
# UPLOADS the LoRA adapter + results to the Hub, because the container's disk is
# wiped when the Job ends.
#
# Submit it from your laptop with (see README "Run it on Hugging Face Jobs"):
#   uvx --from huggingface_hub hf jobs run --flavor a100-large --timeout 4h \
#       --secrets HF_TOKEN \
#       pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
#       bash -c "$(curl -fsSL https://raw.githubusercontent.com/CanadaApollo6/gridiron-grpo/main/scripts/hf_job.sh)"
#
# Tunable via -e on the submit command (all optional):
#   MODEL        base model            (default Qwen/Qwen2.5-1.5B-Instruct)
#   MAX_STEPS    real-run train steps  (default 1200)
#   REPO_NAME    Hub repo to push to   (default gridiron-grpo-qwen15b)
#   N_TRAIN      synthetic train rows  (default 8000)
#   N_EVAL       synthetic eval rows   (default 800)
#   NO_VLLM      set to 1 to disable vLLM rollouts (slower but bulletproof)
# Requires the HF_TOKEN secret (pass --secrets HF_TOKEN on submit).
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-1200}"
REPO_NAME="${REPO_NAME:-gridiron-grpo-qwen15b}"
N_TRAIN="${N_TRAIN:-8000}"
N_EVAL="${N_EVAL:-800}"
NO_VLLM_FLAG=""
[ "${NO_VLLM:-0}" = "1" ] && NO_VLLM_FLAG="--no_vllm"

echo "=================================================================="
echo " gridiron-grpo HF Job"
echo "   model      = $MODEL"
echo "   max_steps  = $MAX_STEPS"
echo "   push to    = <namespace>/$REPO_NAME"
echo "   vllm       = $([ -n "$NO_VLLM_FLAG" ] && echo off || echo on)"
echo "=================================================================="
nvidia-smi || echo "(no nvidia-smi — are you on a GPU flavor?)"

# --- 1. Code -----------------------------------------------------------------
cd /tmp
rm -rf gridiron-grpo
git clone --depth 1 https://github.com/CanadaApollo6/gridiron-grpo.git
cd gridiron-grpo

# --- 2. Deps -----------------------------------------------------------------
# The base image already has torch + CUDA. We add the RL stack. We deliberately
# do NOT install flash-attn here: building it eats billed GPU-minutes and can
# fail, and the code falls back to torch's `sdpa` automatically (see
# src/model_utils.py). vLLM is what actually drives rollout throughput.
pip install --no-cache-dir -q \
    "transformers>=4.48" "trl>=0.14,<0.18" "peft>=0.13" \
    "datasets>=2.20" "accelerate>=1.0" "vllm>=0.6" "matplotlib>=3.8" \
    "huggingface_hub[cli,hf_transfer]>=0.25"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Resolve the namespace from the token so the push target is <you>/REPO_NAME.
NS="$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")"
REPO_ID="$NS/$REPO_NAME"
echo "resolved push target: $REPO_ID"

# --- 3. Data -----------------------------------------------------------------
python src/data/build_dataset.py --n_train "$N_TRAIN" --n_eval "$N_EVAL" --seed 7 --out data_out

# --- 4. Smoke test (confirms TRL's GRPO API on THIS install before the long run)
echo "----- 20-step smoke test -----"
python src/train_grpo.py --model "$MODEL" --data data_out/train.jsonl \
    --out runs/smoke --max_steps 20 $NO_VLLM_FLAG

# --- 5. Real training --------------------------------------------------------
echo "----- real run: $MAX_STEPS steps -----"
python src/train_grpo.py --model "$MODEL" --data data_out/train.jsonl \
    --out runs/grpo-qwen15b --max_steps "$MAX_STEPS" $NO_VLLM_FLAG

# --- 6. Eval base vs. tuned + chart -----------------------------------------
echo "----- eval -----"
python src/eval/evaluate.py --model "$MODEL" --data data_out/eval.jsonl --label baseline
python src/eval/evaluate.py --model "$MODEL" --adapter runs/grpo-qwen15b \
    --data data_out/eval.jsonl --label grpo
python src/eval/make_chart.py results/baseline.json results/grpo.json || true
python src/eval/results_to_md.py results/baseline.json results/grpo.json > results/table.md || true
cat results/table.md || true

# --- 7. Persist everything to the Hub (the container is about to vanish) ------
echo "----- uploading adapter + results to $REPO_ID -----"
hf upload "$REPO_ID" runs/grpo-qwen15b . --repo-type model --commit-message "GRPO LoRA adapter"
hf upload "$REPO_ID" results results --repo-type model --commit-message "eval results + chart"

echo "=================================================================="
echo " DONE. Pull your adapter + results with:"
echo "   uvx --from huggingface_hub hf download $REPO_ID --local-dir ./out"
echo "=================================================================="
