#!/usr/bin/env bash
# =============================================================================
# gridiron-grpo — full pipeline for a Hugging Face Job (ephemeral GPU container).
#
# This is what runs *inside* the rented GPU container. It installs deps, builds
# data, runs a 20-step smoke test, trains for real, evaluates base vs. tuned,
# charts the result, and — crucially — UPLOADS the LoRA adapter + results to the
# Hub, because the container's disk is wiped when the Job ends.
#
# It expects to be run from within a checkout of this repo. The Job command does
# the clone (works the same in PowerShell or bash — no shell-specific syntax):
#   uvx --from huggingface_hub hf jobs run --flavor a100-large --timeout 4h --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "apt-get update -qq && apt-get install -y -qq git && git clone --depth 1 https://github.com/CanadaApollo6/gridiron-grpo.git /tmp/gg && bash /tmp/gg/scripts/hf_job.sh"
#
# Tunable via -e on the submit command (all optional):
#   MODEL        base model            (default Qwen/Qwen2.5-1.5B-Instruct)
#   MAX_STEPS    real-run train steps  (default 1200)
#   REPO_NAME    Hub repo to push to   (default gridiron-grpo-qwen15b)
#   N_TRAIN      synthetic train rows  (default 8000)
#   N_EVAL       synthetic eval rows   (default 800)
#   NO_VLLM      set to 1 to disable vLLM rollouts (slower but bulletproof)
#   SMOKE_ONLY   set to 1 to run ONLY data build + 20-step smoke, no real train
#                or upload -- a few-cents validation of the real environment.
#   Hyperparams: LR, BETA, NUM_GEN, MAX_COMPLETION_LEN, MAX_PROMPT_LEN, SEED,
#                VLLM_GPU_MEM_UTIL.
#   Objective:   LOSS_TYPE (grpo|bnpo|dr_grpo), EPSILON_HIGH, NO_SCALE_REWARDS=1,
#                MASK_TRUNCATED=1, NO_FORMAT_REWARD=1  (the research axes).
# Requires the HF_TOKEN secret (pass --secrets HF_TOKEN on submit), except when
# SMOKE_ONLY=1 (no upload, so no token needed).
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-1200}"
REPO_NAME="${REPO_NAME:-gridiron-grpo-qwen15b}"
N_TRAIN="${N_TRAIN:-8000}"
N_EVAL="${N_EVAL:-800}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
NO_VLLM_FLAG=""
[ "${NO_VLLM:-0}" = "1" ] && NO_VLLM_FLAG="--no_vllm"

# Optional hyperparameter overrides (pass with -e on the submit command). Empty
# = use train_grpo.py defaults. These let you iterate without editing code, e.g.
#   -e MAX_COMPLETION_LEN=1024 -e LR=2e-6
EXTRA_TRAIN_ARGS=""
[ -n "${LR:-}" ]                 && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --lr $LR"
[ -n "${BETA:-}" ]               && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --beta $BETA"
[ -n "${NUM_GEN:-}" ]            && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --num_generations $NUM_GEN"
[ -n "${MAX_COMPLETION_LEN:-}" ] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --max_completion_len $MAX_COMPLETION_LEN"
[ -n "${MAX_PROMPT_LEN:-}" ]     && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --max_prompt_len $MAX_PROMPT_LEN"
[ -n "${VLLM_GPU_MEM_UTIL:-}" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --vllm_gpu_mem_util $VLLM_GPU_MEM_UTIL"
[ -n "${SEED:-}" ]               && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --seed $SEED"
# GRPO objective variants (research axis)
[ -n "${LOSS_TYPE:-}" ]          && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --loss_type $LOSS_TYPE"
[ -n "${EPSILON_HIGH:-}" ]       && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --epsilon_high $EPSILON_HIGH"
[ "${NO_SCALE_REWARDS:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_scale_rewards"
[ "${MASK_TRUNCATED:-0}" = "1" ]    && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --mask_truncated_completions"
[ "${NO_FORMAT_REWARD:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_format_reward"

echo "=================================================================="
echo " gridiron-grpo HF Job"
echo "   model      = $MODEL"
echo "   max_steps  = $MAX_STEPS"
echo "   push to    = <namespace>/$REPO_NAME"
echo "   vllm       = $([ -n "$NO_VLLM_FLAG" ] && echo off || echo on)"
echo "=================================================================="
nvidia-smi || echo "(no nvidia-smi — are you on a GPU flavor?)"

# --- 1. Locate the repo ------------------------------------------------------
# The Job command git-clones the repo and runs this file from it; cd to the repo
# root regardless of where we were invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "repo root: $(pwd)"

# --- 2. Deps -----------------------------------------------------------------
# The base image already has torch 2.6.0 + CUDA 12.4 (matches the pins). We add
# the RL stack from the repo's pinned requirements.txt. We deliberately do NOT
# install flash-attn: building it eats billed GPU-minutes and can fail, and the
# code falls back to torch's `sdpa` automatically (src/model_utils.py). vLLM is
# what actually drives rollout throughput.
pip install --no-cache-dir -q -r requirements.txt
export HF_HUB_ENABLE_HF_TRANSFER=1

# Resolve the namespace from the token so the push target is <you>/REPO_NAME.
# (Skipped in SMOKE_ONLY mode, which needs no token and uploads nothing.)
if [ "$SMOKE_ONLY" != "1" ]; then
  NS="$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")"
  REPO_ID="$NS/$REPO_NAME"
  echo "resolved push target: $REPO_ID"
fi

# --- 3. Data -----------------------------------------------------------------
python src/data/build_dataset.py --n_train "$N_TRAIN" --n_eval "$N_EVAL" --seed 7 --out data_out

# --- 4. Smoke test (confirms TRL's GRPO API on THIS install before the long run)
# Launch via torchrun: vLLM colocate needs the torch-distributed env vars
# (RANK/WORLD_SIZE/...). torchrun sets them even for 1 process; plain python and
# `accelerate launch --num_processes 1` (simple_launcher) do not -> KeyError 'RANK'.
echo "----- 20-step smoke test -----"
torchrun --nproc_per_node 1 src/train_grpo.py --model "$MODEL" \
    --data data_out/train.jsonl --out runs/smoke --max_steps 20 $NO_VLLM_FLAG $EXTRA_TRAIN_ARGS

if [ "$SMOKE_ONLY" = "1" ]; then
  echo "SMOKE_ONLY=1 -> smoke passed; stopping before the real run. No upload."
  exit 0
fi

# --- 5. Real training --------------------------------------------------------
echo "----- real run: $MAX_STEPS steps -----"
torchrun --nproc_per_node 1 src/train_grpo.py --model "$MODEL" \
    --data data_out/train.jsonl --out runs/grpo-qwen15b \
    --max_steps "$MAX_STEPS" $NO_VLLM_FLAG $EXTRA_TRAIN_ARGS

# --- 6. Eval base vs. tuned + chart -----------------------------------------
# Match eval's generation cap to the training completion budget so eval isn't
# truncated shorter than what the model was trained to produce.
EVAL_MAX_NEW="${MAX_COMPLETION_LEN:-512}"
echo "----- eval (max_new_tokens=$EVAL_MAX_NEW) -----"
python src/eval/evaluate.py --model "$MODEL" --data data_out/eval.jsonl --label baseline \
    --max_new_tokens "$EVAL_MAX_NEW"
python src/eval/evaluate.py --model "$MODEL" --adapter runs/grpo-qwen15b \
    --data data_out/eval.jsonl --label grpo --max_new_tokens "$EVAL_MAX_NEW"
python src/eval/make_chart.py results/baseline.json results/grpo.json || true
python src/eval/results_to_md.py results/baseline.json results/grpo.json > results/table.md || true
cat results/table.md || true

# --- 7. Persist everything to the Hub (the container is about to vanish) ------
# Use the Python API rather than the CLI: the CLI command name ("hf" vs the older
# "huggingface-cli") depends on the huggingface_hub version, but upload_folder is
# stable across all of them.
echo "----- uploading adapter + results to $REPO_ID -----"
REPO_ID="$REPO_ID" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
repo_id = os.environ["REPO_ID"]
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo_id, folder_path="runs/grpo-qwen15b",
                  commit_message="GRPO LoRA adapter")
api.upload_folder(repo_id=repo_id, folder_path="results", path_in_repo="results",
                  commit_message="eval results + chart")
print(f"uploaded adapter + results to https://huggingface.co/{repo_id}")
PY

echo "=================================================================="
echo " DONE. Pull your adapter + results with:"
echo "   uvx --from huggingface_hub hf download $REPO_ID --local-dir ./out"
echo "=================================================================="
