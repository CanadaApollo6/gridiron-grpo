#!/usr/bin/env bash
# =============================================================================
# gridiron-grpo — full pipeline for a Hugging Face Job (ephemeral GPU container).
#
# Installs deps, builds data, runs a 20-step smoke test, trains, evaluates base
# vs. tuned WITH confidence intervals + a paired McNemar test + naive-baseline
# floors, optionally measures base pass@k (learnability), charts, and UPLOADS the
# adapter + results to the Hub (the container's disk is wiped when the Job ends).
#
# Job command (works in PowerShell or bash — no shell-specific syntax):
#   uvx --from huggingface_hub hf jobs run --flavor a100-large --timeout 4h --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "apt-get update -qq && apt-get install -y -qq git && git clone --depth 1 https://github.com/CanadaApollo6/gridiron-grpo.git /tmp/gg && bash /tmp/gg/scripts/hf_job.sh"
#
# Tunable via -e on the submit command (all optional):
#   MODEL MAX_STEPS REPO_NAME N_TRAIN N_EVAL NO_VLLM SMOKE_ONLY
#   Hyperparams: LR LR_SCHEDULER_TYPE WARMUP_RATIO BETA NUM_GEN
#                MAX_COMPLETION_LEN MAX_PROMPT_LEN SEED VLLM_GPU_MEM_UTIL
#   Objective:   LOSS_TYPE (grpo|bnpo|dr_grpo) EPSILON_HIGH NO_SCALE_REWARDS=1
#                MASK_TRUNCATED=1 DYNAMIC_SAMPLING=1 GRADED_NUMERIC=1 NO_FORMAT_REWARD=1
#   Learnability: PASSK=1 (base pass@k probe) PASSK_SAMPLES=64 PASSK_LIMIT=120
# Requires the HF_TOKEN secret except when SMOKE_ONLY=1 (no upload).
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

# Optional hyperparameter overrides (empty = use train_grpo.py defaults).
EXTRA_TRAIN_ARGS=""
[ -n "${LR:-}" ]                 && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --lr $LR"
[ -n "${LR_SCHEDULER_TYPE:-}" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --lr_scheduler_type $LR_SCHEDULER_TYPE"
[ -n "${WARMUP_RATIO:-}" ]       && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --warmup_ratio $WARMUP_RATIO"
[ -n "${BETA:-}" ]               && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --beta $BETA"
[ -n "${NUM_GEN:-}" ]            && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --num_generations $NUM_GEN"
[ -n "${MAX_COMPLETION_LEN:-}" ] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --max_completion_len $MAX_COMPLETION_LEN"
[ -n "${MAX_PROMPT_LEN:-}" ]     && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --max_prompt_len $MAX_PROMPT_LEN"
[ -n "${VLLM_GPU_MEM_UTIL:-}" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --vllm_gpu_mem_util $VLLM_GPU_MEM_UTIL"
[ -n "${SEED:-}" ]               && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --seed $SEED"
# GRPO objective / research axes
[ -n "${LOSS_TYPE:-}" ]          && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --loss_type $LOSS_TYPE"
[ -n "${EPSILON_HIGH:-}" ]       && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --epsilon_high $EPSILON_HIGH"
[ "${NO_SCALE_REWARDS:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_scale_rewards"
[ "${MASK_TRUNCATED:-0}" = "1" ]    && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --mask_truncated_completions"
[ "${DYNAMIC_SAMPLING:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --dynamic_sampling"
[ "${GRADED_NUMERIC:-0}" = "1" ]    && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --graded_numeric"
[ "${NO_FORMAT_REWARD:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_format_reward"

echo "=================================================================="
echo " gridiron-grpo HF Job"
echo "   model      = $MODEL"
echo "   max_steps  = $MAX_STEPS"
echo "   push to    = <namespace>/$REPO_NAME"
echo "   extra args =$EXTRA_TRAIN_ARGS"
echo "=================================================================="
nvidia-smi || echo "(no nvidia-smi — are you on a GPU flavor?)"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "repo root: $(pwd)"

# --- Deps (base image already has torch 2.6.0 + CUDA 12.4) -------------------
pip install --no-cache-dir -q -r requirements.txt
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SMOKE_ONLY" != "1" ]; then
  NS="$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")"
  REPO_ID="$NS/$REPO_NAME"
  echo "resolved push target: $REPO_ID"
fi

# --- Data --------------------------------------------------------------------
python src/data/build_dataset.py --n_train "$N_TRAIN" --n_eval "$N_EVAL" --seed 7 --out data_out

# --- Smoke test (confirms TRL's GRPO API on THIS install) --------------------
echo "----- 20-step smoke test -----"
torchrun --nproc_per_node 1 src/train_grpo.py --model "$MODEL" \
    --data data_out/train.jsonl --out runs/smoke --max_steps 20 $NO_VLLM_FLAG $EXTRA_TRAIN_ARGS

if [ "$SMOKE_ONLY" = "1" ]; then
  echo "SMOKE_ONLY=1 -> smoke passed; stopping before the real run. No upload."
  exit 0
fi

# --- Real training -----------------------------------------------------------
echo "----- real run: $MAX_STEPS steps -----"
torchrun --nproc_per_node 1 src/train_grpo.py --model "$MODEL" \
    --data data_out/train.jsonl --out runs/grpo-qwen15b \
    --max_steps "$MAX_STEPS" $NO_VLLM_FLAG $EXTRA_TRAIN_ARGS

# --- Eval base vs. tuned (CIs + floors + per-class), then paired compare ------
EVAL_MAX_NEW="${MAX_COMPLETION_LEN:-1024}"
echo "----- eval (max_new_tokens=$EVAL_MAX_NEW) -----"
python src/eval/evaluate.py --model "$MODEL" --data data_out/eval.jsonl --label baseline \
    --max_new_tokens "$EVAL_MAX_NEW"
python src/eval/evaluate.py --model "$MODEL" --adapter runs/grpo-qwen15b \
    --data data_out/eval.jsonl --label grpo --max_new_tokens "$EVAL_MAX_NEW"
python src/eval/compare.py results/baseline.json results/grpo.json || true
python src/eval/make_chart.py results/baseline.json results/grpo.json || true
python src/eval/results_to_md.py results/baseline.json results/grpo.json > results/table.md || true
cat results/comparison.md || true

# --- Optional: base pass@k learnability probe (costly; gate with PASSK=1) -----
if [ "${PASSK:-0}" = "1" ]; then
  echo "----- base pass@k (learnability; samples=${PASSK_SAMPLES:-64}, limit=${PASSK_LIMIT:-120}) -----"
  python src/eval/pass_at_k.py --model "$MODEL" --data data_out/eval.jsonl --label base \
      --n_samples "${PASSK_SAMPLES:-64}" --temperature 0.9 --limit "${PASSK_LIMIT:-120}" || true
fi

# --- Persist everything to the Hub -------------------------------------------
echo "----- uploading adapter + results to $REPO_ID -----"
REPO_ID="$REPO_ID" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
repo_id = os.environ["REPO_ID"]
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo_id, folder_path="runs/grpo-qwen15b",
                  commit_message="GRPO LoRA adapter + recipe.json")
api.upload_folder(repo_id=repo_id, folder_path="results", path_in_repo="results",
                  commit_message="eval results + CIs + McNemar + chart")
print(f"uploaded adapter + results to https://huggingface.co/{repo_id}")
PY

echo "=================================================================="
echo " DONE. Pull your adapter + results with:"
echo "   uvx --from huggingface_hub hf download $REPO_ID --local-dir ./out"
echo "=================================================================="
