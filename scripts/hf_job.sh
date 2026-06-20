#!/usr/bin/env bash
# =============================================================================
# gridiron-grpo — full pipeline for a Hugging Face Job (ephemeral GPU container).
#
# Installs deps, builds data, runs a 20-step smoke test, trains, evaluates base
# vs. tuned WITH confidence intervals + a paired McNemar test + naive-baseline
# floors, optionally measures base pass@k (learnability), charts, and UPLOADS the
# adapter + results to the Hub (the container's disk is wiped when the Job ends).
#
# The adapter+results upload happens BEFORE the optional pass@k probe, so a long
# pass@k can never cost you the trained adapter.
#
# Job command (works in PowerShell or bash):
#   uvx --from huggingface_hub hf jobs run --flavor a100-large --timeout 4h --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "apt-get update -qq && apt-get install -y -qq git && git clone --depth 1 https://github.com/CanadaApollo6/gridiron-grpo.git /tmp/gg && bash /tmp/gg/scripts/hf_job.sh"
#
# Modes (set with -e):
#   SMOKE_ONLY=1   data build + 20-step smoke only (no token, no upload).
#   PASSK_ONLY=1   data build + BASE pass@k probe only, upload, exit (no training).
#                  Cheap learnability check per family. Needs HF_TOKEN.
#   (default)      full train -> eval -> compare -> upload [-> optional PASSK=1].
#
# Tunables (-e): MODEL MAX_STEPS REPO_NAME N_TRAIN N_EVAL NO_VLLM
#   LR LR_SCHEDULER_TYPE WARMUP_RATIO BETA NUM_GEN MAX_COMPLETION_LEN MAX_PROMPT_LEN
#   SEED VLLM_GPU_MEM_UTIL | LOSS_TYPE EPSILON_HIGH NO_SCALE_REWARDS MASK_TRUNCATED
#   DYNAMIC_SAMPLING GRADED_NUMERIC NO_FORMAT_REWARD | PASSK PASSK_SAMPLES PASSK_LIMIT
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-1200}"
REPO_NAME="${REPO_NAME:-gridiron-grpo-qwen15b}"
N_TRAIN="${N_TRAIN:-8000}"
N_EVAL="${N_EVAL:-800}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
PASSK_ONLY="${PASSK_ONLY:-0}"
NO_VLLM_FLAG=""
[ "${NO_VLLM:-0}" = "1" ] && NO_VLLM_FLAG="--no_vllm"

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
[ -n "${LOSS_TYPE:-}" ]          && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --loss_type $LOSS_TYPE"
[ -n "${EPSILON_HIGH:-}" ]       && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --epsilon_high $EPSILON_HIGH"
[ "${NO_SCALE_REWARDS:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_scale_rewards"
[ "${MASK_TRUNCATED:-0}" = "1" ]    && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --mask_truncated_completions"
[ "${DYNAMIC_SAMPLING:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --dynamic_sampling"
[ "${GRADED_NUMERIC:-0}" = "1" ]    && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --graded_numeric"
[ "${NO_FORMAT_REWARD:-0}" = "1" ]  && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --no_format_reward"

echo "=================================================================="
echo " gridiron-grpo HF Job  | model=$MODEL  mode=$([ "$SMOKE_ONLY" = 1 ] && echo smoke || ([ "$PASSK_ONLY" = 1 ] && echo passk_only || echo full))"
echo "   max_steps=$MAX_STEPS  push=<ns>/$REPO_NAME  extra=$EXTRA_TRAIN_ARGS"
echo "=================================================================="
nvidia-smi || echo "(no nvidia-smi — are you on a GPU flavor?)"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "repo root: $(pwd)"

pip install --no-cache-dir -q -r requirements.txt
export HF_HUB_ENABLE_HF_TRANSFER=1

# Resolve push target unless this is a no-upload smoke.
if [ "$SMOKE_ONLY" != "1" ]; then
  NS="$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")"
  REPO_ID="$NS/$REPO_NAME"
  echo "resolved push target: $REPO_ID"
fi

# --- Data --------------------------------------------------------------------
python src/data/build_dataset.py --n_train "$N_TRAIN" --n_eval "$N_EVAL" --seed 7 --out data_out

# --- PASSK_ONLY: cheap base-model learnability probe, then stop --------------
if [ "$PASSK_ONLY" = "1" ]; then
  echo "----- PASSK_ONLY: base pass@k (samples=${PASSK_SAMPLES:-64}, limit=${PASSK_LIMIT:-120}) -----"
  python src/eval/pass_at_k.py --model "$MODEL" --data data_out/eval.jsonl --label base \
      --n_samples "${PASSK_SAMPLES:-64}" --temperature 0.9 \
      --max_new_tokens "${MAX_COMPLETION_LEN:-1024}" --limit "${PASSK_LIMIT:-120}"
  REPO_ID="$REPO_ID" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(); rid = os.environ["REPO_ID"]
api.create_repo(rid, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=rid, folder_path="results", path_in_repo="results",
                  commit_message="base pass@k probe")
print(f"uploaded pass@k to https://huggingface.co/{rid}")
PY
  echo "PASSK_ONLY done; no training. Exiting."
  exit 0
fi

# --- Smoke test (confirms TRL's GRPO API + the requested flags on THIS install)
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

# --- Persist adapter + results to the Hub (BEFORE any optional extras) --------
echo "----- uploading adapter + results to $REPO_ID -----"
REPO_ID="$REPO_ID" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(); repo_id = os.environ["REPO_ID"]
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo_id, folder_path="runs/grpo-qwen15b",
                  commit_message="GRPO LoRA adapter + recipe.json")
api.upload_folder(repo_id=repo_id, folder_path="results", path_in_repo="results",
                  commit_message="eval results + CIs + McNemar + chart")
print(f"uploaded adapter + results to https://huggingface.co/{repo_id}")
PY

# --- Optional: base pass@k AFTER the upload (so it can't cost the adapter) ----
if [ "${PASSK:-0}" = "1" ]; then
  echo "----- base pass@k (samples=${PASSK_SAMPLES:-64}, limit=${PASSK_LIMIT:-120}) -----"
  python src/eval/pass_at_k.py --model "$MODEL" --data data_out/eval.jsonl --label base \
      --n_samples "${PASSK_SAMPLES:-64}" --temperature 0.9 \
      --max_new_tokens "$EVAL_MAX_NEW" --limit "${PASSK_LIMIT:-120}" || true
  REPO_ID="$REPO_ID" python - <<'PY' || true
import os
from huggingface_hub import HfApi
api = HfApi(); repo_id = os.environ["REPO_ID"]
api.upload_folder(repo_id=repo_id, folder_path="results", path_in_repo="results",
                  commit_message="base pass@k")
PY
fi

echo "=================================================================="
echo " DONE. Pull your adapter + results with:"
echo "   uvx --from huggingface_hub hf download $REPO_ID --local-dir ./out"
echo "=================================================================="
