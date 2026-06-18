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
#   SMOKE_ONLY   set to 1 to run ONLY data build + 20-step smoke, no real train
#                or upload -- a few-cents validation of the real environment.
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
echo "----- 20-step smoke test -----"
python src/train_grpo.py --model "$MODEL" --data data_out/train.jsonl \
    --out runs/smoke --max_steps 20 $NO_VLLM_FLAG

if [ "$SMOKE_ONLY" = "1" ]; then
  echo "SMOKE_ONLY=1 -> smoke passed; stopping before the real run. No upload."
  exit 0
fi

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
