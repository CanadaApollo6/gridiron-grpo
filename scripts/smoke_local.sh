#!/usr/bin/env bash
# =============================================================================
# Local smoke test for a CUDA GPU on Linux or WSL2 (e.g. an RTX 3080, 10 GB).
#
# Purpose: validate the full TRL/vLLM GRPO stack + this repo's reward wiring for
# FREE before spending money on an A100 Job. It does NOT produce a useful model
# -- it runs a tiny 0.5B model for a handful of steps just to prove it trains.
#
# Why WSL/Linux and not native Windows: vLLM has no Windows build, and trl>=0.16
# imports vLLM at module load, so GRPO simply won't import on Windows.
#
# Prereqs (WSL2): a recent NVIDIA driver on the Windows host (gives WSL CUDA
# automatically -- no CUDA toolkit install needed) and `uv` available in WSL.
# Check the GPU is visible first:  nvidia-smi
#
# Run from the repo root inside WSL/Linux:
#   bash scripts/smoke_local.sh
#
# If it prints "saved LoRA adapter + tokenizer to runs/smoke-local", the GRPO
# pipeline works on your installed stack and the real A100 Job is safe to launch.
# =============================================================================
set -euo pipefail

export ATTN_IMPL="${ATTN_IMPL:-sdpa}"   # no flash-attn needed for the smoke

echo "=== GPU visible? ==="
nvidia-smi -L || { echo "No GPU detected. In WSL, update the Windows NVIDIA driver."; exit 1; }

echo "=== creating uv venv (Python 3.12) ==="
uv venv --python 3.12 .venv

echo "=== installing pinned stack (this pulls vLLM + torch; a few minutes) ==="
# vLLM brings its own matching torch 2.6.0+cu124, so install it via requirements.
uv pip install --python .venv -r requirements.txt

PY=.venv/bin/python

echo "=== sanity: CUDA + GRPO import ==="
$PY -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"
$PY -c "from trl import GRPOConfig, GRPOTrainer; print('TRL GRPO import OK')"

echo "=== building a tiny dataset ==="
$PY src/data/build_dataset.py --n_train 64 --n_eval 32 --seed 7 --out data_out

# By default the smoke runs WITHOUT vLLM to stay comfortably inside 10 GB.
# Set USE_VLLM=1 to validate the *exact* path the A100 Job uses (vLLM colocate,
# in-process) on your 3080 -- this is the most faithful free pre-flight. It fits
# 0.5B on 10 GB, but is tighter on VRAM.
VLLM_FLAG="--no_vllm"
if [ "${USE_VLLM:-0}" = "1" ]; then
    VLLM_FLAG="--vllm_mode colocate"
    echo "=== 5-step GRPO smoke (Qwen2.5-0.5B, vLLM colocate) ==="
else
    echo "=== 5-step GRPO smoke (Qwen2.5-0.5B, no vLLM) ==="
fi
# Launch via torchrun (not plain python, and not `accelerate launch
# --num_processes 1` -- that uses accelerate's simple_launcher, which does NOT
# set the distributed env vars). vLLM's colocate backend reads RANK/WORLD_SIZE/...
# torchrun always sets them, even for a single process.
$PY -m torch.distributed.run --nproc_per_node 1 src/train_grpo.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data_out/train.jsonl \
    --out runs/smoke-local \
    $VLLM_FLAG \
    --num_generations 2 \
    --per_device_bs 2 \
    --grad_accum 1 \
    --max_prompt_len 512 \
    --max_completion_len 256 \
    --max_steps 5

echo ""
echo "If you saw 'saved LoRA adapter + tokenizer to runs/smoke-local', the GRPO"
echo "pipeline works on your stack. You're clear to launch the A100 Job."
