# =============================================================================
# Local smoke test for an RTX 3080 (10 GB) on Windows.
#
# Purpose: validate the TRL GRPO API + this repo's reward wiring for FREE before
# spending money on a GPU Job. It does NOT produce a useful model -- it runs a
# tiny 0.5B model for a handful of steps just to prove the pipeline trains.
#
#   - tiny model (Qwen2.5-0.5B-Instruct)   -> fits in 10 GB easily
#   - --no_vllm                            -> vLLM has no Windows build
#   - sdpa attention                       -> no flash-attn needed (auto-detected)
#   - num_generations 2, 5 steps           -> seconds, not hours
#
# Run from the repo root in PowerShell:
#   ./scripts/smoke_local.ps1
#
# If this completes and prints "saved LoRA adapter ...", the GRPO API matches
# the code and the real HF Job is safe to launch.
# =============================================================================
$ErrorActionPreference = "Stop"

# Force the no-flash-attn path explicitly (the code would auto-detect this anyway).
$env:ATTN_IMPL = "sdpa"

Write-Host "=== creating uv venv (Python 3.12) ===" -ForegroundColor Cyan
uv venv --python 3.12 .venv-smoke

Write-Host "=== installing torch (CUDA 12.4) + RL stack ===" -ForegroundColor Cyan
# CUDA 12.4 wheels cover Ampere (RTX 3080, sm_86). No vLLM / flash-attn on Windows.
uv pip install --python .venv-smoke `
    torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-smoke `
    "transformers>=4.48" "trl>=0.14,<0.18" "peft>=0.13" "datasets>=2.20" "accelerate>=1.0"

$py = ".\.venv-smoke\Scripts\python.exe"

Write-Host "=== sanity: is CUDA visible? ===" -ForegroundColor Cyan
& $py -c "import torch; print('cuda:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU only'))"

Write-Host "=== building a tiny dataset ===" -ForegroundColor Cyan
& $py src/data/build_dataset.py --n_train 64 --n_eval 32 --seed 7 --out data_out

Write-Host "=== 5-step GRPO smoke (tiny model, no vLLM) ===" -ForegroundColor Cyan
& $py src/train_grpo.py `
    --model Qwen/Qwen2.5-0.5B-Instruct `
    --data data_out/train.jsonl `
    --out runs/smoke-local `
    --no_vllm `
    --num_generations 2 `
    --per_device_bs 2 `
    --grad_accum 1 `
    --max_prompt_len 512 `
    --max_completion_len 256 `
    --max_steps 5

Write-Host ""
Write-Host "If you saw 'saved LoRA adapter + tokenizer to runs/smoke-local', the" -ForegroundColor Green
Write-Host "GRPO pipeline works on your installed TRL. You're clear to run the HF Job." -ForegroundColor Green
