#!/usr/bin/env bash
# =============================================================================
# Phase 1 launch reference — gridiron-grpo RLVR study.
# These are the commands to run FROM YOUR machine (authenticated to HF), not in
# the sandbox. They submit Hugging Face Jobs that clone the repo and run it on a
# rented GPU. Run them one block at a time; this file is a reference, not a
# run-it-all script.
#
# Prereqs (once):
#   uvx --from huggingface_hub hf auth login        # HF Pro account + credits
#   export NS="$(uvx --from huggingface_hub hf auth whoami | head -1)"   # your HF username
#
# Decisive base model: SmolLM2-1.7B (open, clean non-Qwen). Swap MODEL to
# allenai/OLMo-2-0425-1B-SFT or meta-llama/Llama-3.2-1B-Instruct (gated) as desired.
# =============================================================================
IMG="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
RUN='bash -c "apt-get update -qq && apt-get install -y -qq git && git clone --depth 1 https://github.com/CanadaApollo6/gridiron-grpo.git /tmp/gg && bash /tmp/gg/scripts/hf_job.sh"'
MODEL="HuggingFaceTB/SmolLM2-1.7B-Instruct"

# --- 0. Push the tooling update first (adds PASSK_ONLY mode + aggregator) -----
#   git add -A && git commit -m "Add PASSK_ONLY job mode + results aggregator" && git push

# --- 1. SMOKE: validate the R1 recipe + new flags on TRL 0.19 (~5 min, cents) -
#        no token needed (SMOKE_ONLY uploads nothing).
eval "uvx --from huggingface_hub hf jobs run --flavor l40sx1 --timeout 1h \
  -e SMOKE_ONLY=1 -e MODEL=$MODEL \
  -e LOSS_TYPE=dr_grpo -e NO_SCALE_REWARDS=1 -e MASK_TRUNCATED=1 \
  $IMG $RUN"

# --- 2. BASE pass@k learnability probe (no training; ~$1-2) -------------------
#        which tasks can move at all? Repeat with MODEL=Qwen/... for the contrast.
eval "uvx --from huggingface_hub hf jobs run --flavor l40sx1 --timeout 2h --secrets HF_TOKEN \
  -e PASSK_ONLY=1 -e MODEL=$MODEL -e REPO_NAME=gg-smollm2-passk \
  $IMG $RUN"

# --- 3. R0 (naive baseline) — does the Qwen failure replicate off-Qwen? (~$6) -
eval "uvx --from huggingface_hub hf jobs run --flavor a100-large --timeout 4h --secrets HF_TOKEN \
  -e MODEL=$MODEL -e REPO_NAME=gg-smollm2-r0-s7 \
  $IMG $RUN"

# --- 4. R1 (Dr. GRPO + DAPO stability) — do the fixes recover learning? (~$6) -
eval "uvx --from huggingface_hub hf jobs run --flavor a100-large --timeout 4h --secrets HF_TOKEN \
  -e MODEL=$MODEL -e REPO_NAME=gg-smollm2-r1-s7 \
  -e LOSS_TYPE=dr_grpo -e NO_SCALE_REWARDS=1 -e MASK_TRUNCATED=1 \
  $IMG $RUN"

# Jobs 2-4 are independent (separate repos) and can run in parallel.
# Watch:   uvx --from huggingface_hub hf jobs ps
#          uvx --from huggingface_hub hf jobs logs <JOB_ID>

# --- 5. When they finish: download + aggregate locally -----------------------
#   mkdir -p runs_dl
#   for r in gg-smollm2-r0-s7 gg-smollm2-r1-s7 gg-smollm2-passk; do
#     uvx --from huggingface_hub hf download "$NS/$r" --local-dir "runs_dl/$r"; done
#   python src/eval/aggregate.py --out results runs_dl/*/
#   # -> results/aggregate.md (family x recipe x kind) + aggregate.csv
