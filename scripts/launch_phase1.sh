#!/usr/bin/env bash
# Phase-2-lite launcher — {SmolLM2, Qwen} x {R0 naive, R1 Dr.GRPO+DAPO}.
# Run from YOUR authenticated machine (jobs bill your HF account; clone is from GitHub).
# Prereqs (once):
#   uvx --from huggingface_hub hf auth login
#   export NS="$(uvx --from huggingface_hub hf auth whoami | head -1)"   # your HF username (for `dl`)
# Usage: run the smoke first, then the four training jobs, then `dl` to aggregate:
#   bash scripts/launch_phase1.sh smoke        # ~5 min, cents — wait for "smoke passed"
#   bash scripts/launch_phase1.sh smol-r0      # each ~$5-6 on l40sx1; run in 4 terminals
#   bash scripts/launch_phase1.sh smol-r1      #   (or Ctrl-C the log stream after submit;
#   bash scripts/launch_phase1.sh qwen-r0      #    the job keeps running on HF)
#   bash scripts/launch_phase1.sh qwen-r1
#   bash scripts/launch_phase1.sh dl           # download all 4 + aggregate -> results/aggregate.md
set -euo pipefail

job() {  # job "<-e flags>" [flavor] [timeout]
  uvx --from huggingface_hub hf jobs run --flavor "${2:-l40sx1}" --timeout "${3:-5h}" --secrets HF_TOKEN $1 \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c "apt-get update -qq && apt-get install -y -qq git && git clone --depth 1 https://github.com/CanadaApollo6/gridiron-grpo.git /tmp/gg && bash /tmp/gg/scripts/hf_job.sh"
}

SMOL=HuggingFaceTB/SmolLM2-1.7B-Instruct
QWEN=Qwen/Qwen2.5-1.5B-Instruct
FIX="-e LOSS_TYPE=dr_grpo -e NO_SCALE_REWARDS=1 -e MASK_TRUNCATED=1"   # R1 = Dr.GRPO + DAPO stability
REPOS="gg-smollm2-r0-s7 gg-smollm2-r1-s7 gg-qwen15b-r0-s7 gg-qwen15b-r1-s7"

case "${1:-help}" in
  smoke)   job "-e SMOKE_ONLY=1 -e MODEL=$SMOL $FIX" l40sx1 1h ;;
  smol-r0) job "-e MODEL=$SMOL -e REPO_NAME=gg-smollm2-r0-s7" ;;          # R0 = TRL defaults (bnpo, beta .04)
  smol-r1) job "-e MODEL=$SMOL -e REPO_NAME=gg-smollm2-r1-s7 $FIX" ;;
  qwen-r0) job "-e MODEL=$QWEN -e REPO_NAME=gg-qwen15b-r0-s7" ;;
  qwen-r1) job "-e MODEL=$QWEN -e REPO_NAME=gg-qwen15b-r1-s7 $FIX" ;;
  dl)
    : "${NS:?set NS to your HF username: export NS=\$(uvx --from huggingface_hub hf auth whoami | head -1)}"
    mkdir -p runs_dl
    for r in $REPOS; do uvx --from huggingface_hub hf download "$NS/$r" --local-dir "runs_dl/$r"; done
    python src/eval/aggregate.py --out results runs_dl/*/ ;;
  *) sed -n '2,16p' "$0" ;;
esac
