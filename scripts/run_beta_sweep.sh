#!/usr/bin/env bash
# KL-discipline sweep: is "drift without learning" a KL problem? Run R0 at several
# beta values on ONE non-Qwen base and compare. beta=0 = no KL anchor (DAPO/Dr.GRPO
# style); higher beta pins the policy closer to the base. (See REVIEW.md / Q-beta.)
#   bash scripts/run_beta_sweep.sh <model>
# Env: BETAS="0 0.04 0.1 0.2"  MAX_STEPS=1200  EXTRA="--loss_type dr_grpo --no_scale_rewards"
set -euo pipefail
MODEL="${1:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
STEPS="${MAX_STEPS:-1200}"
BETAS="${BETAS:-0 0.04 0.1 0.2}"

python src/data/build_dataset.py --n_train "${N_TRAIN:-8000}" --n_eval "${N_EVAL:-800}" --seed 7 --out data_out
python src/eval/evaluate.py --model "$MODEL" --data data_out/eval.jsonl --label baseline --max_new_tokens 1024
for b in $BETAS; do
  out="runs/beta_${b}"
  echo "================= beta=$b ================="
  torchrun --nproc_per_node 1 src/train_grpo.py --model "$MODEL" --data data_out/train.jsonl \
     --out "$out" --max_steps "$STEPS" --beta "$b" ${EXTRA:-}
  python src/eval/evaluate.py --model "$MODEL" --adapter "$out" \
     --data data_out/eval.jsonl --label "grpo_beta_${b}" --max_new_tokens 1024
  python src/eval/compare.py results/baseline.json "results/grpo_beta_${b}.json" || true
done
echo "sweep done; compare results/comparison*.md and the KL trajectories in each run log."
