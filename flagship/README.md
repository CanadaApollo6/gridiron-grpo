# Flagship (Paper 4) — the measure-zero blind spot: one task, two optimizers

Unifies the clock-kill *deductive residual* (Paper 1) and the GRPO *composite-arithmetic
wall* (Paper 3) as **one phenomenon**: expectation-optimizers (supervised average-loss AND
RL expected-reward) do not localize a low-measure, sharp, deductively-decidable cell. Spec:
[`../FLAGSHIP_SPEC.md`](../FLAGSHIP_SPEC.md).

## What's here
- `task.py` — the synthetic task, torch-free. Two instantiations sharing one cell structure:
  **binary** (`make_dataset`: bulk `[u≥v]`, sharp override band on `v`) for the supervised/MLP
  probe; **mod-arithmetic** (`make_modarith`: bulk `(a+b)%M`, override `(a*b)%M` on a thin
  `(a−b)` band) for the torch SFT+GRPO harness — its large answer space is what makes GRPO's
  *exploration*-blindness bite.
- `probe_supervised.py` — **numpy, runs anywhere in ~30s.** The go/no-go for the supervised
  half. Already green:

| ε | bulk acc | cell recovery |
|---|---|---|
| 0.20 | 0.86 | **1.00** |
| 0.10 | 0.98 | **0.52** |
| 0.05 | 0.99 | **0.34** |
| 0.02 | 0.99 | **0.31** |
| 0.01 | 0.99 | **0.30** |

Cell recovery collapses to the bulk floor (~0.30) while bulk acc stays ~0.99 — **with the
trigger feature handed over** (so it's not representation; cell-only training recovers 1.0).
See `fig_supervised_collapse.png`.

- `flagship.py` — **torch, needs a CUDA GPU** (your 3080; minutes/cell). Tiny char-GPT, SFT
  (average loss) and GRPO (verifiable 0/1 reward) from a shared bulk-only warm-start, on the
  mod-arithmetic task. Produces the *overlay*: does GRPO collapse the same way SFT does?

## Run (on the 3080, in WSL)
```bash
pip install torch numpy matplotlib
# 0) validate the harness in seconds
python flagship.py warmstart --out runs/ws --smoke
python flagship.py grpo --init runs/ws --eps 0.05 --arm raw --smoke
# 1) real shared warm-start (SFT on bulk only)
python flagship.py warmstart --out runs/ws
# 2) the matrix: eps x {sft,grpo} x arm  (raw|trigger|region|cell ; explore = grpo only)
for eps in 0.20 0.10 0.05 0.02 0.01; do for obj in sft grpo; do for arm in raw region cell; do
  python flagship.py $obj --init runs/ws --eps $eps --arm $arm --log results.jsonl
done; done; done
python flagship.py grpo --init runs/ws --eps 0.02 --arm explore --log results.jsonl  # exploration-fix control
```
Then plot cell-recovery vs ε with SFT and GRPO overlaid (the headline figure). 3+ seeds with
the gridiron `src/eval/stats.py` CIs before any claim.

## What to look for (predictions)
- **P1** both SFT and GRPO cell-recovery → bulk floor as ε→0 (curves collapse together).
- **P2** `--arm trigger` (exact threshold feature) still collapses → not representation.
- **P3** `--arm region` drifts to region average, not the answer.
- **P4** `--arm cell` recovers (+ a near-cell **bleed** in the `bleed` metric — the clock-kill
  overshoot reproduced here).
- **mechanism split** `--arm explore` (GRPO) recovers → GRPO's failure was *exploration*;
  SFT's was *measure/smoothing*. One blind spot, two faces.

> `flagship.py` is untested in-sandbox (no GPU here). Validate with `--smoke` first; the GRPO
> loss and tiny-GPT are standard but eyeball the first smoke run before the full sweep.
