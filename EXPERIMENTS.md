# When does RLVR help on structured/tabular verifiable reasoning, across model families?

A controlled study using the gridiron-grpo pipeline. This file is the experimental
design (and the scaffold for a write-up). It is intentionally opinionated about
*controls*, because the contribution is rigor in a setting the existing literature
skips, not a new algorithm.

## Motivation & gap

RL with verifiable rewards (RLVR / GRPO) is now well-studied, but almost entirely on
**math and code, with Qwen models**. Three facts from that literature frame this study:

1. **GRPO has a length bias** that inflates response length (esp. for wrong answers);
   the fix is the unbiased *Dr. GRPO* objective. ([Liu et al., 2025](https://arxiv.org/abs/2503.20783))
2. **RLVR mostly amplifies latent base-model capability** rather than teaching new
   skills, and can *narrow* exploration (better pass@1, worse pass@k).
   ([Yue et al.](https://openreview.net/forum?id=4OsgYD7em5), [Limit of RLVR](https://limit-of-rlvr.github.io/))
3. **The Qwen confound:** on Qwen, even *random/incorrect* rewards "work," because RLVR
   elicits pretrained behaviors; the same rewards fail on Llama/OLMo. Any Qwen-only
   result is therefore not credible. ([Shao et al., Spurious Rewards](https://openreview.net/forum?id=4NeiwxQ2Bp))

**Under-explored:** structured/tabular verifiable reasoning (box scores, invoices,
telemetry) — short contexts, exact answers, decomposable into task types of varying
compositional depth. We ask **when RLVR helps there**, with the controls the field skips.

## Research questions

- **Q1 (confound).** Does the "drift without learning" failure we observed on Qwen
  (KL ↑15×, accuracy flat, completions non-terminating) **replicate off-Qwen**?
- **Q2 (fixes).** Do the published fixes — **Dr. GRPO** (`loss_type=dr_grpo`,
  `scale_rewards=False`) and **DAPO** stability (`mask_truncated_completions`,
  `epsilon_high=0.28`) — **recover learning** in this domain, across families?
- **Q3 (taxonomy).** Does GRPO's per-task effect track **compositional depth** —
  helping low-depth "commit-early" tasks and hurting high-depth multi-step/set tasks —
  and is that pattern **consistent across families**?

## Models (≈1–1.7B instruct; size-matched-ish)

| Family | HF id | Access | Notes |
|---|---|---|---|
| Qwen2.5 | `Qwen/Qwen2.5-1.5B-Instruct` | open | the confounded reference |
| Llama 3.2 | `meta-llama/Llama-3.2-1B-Instruct` | **gated** (accept Meta license w/ token account) | |
| SmolLM2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | open | |
| OLMo 2 | `allenai/OLMo-2-0425-1B-Instruct` | open | already RLVR'd on Tülu 3 → also test `...-1B-SFT` as a clean base |

LoRA uses `target_modules="all-linear"` so the adapter recipe is identical regardless
of per-architecture module names.

## Task taxonomy (compositional depth)

Six verifiable kinds from `src/data/tasks.py`, ordered low→high by how many table rows
and ops the answer requires:

| Kind | Type | Depth | Why |
|---|---|---|---|
| `td_or_fg` | decision | **low** | one comparison (deficit < 3); commit-early |
| `scrimmage_total` | numeric | low–mid | locate one player, add 2 fields |
| `team_points` | numeric | mid | combine 4 scoring components |
| `total_tds` | numeric | high | sum a field across all rows |
| `most_scrimmage` | argmax | high | per-row total, then argmax |
| `hundred_yd_rec` | set | **high** | filter all rows by threshold → set |

**Q3 hypothesis:** Δaccuracy (tuned − base) is positive/neutral at low depth and
negative at high depth, with the same ordering across families.

## Recipes (the objective axis)

| ID | Config | Purpose |
|---|---|---|
| **R0 naive** | `loss_type=bnpo` (TRL default), `scale_rewards=True`, `beta=0.04` | what a typical user gets; the failing baseline |
| **R1 drgrpo** | `loss_type=dr_grpo`, `NO_SCALE_REWARDS=1`, `MASK_TRUNCATED=1`, `beta=0.04` | the published length-bias + stability fixes |
| **R2 +dapoclip** | R1 + `EPSILON_HIGH=0.28` | add DAPO asymmetric clipping |
| **R3 corr-only** | best recipe + `NO_FORMAT_REWARD=1` | reward ablation (is the format bonus a trap here?) |

Shared: 1200 steps, 1024 completion budget, num_generations=8, temp 0.9, LoRA r=16,
fixed left-padded length-matched eval, n_train=8000 / n_eval=800, seed=7.

## Metrics

- **Primary:** Δaccuracy overall and **per kind** (base vs tuned, greedy pass@1).
- **Mechanistic (from train logs):** KL trajectory, completion mean length + terminated
  fraction, reward/correctness trajectory, grad_norm. (Parse via the same approach as
  `job_15b.log` analysis.)
- **Narrowing check (Q-secondary):** sampled **pass@k** (k=1,8,64) base vs tuned — does
  RLVR raise pass@1 but lower pass@k, per Yue et al.? (eval extension, Phase 3)

## Phases & rough budget (A100-large @ $2.50/hr; ~$5–6 per full run)

- **Phase 0 — infra pilots (~$1):** `SMOKE_ONLY=1` per family to confirm each loads,
  trains (chat template, pad/eos), and vLLM-colocates.
- **Phase 1 — decisive (~$5):** R0 on one non-Qwen (SmolLM2 or OLMo) → does the failure
  replicate off-Qwen (Q1)?
- **Phase 2 — core matrix (~$40):** {4 families} × {R0, R1} → the headline table (Q1+Q2).
- **Phase 3 — depth (~$60–100):** R2/R3 on best families; **2–3 seeds** on key cells;
  `beta` sweep (KL discipline); pass@k; generalization domain (invoices/telemetry
  generator). Multi-seed is required before any claim.

## Result bookkeeping

Each run pushes to a Hub repo named `gg-{family}-{recipe}-s{seed}` (set via `REPO_NAME`),
containing `results/baseline.json`, `results/grpo.json`, `results/table.md`, and the
training log. A `scripts/aggregate_results.py` (Phase 2) pulls all repos into one
family×recipe×kind table + figures.

## Threats to validity (state them up front)

- **Single seed** is not a result → multi-seed key cells before claiming anything.
- **Qwen confound** → the whole multi-family design; report per-family, never pooled-only.
- **OLMo-Instruct is already RLVR'd** → also run the `-SFT` base.
- **LoRA, not full FT** → scope claims to the LoRA regime (note it explicitly).
- **Synthetic data** → add a real-stat-line held-out eval (nfl_data_py) for the headline.
- **Greedy pass@1 eval** → add pass@k to test the narrowing hypothesis.
- **Size spread (1B–1.7B)** → acknowledge; size-match where possible.
