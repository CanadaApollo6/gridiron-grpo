# When does RLVR help on structured/tabular verifiable reasoning, across model families?

A controlled study using the gridiron-grpo pipeline. This file is the experimental
design (and the scaffold for a write-up). It is intentionally opinionated about
_controls_, because the contribution is rigor in a setting the existing literature
skips, not a new algorithm. The methodology hardening below is tracked in
[`REVIEW.md`](REVIEW.md).

## Motivation & gap

RL with verifiable rewards (RLVR / GRPO) is now well-studied, but almost entirely on
**math and code, with Qwen models**. Three facts from that literature frame this study:

1. **GRPO has a length bias** that inflates response length (esp. for wrong answers);
   the fix is the unbiased _Dr. GRPO_ objective. ([Liu et al., 2025](https://arxiv.org/abs/2503.20783))
2. **RLVR mostly amplifies latent base-model capability** rather than teaching new
   skills, and can _narrow_ exploration (better pass@1, worse pass@k).
   ([Yue et al., 2025](https://arxiv.org/abs/2504.13837) / [Limit of RLVR](https://limit-of-rlvr.github.io/))
3. **The Qwen confound:** on Qwen, even _random/incorrect_ rewards "work," because RLVR
   elicits pretrained behaviors; the same rewards fail on Llama/OLMo. Any Qwen-only
   result is therefore not credible. ([Spurious Rewards](https://openreview.net/forum?id=4NeiwxQ2Bp))

**Under-explored:** structured/tabular verifiable reasoning (box scores, invoices,
telemetry) — short contexts, exact answers, decomposable into task types of varying
compositional depth. We ask **when RLVR helps there**, with the controls the field skips.

## Research questions

- **Q1 (confound).** Does the "drift without learning" failure we observed on Qwen
  (KL ↑~30×, accuracy flat, completions non-terminating) **replicate off-Qwen**?
- **Q2 (fixes).** Do the published fixes — **Dr. GRPO** (`loss_type=dr_grpo`,
  `scale_rewards=False`) and **DAPO** stability (`mask_truncated_completions`,
  `epsilon_high=0.28`, dynamic sampling) — **recover learning** in this domain, across
  families?
- **Q3 (taxonomy).** Does GRPO's per-task effect track **compositional depth** —
  measured **against each task's naive baseline, with CIs** — and is that pattern
  **consistent across families**?
- **Q-learnability (gating).** Per task and family, what is the **base model's
  pass@k**? GRPO can only reinforce what the base already samples, so a task with base
  pass@8 ≈ 0 is unlearnable by GRPO regardless of recipe. This is measured _first_ and
  gates the rest.
- **Q-KL (mechanism).** Is the instability a KL-discipline problem? A `beta` sweep
  (0 → 0.2) tests whether a stronger/zero anchor changes the drift picture.

## Models (≈1–1.7B instruct; size-matched-ish)

| Family    | HF id                                 | Access | Notes                                                             |
| --------- | ------------------------------------- | ------ | ----------------------------------------------------------------- |
| Qwen2.5   | `Qwen/Qwen2.5-1.5B-Instruct`          | open   | the confounded reference                                          |
| Llama 3.2 | `meta-llama/Llama-3.2-1B-Instruct`    | gated  |                                                                   |
| SmolLM2   | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | open   |                                                                   |
| OLMo 2    | `allenai/OLMo-2-0425-1B-Instruct`     | open   | already RLVR'd on Tülu 3 → also test `...-1B-SFT` as a clean base |

LoRA uses `target_modules="all-linear"` so the adapter recipe is identical regardless
of per-architecture module names.

## Task taxonomy (compositional depth)

Six verifiable kinds from `src/data/tasks.py`, ordered low→high by how many table rows
and ops the answer requires. `depth` ships on every row (`KIND_DEPTH`) so eval groups by it.

| Kind              | Type     | Depth | Naive floor\* | Why                                          |
| ----------------- | -------- | :---: | :-----------: | -------------------------------------------- |
| `td_or_fg`        | decision |   1   |     ~52%      | one comparison (deficit < 3); commit-early   |
| `scrimmage_total` | numeric  |   2   |      low      | locate one player, add 2 fields              |
| `total_tds`       | numeric  |   3   |      low      | sum a field across all rows                  |
| `most_scrimmage`  | argmax   |   3   |      low      | per-row total, then argmax                   |
| `team_points`     | numeric  |   4   |      low      | sum player TDs (×6) + combine FG/XP/2pt line |
| `hundred_yd_rec`  | set      |   4   |     ~29%      | filter all rows by threshold → set           |

\*Naive floor = accuracy of always guessing the kind's majority answer (reported by
`evaluate.py` as `best_constant_by_kind`). Every Δ is judged against this, never against 0.

**Data invariants (hardening pass — see REVIEW.md):**

- `td_or_fg` deficit is sampled ~50/50, so the decision can't be won by always
  answering "TD" (was 74% TD).
- Boxes are **physically consistent**: team points derive from the players' own TDs +
  the FG/XP/2pt line (was contradictory in ~86% of boxes). The aggregate TD count is not
  printed, so `total_tds`/`team_points` still require reading the player column.
- **Last names are unique per box**, so "which player" is well-posed; the eval checker
  accepts an unambiguous last-name answer.
- Receiving production nudged up so the set task isn't dominated by the empty set
  (`none` ~29%, was ~38%).

**Q3 hypothesis:** Δaccuracy (tuned − base, **CI-bounded, vs. floor**) is positive/neutral
at low depth and negative at high depth, with the same ordering across families.

## Recipes (the objective axis)

| ID               | Config                                                            | Purpose                                               |
| ---------------- | ----------------------------------------------------------------- | ----------------------------------------------------- |
| **R0 naive**     | `loss_type=bnpo` (TRL default), `scale_rewards=True`, `beta=0.04` | what a typical user gets; the failing baseline        |
| **R1 drgrpo**    | `loss_type=dr_grpo`, `NO_SCALE_REWARDS=1`, `beta=0.04`            | Dr. GRPO core (length-bias + difficulty-bias fix)     |
| **R2 +dapoclip** | R1 + `MASK_TRUNCATED=1`, `EPSILON_HIGH=0.28`                      | DAPO stability (mask-truncated + asymmetric clip)     |
| **R3 corr-only** | best recipe + `NO_FORMAT_REWARD=1`                                | reward ablation (is the format bonus a trap here?)    |
| **R4 graded**    | best recipe + `GRADED_NUMERIC=1`                                  | partial-credit numeric reward (densify sparse signal) |
| **β-sweep**      | R0 with `BETA ∈ {0, 0.04, 0.1, 0.2}`                              | is the instability a KL-discipline problem?           |

Shared: 1200 steps, 1024 completion budget, num_generations=8, temp 0.9, LoRA r=16,
fixed left-padded length-matched eval, n_train=8000 / n_eval=800, seed=7. **LR schedule
is a knob** (`--lr_scheduler_type`, `--warmup_ratio`): the default `cosine` decays to ~0,
so for a fair "does it learn" probe also try `constant_with_warmup` and a higher peak LR
(1e-6 is low for LoRA) — an under-powered optimizer and "RLVR doesn't help" look identical.

> **Termination is a prerequisite for `mask_truncated`.** If rollouts never emit a stop
> token (e.g. vLLM not honoring Qwen2.5-Instruct's `<|im_end|>` -> `clipped_ratio=1.0`),
> masking truncated completions zeroes the batch -> `grad_norm=0` / NaN KL -> a silent
> no-op adapter. `train_grpo.py` now passes each model's EOS ids as vLLM stop tokens and
> aborts on a no-update run; `mask_truncated` is confined to R2.

## Metrics

- **Primary:** Δaccuracy overall and **per kind**, each with a **Wilson 95% CI**, judged
  against the **naive floor** (`best_constant_by_kind`). Base vs tuned is paired, so the
  overall change is tested with **McNemar's exact test** (`src/eval/compare.py`).
- **Per-class (collapse check):** for the decision and set tasks, per-class recall and the
  model's predicted-class distribution — to catch majority-class collapse masquerading as
  learning (`per_class` in the results JSON).
- **Learnability:** base **pass@1/8/64** per kind (`src/eval/pass_at_k.py`), plus the
  fraction of items the base never solves (`frac_never`) — the headroom GRPO can/can't use.
- **Mechanistic (from train logs):** KL trajectory, **measured terminated fraction**
  (eval-side, because the colocate length telemetry is unreliable), reward/correctness
  trajectory, grad_norm.
- **Narrowing check:** sampled pass@k base vs tuned — does RLVR raise pass@1 but lower
  pass@k, per Yue et al.?

## Phases & rough budget (A100-large @ $2.50/hr; ~$5–6 per full run)

Reordered so the cheap measurements that can kill or redirect the matrix come first.

- **Phase 0 — infra pilots (~$1):** `SMOKE_ONLY=1` per family to confirm each loads,
  trains (chat template, pad/eos), and vLLM-colocates.
- **Phase 1 — learnability + one decisive run (~$6):**
  (a) **base pass@k per task, per family** (`PASSK=1`, no training) → which cells can move;
  (b) **R0 vs R1 on one non-Qwen** (SmolLM2 or OLMo-SFT) on the **rebalanced** data, with a
  **β-sweep** and a higher-LR arm, reported with **CIs + McNemar**. Answers Q1, Q2-first,
  Q-learnability, Q-KL before any matrix spend.
- **Phase 2 — core matrix (~$40):** {4 families} × {R0, R1} on the cells Phase 1 flagged as
  reachable → the headline family×recipe×kind table.
- **Phase 3 — depth & rigor (~$60–100):** R2/R3/R4 on best families; **2–3 seeds** (vary the
  **data** seed too, not just training seed) on key cells; pass@k narrowing; a second-domain
  generator (invoices/telemetry) to back the domain-agnostic claim. **Multi-seed is required
  before any claim.**

## Result bookkeeping

Each run writes `runs/<out>/recipe.json` (full resolved args + GRPOConfig + reward fns), and
pushes to a Hub repo `gg-{family}-{recipe}-s{seed}` (via `REPO_NAME`) containing
`results/baseline.json`, `results/grpo.json`, `results/comparison.md` (CIs + McNemar),
`results/table.md`, and the training log. A `scripts/aggregate_results.py` (Phase 2) pulls all
repos into one family×recipe×kind table + figures; because each run is self-describing
(`recipe.json` + per-item `items`), the aggregator never has to parse run names.

## Threats to validity (state them up front)

- **Single seed** is not a result → multi-seed key cells (training _and_ data seed) before
  claiming anything.
- **Qwen confound** → the whole multi-family design; report per-family, never pooled-only.
- **OLMo-Instruct is already RLVR'd** → also run the `-SFT` base.
- **LoRA, not full FT** → scope claims to the LoRA regime (note it explicitly).
- **Synthetic + in-distribution eval** → add a real-stat-line held-out eval (nfl_data_py) for
  the headline, and the second-domain generator for generalization.
- **Imbalanced tasks** → addressed by rebalancing + reporting every Δ against the naive floor
  with per-class breakdowns. (Note: the set task's exploitable floor was ~38%, not the ~62%
  first estimated — the 62% was the non-empty fraction, which isn't constant-exploitable.)
- **Greedy pass@1 eval** → pass@k added to test the narrowing hypothesis.
- **Size spread (1B–1.7B)** → acknowledge; size-match where possible.
