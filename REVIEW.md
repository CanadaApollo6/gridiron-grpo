# Technical review — gridiron-grpo as a controlled RLVR study

_An outside audit of the experiment design, the code, and the claims, with prioritized
corrections. Written against commit `0381ccb`. Numbers below were recomputed from the
generators (seed 7, n=8000) and parsed from `job_15b.log`, not taken on faith._

---

## Bottom line

The infra, the diagnosis, and the literature review are solid and honest — better than
most practitioner write-ups. The pivot to a multi-family study is the right call and the
right framing. **But the headline contribution you're betting on (the compositional-depth
taxonomy, Q3) is currently confounded by task design, and your per-task deltas are inside
the noise floor.** Two of six tasks have majority-class baselines that make "accuracy went
up/down" uninterpretable, and at n≈130 per task a single run can't distinguish a real 2pp
effect from sampling noise. Fix those two things _before_ spending the Phase 2 budget, or
the matrix produces a pretty table you can't defend.

Everything else is enhancement. These two are blockers.

---

## What's genuinely strong (keep doing it)

- **The validate-cheaply-in-layers discipline.** CPU → 3080/WSL → $0.42 of `SMOKE_ONLY`
  A100 pre-flights catching bugs #4–6 is exactly right. Keep it.
- **The diagnosis is real and I reproduced it.** From `job_15b.log` (122 logged steps):
  correctness is dead flat (first-quartile mean **0.290**, last-quartile **0.293**; range
  0.197–0.394 is noise), while KL climbs **0.038 → 0.666, peak 1.237** (~33×, not 15×) and
  crosses 0.5 at ~42% through the run. `grad_norm` drifts 8.8 → 64.6. This is a textbook
  "drift without learning" signature. Your read is correct.
- **The lit review checks out — every citation is real and fairly characterized.** I
  verified all four, including the one that looks fake: `arxiv 2602.05494` is a genuine
  Feb 2026 paper ("A Unified Framework for Rethinking Policy Divergence Measures in GRPO").
  See the citation table at the bottom.
- **The honesty.** Acknowledging the Qwen confound invalidates your own first result is the
  thing that makes the rest credible. Don't lose that voice in the write-up.

---

## P0 — Blockers (fix before Phase 2)

### 1. Majority-class confound contaminates the taxonomy (Q3)

Two of your six tasks are label-imbalanced enough that a model can move the accuracy number
without doing any reasoning:

| Task | Imbalance (n=8000 draws) | Naive baseline | What "improvement" might really be |
| --- | --- | --- | --- |
| `td_or_fg` | TD 983 / FG 328 | **75.0%** always-TD | Drift toward the majority class, not a learned decision |
| `hundred_yd_rec` | 38% of answers are `none` | **62.1%** always-`none` | Collapse to/from the empty set |

This is not hypothetical — it's exactly how `td_or_fg` ended up as your "easy task that
improved." `score_diff` is drawn uniformly from `{-8..-1}`, and FG only wins when the
deficit is 1 or 2, so the label is 75/25 by construction. A policy that quietly shifts its
prior toward "TD" gains on this task **and that's indistinguishable from reasoning in your
current metric.** The set task has the same problem in reverse.

Because Q3 is your stated contribution, this confound sits directly under the headline.

**Fix (do all three):**
- Add a **majority-class baseline column** to every results table. A delta is only
  interesting relative to that floor, not to 0%.
- Report **per-class** accuracy (recall for TD and FG separately; precision/recall on the
  set task), not just overall accuracy, for `td_or_fg` and `hundred_yd_rec`.
- **Rebalance the generators.** Draw `score_diff` so FG/TD is ~50/50 (sample the deficit
  bucket, then a value within it). Tune the receiving-yards distribution / threshold so the
  set task is ~50% non-empty. Cheap, and it makes the taxonomy interpretable.

### 2. You're measuring the wrong "headroom" — measure base pass@k per task first

Act III concludes "the strong 1.5B base had no headroom." Overall 57.5% pass@1 is _not_ no
headroom — it's 42 points from ceiling. The real constraint is sharper and more useful:
**GRPO can only reinforce what the base model already samples with nonzero probability.**
If base pass@8 on a task is ≈0, no recipe — Dr. GRPO, DAPO, any beta — can teach it, because
every group is all-zero-reward and the advantage is identically zero. Your rising
`frac_reward_zero_std` (0 → **0.15**) is this happening in real time.

So the single most decisive, cheapest measurement you can make — and you haven't yet — is
**base-model pass@k (k=1,8,64) per task kind, before any training.** It predicts which cells
of the Phase 2 matrix can move at all. Tasks with base pass@8 ≈ 0 are unlearnable-by-GRPO
and should be reported as such, not as "GRPO failed." This reframes Q1 from "did it fail?"
to "was there anything to learn?" — a much stronger claim. **Promote this from Phase 3 to
Phase 1.**

---

## P1 — Rigor gaps (will sink the write-up if unaddressed)

### 3. Single-run per-task deltas are inside the noise floor

`n_eval=800` → ~130 per kind. A 95% Wilson interval at p≈0.5, n=130 is **±8.6pp**. Your
observed per-task deltas (−1 to +2pp) and overall deltas (−1.2 to +0.9pp) are all smaller
than that. As stated in `EXPERIMENTS.md`, "single seed is not a result" — agreed, but it's
deeper than seeds: even one seed's per-task numbers need intervals. Before any claim:
- Put **Wilson CIs** on every cell (overall and per-kind).
- Use **McNemar's test** for the base-vs-tuned overall delta (same eval items, paired — far
  more powerful than treating them as independent).
- **3+ seeds** on decisive cells, and vary the **data** seed on a subset, not just the
  training seed (right now `hf_job.sh` hardcodes `--seed 7` for data, so multi-seed only
  captures optimization noise, not data noise — a defensible control, but state it).

### 4. The "fixes" may be under-powered to show learning even where it exists

Before concluding Dr. GRPO/DAPO don't help in this domain, rule out that the optimizer is
just too timid:
- **LR.** Peak is `1e-6` and the schedule decays to **8.3e-10** by the end — the tail of the
  run contributes almost nothing, and 1e-6 is low for LoRA (most GRPO-LoRA recipes run
  1e-5–3e-5). The schedule type isn't pinned in `GRPOConfig`. Make it explicit and sweep
  peak LR — an under-powered optimizer and "RLVR doesn't help here" look identical.
- **beta.** You diagnosed KL runaway, yet every recipe (R0–R3) keeps `beta=0.04`. DAPO and
  Dr. GRPO typically run **beta=0**. A `beta ∈ {0, 0.04, 0.1, 0.2}` arm is the most direct
  test of the "is this a KL-discipline problem" branch and it's currently missing from R0–R2.
  You list a beta sweep in Phase 3; it belongs next to the decisive run.

### 5. Reward sparsity / zero-advantage groups

Correctness mean ~0.29 with 8 generations and a hard 0/1 reward means a lot of groups are
all-wrong or all-right → zero advantage → no gradient, and it's getting worse over training
(`frac_zero_std` → 0.15). Two cheap levers, both already in the literature you cite:
- **Dynamic sampling** (DAPO): over-sample and drop prompts whose group has zero advantage,
  so every gradient step carries signal.
- **Partial-credit reward on the numeric tasks** (e.g. reward decreasing in |pred−gt|, or a
  bucketed tolerance) to densify the signal instead of the all-or-nothing 1e-6 match. Keep a
  strict-correctness metric for _eval_; only soften the _training_ reward. Worth an ablation
  arm of its own.

---

## P2 — Worth doing, not blocking

### 6. Eval realism and the untested generalization claim

All eval is in-distribution synthetic. The README sells "trained on synthetic, evaluated on
real games" and domain-agnosticism (invoices/telemetry) — both currently **claimed but
unbuilt**. A small real-NFL holdout (`nfl_data_py`, formatted to match the generator) is
high-credibility and cheap, and it's the honest version of the headline. Build at least one
non-football generator to back the "swap the data layer" claim, or soften it.

### 7. Code / telemetry nits (each is small; the telemetry one matters for Q1)

- **Length/termination telemetry is broken in colocate mode.** In `job_15b.log`,
  `completions/{min,mean,max}_length` are all identical per step, `mean_terminated_length`
  is 0, and `clipped_ratio` is pinned at 1.0. You already flagged `clipped_ratio` as
  misleading — but "non-terminating completions" is one of your Q1 signals, and the native
  metric can't measure it. Log a real **terminated fraction** yourself (count EOS in a held-
  out sample each eval) so the claim rests on a number you trust.
- **Eval token cap drift.** `evaluate.py` defaults `--max_new_tokens=512` and `run_eval.sh`
  never passes it, while the study design specifies a 1024 completion budget. `hf_job.sh`
  wires it correctly, but the standalone path silently truncates. Align defaults with
  `EXPERIMENTS.md` (and make `train_grpo.py`'s `max_completion_len` default 1024 too, so you
  don't under-run the study by forgetting an env var).
- **Name matching is brittle.** `_check` for `name` answers requires an exact normalized
  match, so a model answering "Smith" when GT is "A. Smith" scores 0. Either make the
  generator guarantee unique last names per box (so last-name answers are well-posed) or
  accept a last-name match. Right now you may be undercounting `most_scrimmage`.
- **Physically inconsistent boxes.** Team-scoring TDs are an independent draw from players'
  rush/rec TDs in the same box, so `team_points` and `total_tds` describe contradictory
  games. It doesn't break verifiability, but a model that reasons about football can be
  penalized for trying to reconcile them. Either derive team points to be consistent with
  player TDs, or note the independence explicitly in the prompt.
- **`most_scrimmage` tie fallback** silently re-labels the sample as `team_points`, nudging
  the realized distribution. Re-draw instead of switching kind.

### 8. Bookkeeping

Log the **full recipe** (every env var / resolved `GRPOConfig`) into each `results/*.json`,
not just into the run name. When the Phase 2 aggregator pulls 8+ Hub repos, you'll want the
config self-describing in the artifact, not encoded in a string you have to parse.

---

## A sharper next-step sequence (replaces "Phase 0 → 1 → 2" ordering)

The point of reordering is to front-load the cheap measurements that can _kill or redirect_
the expensive matrix:

1. **Rebalance the two confounded generators** + add majority-class baseline and per-class
   reporting. (Code only, ~$0.)
2. **Base pass@k per task, per family** (k=1,8,64, greedy + sampled). (~$1–2 of eval, no
   training.) → tells you which matrix cells can move at all.
3. **One decisive run** on a non-Qwen base (SmolLM2 or OLMo-SFT), R0 vs best-fix, **with a
   beta sweep and a higher peak LR**, reported with CIs and McNemar. → answers Q1+Q2 with a
   defensible number before you commit Phase 2 budget.
4. Only then the **family × recipe matrix**, multi-seed on the cells that step 2 says are
   reachable.

This turns "we ran a big matrix and mostly saw flat" into "we predicted which cells could
move, then showed which recipe moves them" — which is a result, not a null.

---

## Citations — all verified real and fairly used

| Claim in your docs | Source | Status |
| --- | --- | --- |
| GRPO length bias → Dr. GRPO fix | [Liu et al. 2025, arXiv:2503.20783](https://arxiv.org/abs/2503.20783) | ✓ correct |
| Qwen confound (spurious/random rewards "work" on Qwen, fail off-Qwen) | [Spurious Rewards, OpenReview 4NeiwxQ2Bp](https://openreview.net/forum?id=4NeiwxQ2Bp) | ✓ correct |
| RLVR amplifies base capability; narrows pass@k | [Yue et al. 2025, arXiv:2504.13837](https://arxiv.org/abs/2504.13837) (= the "Limit of RLVR" project page; one paper, not two) | ✓ correct |
| KL blow-up / policy-divergence instability | [arXiv:2602.05494](https://arxiv.org/abs/2602.05494) | ✓ real (Feb 2026), correctly used |

One nit: README/JOURNEY cite the Yue OpenReview link _and_ `limit-of-rlvr.github.io` as if
they were independent corroboration — they're the same work. Fine to cite both (paper +
project page), just don't lean on it as two sources.
