# Findings — running results log

Empirical results as they land. Design lives in [`EXPERIMENTS.md`](EXPERIMENTS.md);
methodology rationale in [`REVIEW.md`](REVIEW.md); narrative in [`JOURNEY.md`](JOURNEY.md).

---

## 2026-06-20 — Phase 1: base-model learnability (pass@k), SmolLM2-1.7B

**Setup.** `HuggingFaceTB/SmolLM2-1.7B-Instruct`, base (no adapter). First 60 items of
the seed-7 eval set, 64 samples/item, temperature 0.9, `max_new_tokens=384`. Unbiased
pass@k (Chen et al. 2021); strict verifiable check (same as the training reward). Run
locally on an RTX 3080 (10GB) via `scripts/run_local.sh passk`. One seed, one model —
**directional, not final**.

**Why this run.** GRPO can only reinforce what the base already samples, so a task with
base pass@8 ≈ 0 is unlearnable by GRPO regardless of recipe. Measuring this _before_ the
recipe matrix tells us which cells can move (the Q-learnability gate).

### Per-task headroom (sorted by GRPO-exploitable headroom)

| Task              | depth | type        | pass@1 |    pass@8 |   pass@64 | frac_never |  n  | GRPO outlook       |
| ----------------- | :---: | ----------- | -----: | --------: | --------: | ---------: | :-: | ------------------ |
| `td_or_fg`        |   1   | decision    |  12.2% | **60.7%** |  **100%** |         0% |  6  | strong gain likely |
| `most_scrimmage`  |   3   | argmax/name |   5.7% | **35.4%** | **88.2%** |        12% | 17  | strong gain likely |
| `hundred_yd_rec`  |   4   | set         |   1.6% |     10.7% |     30.0% |        70% | 10  | capped low         |
| `team_points`     |   4   | numeric     |   0.5% |      3.8% |     30.0% |        70% | 10  | capped low         |
| `scrimmage_total` |   2   | numeric     |   3.4% |     17.5% |     28.6% |        71% |  7  | capped low         |
| `total_tds`       |   3   | numeric     |   0.5% |      3.6% |     20.0% |        80% | 10  | weakest            |
| **Overall**       |   —   | —           |   3.7% |     21.2% |     51.7% |        48% | 60  | —                  |

### Key finding: the wall is _arithmetic_, not depth

The dividing line tracks **task type, not compositional depth**. `scrimmage_total` is
depth-2 (our "low") yet capped just like the depth-4 tasks (pass@64 ≈ 29%), while
`most_scrimmage` is depth-3 yet highly reachable (pass@64 ≈ 88%). The split is clean:

- **Reachable (select / decide):** `td_or_fg` (pick TD vs FG) and `most_scrimmage`
  (argmax a row, copy the name). Big pass@1→pass@64 gaps with high ceilings — the base
  _can_ produce the answer, just rarely. Ideal RLVR targets.
- **Arithmetic wall:** every task requiring multi-step computation — sum a column
  (`total_tds`), combine a scoring line (`team_points`), add two fields
  (`scrimmage_total`), threshold-filter to a set (`hundred_yd_rec`) — has `frac_never`
  70–80%. The base **never** produces a correct rollout for most of these items.

So SmolLM2-1.7B base can _locate, select, and decide_ but cannot reliably _compute_.
This refines **Q3**: the operative axis is select/decide vs multi-step arithmetic, which
**cross-cuts** the depth labels — a stronger, more defensible taxonomy than depth alone.

### Falsifiable prediction for the R0/R1 training runs

GRPO can only sharpen what's already sampled, so:

- **Gains should concentrate on `td_or_fg` + `most_scrimmage`** (high pass@8 = in-group
  signal to reinforce).
- **The four arithmetic/set tasks should stay near their low pass@64 ceilings** — GRPO
  has almost nothing to reinforce (zero-advantage groups dominate).
- If **R1 (Dr. GRPO) moves the arithmetic tasks where R0 doesn't** → a real recipe effect.
  If **neither moves them** → a base-capability ceiling (the Yue et al. "RLVR amplifies,
  doesn't teach" story) demonstrated in the structured-data domain. Either outcome is a
  result _because_ headroom was measured first.

### Caveats

- **Small per-kind n** (6–17). Directional; tighten with the full 800-item set and/or
  multiple seeds before any claim.
- **`pass@1` here is sampled at temp 0.9, not greedy** — it is _lower_ than the greedy
  pass@1 the training eval will report. Do not compare this 3.7% to the eval baseline; the
  real outputs of this run are the headroom metrics (pass@8/64, `frac_never`).
- **`frac_never` ≈ 48% overall** caps what any GRPO run on this base can show: roughly half
  the eval items are unreachable in 64 samples.
- One model, one seed, `max_new=384`. Parsing verified healthy (every kind has pass@64 > 0,
  so the checker does catch correct answers).

### Cross-family: Qwen2.5-1.5B vs SmolLM2-1.7B (same 60 items)

| Task              | type                 | SmolLM2 p@64 / never | Qwen p@64 / never | verdict                     |
| ----------------- | -------------------- | -------------------: | ----------------: | --------------------------- |
| `td_or_fg`        | decision             |            100% / 0% |         100% / 0% | both reach it               |
| `most_scrimmage`  | argmax+name          |            88% / 12% |         100% / 0% | both reach it               |
| `scrimmage_total` | add 2 fields         |        **29% / 71%** |     **100% / 0%** | wall only on SmolLM2        |
| `total_tds`       | sum a column         |        **20% / 80%** |     **100% / 0%** | wall only on SmolLM2        |
| `hundred_yd_rec`  | set/threshold        |            30% / 70% |         70% / 30% | hard for both               |
| `team_points`     | composite arithmetic |        **30% / 70%** |     **40% / 60%** | **wall on BOTH**            |
| **Overall**       | —                    |            52% / 48% |         85% / 15% | Qwen ~6x stronger at pass@1 |

Two findings:

1. **The Qwen confound is live.** The two simple-arithmetic tasks (add two fields, sum a
   column) go from _unreachable_ on SmolLM2 to _perfectly reachable_ on Qwen (frac_never
   71->0%, 80->0%). A Qwen-only study would watch GRPO light these up and conclude "RLVR
   taught structured-data arithmetic" — when Qwen's pretraining already knew it. The
   SmolLM2 control catches it. This is the Spurious Rewards effect demonstrated in-domain,
   and the empirical justification for the multi-family design.
2. **A model-invariant ceiling exists.** `team_points` (composite: TD count x6, then
   - FG x3 + XP + 2pt) walls _both_ models (frac*never 60-70%, pass@64 <= 40%);
     `hundred_yd_rec` is hard for both too. These confound-resistant tasks are where a
     \_credible* claim about GRPO's limits can be made, because neither base samples the
     answer regardless of pretraining.

**Refined Q3.** The dividing line is not "arithmetic vs select/decide." It is a
**model-dependent simple-arithmetic threshold** (the confound) sitting atop a
**model-invariant composite-arithmetic ceiling** (the real finding). Treat
`team_points` (+ `hundred_yd_rec`) as the confound-resistant subset for headline claims.

## **Updated prediction for training.** On Qwen, exp

## 2026-06-21 — Correction: the Qwen R0/R1 training cells are INVALID

A Qwen generation bug broke the first Phase-2-lite matrix: vLLM rollouts never emitted
Qwen2.5-Instruct's `<|im_end|>` stop token, so `clipped_ratio=1.0` the whole run (every
completion hit the length cap). One root cause, two outcomes:

- **Qwen/R1** (`mask_truncated` on): masking every truncated completion zeroed the loss →
  `grad_norm=0`, NaN KL → the optimizer never stepped → LoRA stayed at init (no-op) → eval
  byte-identical to base → a fake `+0.0` on every task. **Void.**
- **Qwen/R0**: trained, but on 100%-truncated completions (degenerate regime). The
  `+0.3 (ns)` "saturated" read probably survives (paired delta cancels a shared truncation
  handicap to first order), but **re-run for a clean comparison.**

**Do NOT** conclude "Dr. GRPO underperforms / vanishing updates on Qwen" — there was no
update to compare. The Qwen R0-vs-R1 contrast is void until both re-run.

**SmolLM2 is untouched and real:** it terminates and trained; the `+14.4 / +10.5pp (***)`
gains and the `team_points` invariant wall stand. Eval (HF `.generate`) terminated Qwen
fine — only the vLLM _training_ path didn't, which localized the fix to rollout stop-tokens.

**Fixes shipped:** `train_grpo.py` passes each model's EOS ids as vLLM stop tokens and
aborts on a no-update run (NaN **or** `grad_norm≈0`); recipe split so R1 = Dr. GRPO core and
`mask_truncated` moved to R2 (it's a DAPO technique, not Dr. GRPO). Re-run sequence:
`qwen-smoke` (validate, cents) → `qwen-r0` + `qwen-r1`.

> **Telemetry caveat (re the entry above):** `clipped_ratio=1.0`, `terminated_length=0`, and
> `min=mean=max` are unreliable in vLLM colocate — they appear in _trusted_ runs too (SmolLM2,
> and the original 1.5B log) with `mean_length` well below the cap. They do **not** prove
> (non-)termination. The decisive signals were NaN KL + `grad_norm=0`. Root cause = `mask_truncated`
> making the masked loss `0/0`; fix = drop it from R1 (+ EOS stop-tokens as belt-and-suspenders).
>
> **Re-run smoke (2026-06-22) PASSED:** Qwen R1 `grad_norm` 0.95→11.6, `kl` ~0.03–0.04 (finite),
> guard clear. No-op fixed; full Qwen R0/R1 cleared to run.

---

## 2026-06-22 — Validated R0/R1 matrix (Qwen re-run, finite KL/grad)

Qwen R0 + R1 re-run after the termination / no-op fix; logs confirm **finite KL and nonzero
grad_norm** (the no-op is gone). `clipped_ratio` still reads 1.0 — the known-broken colocate
metric; ignore it.

| Family / recipe        |    overall Δ | td_or_fg | scrimmage | most_scrim | total_tds | hundred_yd | team_points |
| ---------------------- | -----------: | -------: | --------: | ---------: | --------: | ---------: | ----------: |
| Qwen2.5 / R0           |      +0.9 ns |     +8.2 |      +0.9 |       +0.0 |      -4.7 |       +1.4 |        +0.0 |
| Qwen2.5 / R1 (Dr.GRPO) |      +0.9 ns |     +4.5 |      +0.9 |       +0.0 |      -0.7 |       +0.7 |        +0.0 |
| SmolLM2 / R0           | +14.4 \*\*\* |    +47.0 |      +2.8 |       +4.6 |     +10.0 |      +20.3 |        +0.0 |
| SmolLM2 / R1 (+mask)\* | +10.5 \*\*\* |    +47.0 |      +3.7 |       +3.8 |      +0.7 |       +8.0 |        +0.0 |

\*SmolLM2/R1 still used `mask_truncated` (now an R2 ingredient); its weaker total_tds/hundred
gains may be a masking artifact, not a Dr. GRPO effect. **Re-run clean for a matched R1.**

**Headline (now trustworthy — valid adapters, healthy training dynamics):**

1. **RLVR amplifies a weak base with reachable headroom, not a saturated one** — SmolLM2
   (+14.4 / +10.5pp, *\*\*) vs Qwen (+0.9, ns in *both\* recipes). Qwen's flatness is real
   saturation (high base pass@1), confirmed by finite KL/grad — not the earlier no-op bug.
2. **`team_points` is a model- and recipe-invariant wall (+0.0 across all four cells)** —
   composite arithmetic neither base samples, so GRPO can't teach it.
3. **Qwen-only evaluation would have reported "GRPO doesn't help" and missed the SmolLM2
   effect.** The multi-family control is the contribution.

Possible mild signal (within noise, n~130/kind, ns): Qwen `total_tds` regressed under R0
(-4.7), less so

---

## 2026-06-22 — Clean SmolLM2/R1 (Dr. GRPO core); the EOS fix demonstrably works

SmolLM2/R1 re-run without `mask_truncated`. **The EOS stop-token fix worked:** `clipped_ratio`
**1.0 → 0.0**, completions terminate (mean 55 tok, min 3 / max 458 — real spread). So the earlier
`clipped_ratio=1.0` was partly a _real_ non-termination problem, not pure telemetry noise.
(Correction to the 2026-06-22 telemetry caveat above.) Note: the fix took for SmolLM2 but **not**
Qwen (Qwen still `clipped_ratio=1.0`) — see the Qwen caveat below.

**Result: 5.0% → 16.0% (+11.0pp), McNemar p<0.001, 89 gains vs 1 regression.** Essentially
identical to the masked R1 (+10.5) → confirms `mask_truncated` was inert on SmolLM2, as predicted.
Per-task: `td_or_fg` 0→47% (base scored _below_ its 53% floor — GRPO taught the format + decision);
`team_points` 0→0% (wall holds); gains track pass@k headroom.

**Q2 (do the published fixes help?) — preliminary and confounded.** R0 (+14.4) ≥ R1 (+11.0): Dr.
GRPO did **not** beat naive GRPO; if anything marginally worse. _Caveat:_ the EOS fix landed
mid-matrix, so R0 trained in the long-completion regime and R1 in the terminating regime — not a
clean same-regime comparison. Honest claim: **no evidence the published fixes improve over naive
GRPO in this domain.**

**Qwen caveat:** the EOS fix did not take for Qwen, so its training runs were in the degenerate
long-completion regime. The Qwen-flat / saturation conclusion still holds (pass@k independently
shows Qwen's base is already capable → low headroom), but an airtight Qwen cell needs the
termination fixed + a re-run.

**Status:** headline (headroom-gated amplification + invariant `team_points` wall + confound) is
solid from multiple angles. Remaining for publishable numbers: **seeds** on the SmolLM2 headline
cell. Optional polish: same-regime R0/R1 (the Qwen re-run is no longer needed — see 2026-06-23).

---

## 2026-06-23 — Qwen termination resolved: `clipped_ratio=1.0` is a colocate telemetry artifact (no re-run needed)

The open Qwen caveat ("an airtight Qwen cell needs the termination fixed + a re-run") is **resolved
without a re-run.** The EOS stop-token fix in `train_grpo.py` already works for Qwen; the
`clipped_ratio=1.0` that looked like a degenerate long-completion regime is the known-unreliable
vLLM **colocate metric**, not real non-termination.

**Evidence.**

1. **Plumbing.** TRL 0.19.0's colocate path merges `args.generation_kwargs` — our
   `stop_token_ids=[151645, 151643]` (`<|im_end|>`, `<|endoftext|>`) — straight into the vLLM
   `SamplingParams` (`trl/trainer/grpo_trainer.py` ~L1117–1128). vLLM honors `stop_token_ids`.
2. **Direct measurement** on real training prompts with that exact `SamplingParams`:
   - Qwen2.5-**0.5B**: 16/16 completions stop at `<|im_end|>`, mean 3.9 tok, 0 clipped.
   - Qwen2.5-**1.5B** (`max_tokens=384`, room to reason): **24/24 stop at `<|im_end|>`**, mean
     76.8 tok (max 206), **0 clipped** (`finish_reason="stop"` for every one).

So Qwen rollouts terminate; `clipped_ratio=1.0` is telemetry noise (consistent with the earlier
caveat that trusted runs show it too, with `mean_length` well below the cap). The validated
Qwen-1.5B cell (29.4% → 30.2%, **+0.9pp, ns**) was **already clean**; its flatness is **real
saturation** (high base pass@1), not a training artifact, so a re-run only reproduces +0.9pp.

**Consequence for the headline.** The README hero is now a **two-model** result on the same task
and recipe: GRPO amplifies the weak base (SmolLM2-1.7B **+14.4pp, p<0.001**) and barely moves the
saturated one (Qwen2.5-1.5B +0.9pp, ns) — certified-clean both. "Terminating Qwen re-run" is struck
from the open-polish list; remaining polish is just **seeds** for CIs on the SmolLM2 cell.

---

## 2026-06-23 — Train synthetic, eval REAL: the SmolLM2 gain transfers (and grows)

First real-data eval (`src/data/build_real_eval.py`): the SAME verifiable tasks, but the box scores
are **real nflverse weekly stat lines** (2023 REG wk 1–4, n=509), rendered byte-identically to the
synthetic format. Models train on synthetic; we evaluate on real games. Single seed (R0, s7) on the
existing adapters — a **preview**; the multi-seed matrix will seed-average it.

| Model | base | GRPO-R0 | Δ | McNemar |
| --- | ---: | ---: | ---: | --- |
| **SmolLM2-1.7B** | 8.2% | **29.7%** | **+21.4pp** | p<0.0001, `***` (112 gains / 3 regressions) |
| Qwen2.5-1.5B (saturated) | 45.6% | 45.6% | +0.0pp | p=1.0, ns (14 / 14) |

Per-kind (SmolLM2): `hundred_yd_rec` 0.0→58.6 (`***`), `most_scrimmage` 21.6→36.0 (`***`),
`total_tds` 0.0→10.9 (`***`), `scrimmage_total` 11.7→13.3 (ns).

**Reads.**

1. **The headline transfers to real games** — SmolLM2 +21.4pp (`***`), even *larger* than synthetic
   (+14.4), while saturated Qwen stays flat (+0.0). The amplify-weak-not-saturated split holds on a
   real distribution — the strongest external-validity evidence so far.
2. **Why larger:** `hundred_yd_rec` is much cleaner on real games (0→58.6 vs synthetic 0→20.3) — real
   boxes have fewer, better-separated 100-yd receivers. The synthetic generator is the *harder* test
   (base is lower on synthetic: SmolLM2 5.0% vs real 8.2%; Qwen 29.4% vs real 45.6%).
3. **The arithmetic wall persists on real too:** `scrimmage_total` (add two fields) +1.6 (ns) — GRPO
   can't teach what the base never samples, real or synthetic.

`team_points` and `td_or_fg` are omitted on real data (not reconstructable from weekly skill rows /
no game-state — see `build_real_eval.py`). Next: re-run across all 3 seeds once the matrix lands for
a seed-averaged real number + CI.
