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
