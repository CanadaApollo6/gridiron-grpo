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

### Status / next

- [x] SmolLM2-1.7B base pass@k (this run) → `results/passk_base_smollm2.json`
- [ ] Same probe on **Qwen2.5-1.5B** (does the arithmetic wall replicate off-SmolLM2? — Q1)
- [ ] **R0 vs R1** training on SmolLM2 (test the prediction above, read via `compare.py`)
- [ ] Scale the probe to the full 800-item set / add a second seed before publishing numbers
