# Flagship spec (Paper 4) — The measure-zero blind spot: one task, two optimizers

**Goal.** Demonstrate, on a single controlled synthetic task, that **supervised average-loss
learning (SFT)** and **RL expected-reward learning (GRPO/RLVR)** are blind to the *same*
sharp, rare, deductively-decidable cell — and that supplying the structure (the rule, or a
label that localizes the cell) recovers it for *both*. The thesis in one figure.

This unifies Paper 1 (the clock-kill *deductive residual no average-loss learner recovers*)
and Paper 3 (the GRPO *composite-arithmetic wall the base never samples*) as two faces of one
phenomenon: **expectation-optimizers do not localize low-measure sharp structure.**

## Claim & falsifiable predictions
- **P1 (joint blindness).** Held-out cell recovery → 0 as cell rarity ε → 0, for **both** SFT
  and GRPO. The recovery-vs-ε curves collapse together; the oracle stays at 1.
- **P2 (not representation).** Handing the learner the *exact* trigger variable the rule
  thresholds on does **not** recover the cell at small ε. (Clock-kill "kneel-margin" control.)
- **P3 (not neighborhood).** Up-weighting a rule-free *region* containing the cell drifts to the
  region's average, not the deductive answer. (Clock-kill region control.)
- **P4 (structure supplied recovers).** Localizing the exact cell (label up-weight) or applying
  the rule (oracle) recovers it — for **both** objectives — and label-localization reproduces
  the clock-kill **bleed**: it overshoots the near-cell band (the smoothing bias from the other
  side). Reproducing that bleed here is an internal-consistency check linking Papers 1 & 4.
- **P5 (sets up Paper 5).** There is a critical ε*(sharpness, smoothing-bias) below which
  recovery collapses. Paper 4 *measures* the curve; Paper 5 *derives* ε*.

**What would kill the thesis:** if GRPO recovers where SFT doesn't (or vice-versa) on the same
cell at the same ε; if recovery does *not* fall with ε in the generalizing regime; or if the
exact-feature control (P2) recovers at small ε (then it's representation, not measure).

## The synthetic task family (fully deductive, supports both objectives)
- **Inputs.** a, b ~ Uniform{0..99}; rendered as a short prompt `a=37 b=52 =>`. Model emits the
  answer token(s).
- **Bulk rule (the "smooth/common" relationship).** `h(a,b) = (a+b) mod 100`. Easy; the model
  learns it from the majority and generalizes it everywhere.
- **Margin & cell.** `m(a,b) = (a-b) mod 100`. Cell `C = { m in [K, K+w) }`, a sharp band of
  width w at location K, so **ε = w/100** — the single rarity knob.
- **Deductive override on the cell.** `r(a,b) = (a*b) mod 100` — exactly computable, distinct
  from the bulk, and *not* signalled by any smooth feature except the threshold on m.
- **Global target.** `y = r(a,b) if m in [K,K+w) else h(a,b)`. **Oracle** = compute the predicate,
  apply r or h. Recovery is measured on held-out cell instances.

Three knobs: **ε** (band width w), **sharpness s** (hard step vs. an override probability that
ramps 0→1 across a transition zone — tests that *soft* cells ARE recovered), and **smoothing**
(weight decay / model size / data size — the generalizing-vs-memorizing regime).

## The two optimizers (matched)
- **SFT.** Cross-entropy on (prompt → y) pairs; minimizes average loss over the data, ε of which
  is the cell.
- **GRPO.** Verifiable 0/1 reward (emitted == y); maximizes expected reward over the same prompt
  distribution, ε of which is the cell.
- **Shared warm-start.** Both branch from one checkpoint trained to competence on the **bulk
  only**, so (a) GRPO can emit valid answers (no cold-start exploration artifact) and (b) it
  matches the realistic SFT→RLVR pipeline. Match steps/tokens/data budget across the two arms.

**A bonus the prior papers couldn't do — decompose the proximate mechanism.** SFT's blindness is
*measure + smoothing* (the cell is ε of the loss). GRPO's is *measure + exploration* (after warm-
start the model never samples `r` on cell prompts → 0 reward → 0 advantage → no gradient, the
exact gridiron wall). Add a control that **fixes GRPO's exploration** (inject occasional oracle
answers into the rollout group / raise temperature / dynamic-sample the cell) and show it *still*
fails from measure alone — isolating measure-blindness from exploration-blindness. That separation
is a genuine new result, not in Papers 1 or 3.

## Recovery ladder (run every rung under BOTH objectives)
1. **No structure** — raw task. (Predict: blind, ε→0.)
2. **+ exact trigger feature** — also give `m`. (P2: still blind.)
3. **+ region emphasis (no rule)** — up-weight a coarse band `[K-Δ, K+w+Δ]`, Δ≫w. (P3: drifts.)
4. **+ exploration fix (GRPO only)** — isolate measure from exploration (above).
5. **+ cell localization (label)** — up-weight exactly C. (P4: recovers + bleed.)
6. **Oracle (rule at inference).** Ceiling = 1.

## Substrates (two tiers)
- **Primary: tiny transformer from scratch (~1–10M params).** Cleanest — *no pretrained-prior
  confound*, so the recovery collapse is a property of the optimizer, not the prior. Runs in
  minutes on the 3080, enabling the full ε × sharpness × smoothing × objective × rung sweep with
  3+ seeds. This produces the headline figure.
- **Secondary: a 0.5–1.5B pretrained LLM (LoRA).** Replicate the headline to show it transfers to
  the LLM/RLVR regime and connects to Papers 1 & 3. Reuse the gridiron harness (verifiable reward,
  pass@k, CIs, McNemar, the fail-loud guard).

## Metrics & the headline figure
- **Cell recovery** (held-out C accuracy) — the dependent variable.
- **Bulk accuracy** (off-cell) — sanity; both objectives must nail the bulk.
- **Near-cell bleed** — accuracy on the adjacent band; quantifies the smoothing overshoot (P4).
- **Representability check** — a model trained *only* on C learns it (isolates localization from
  capacity; the gridiron pass@k / clock-kill "learnable once localized" analog).
- **Headline figure:** cell-recovery vs ε, SFT and GRPO overlaid, collapsing together as ε→0,
  oracle flat at 1. Secondary panels: sharpness sweep (soft cells recovered), the bleed, and the
  exploration-fix control.

## Controls / threats
- **Generalizing regime, not memorization.** Pick capacity/weight-decay so the learner
  *generalizes* the bulk (doesn't pure-memorize the cell); show recovery collapses with ε *there*.
  Over-capacity + no reg can memorize a tiny cell — but that's overfitting, and it pays in the
  near-cell bleed (which is the point, not a defect).
- **Matched compute/data** between SFT and GRPO.
- **Bulk must be learnable**; cell answer must not be guessable from the bulk pattern.
- **Multiple seeds** (3+) with CIs on every curve (reuse `src/eval/stats.py`).

## Compute & sequence (cheap, mostly local)
1. Tiny-transformer harness: task generator (ε/sharpness), SFT loop, GRPO loop, oracle, the ladder.
   (~a weekend; 3080, runs in minutes/cell.)
2. Full sweep + headline figure + seeds. (~local, <$5.)
3. 0.5B-LLM replication of the headline. (~$10–20 rented or local 0.5B.)
4. Write-up. Target: ICLR/NeurIPS workshop first; a strong tiny+LLM version with the mechanism
   decomposition is a plausible main-track or TMLR submission.

## Why this is the keystone
It puts SFT and RL on the *same* task and shows one blindness with two faces, reproduces both the
gridiron wall and the clock-kill bleed in a controlled setting, and hands Paper 5 a measured curve
to derive ε* against. After Paper 4, Papers 1 and 3 are case studies *of* a named phenomenon.
