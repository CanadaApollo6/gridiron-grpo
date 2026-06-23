# Paper 3 — Project Charter (LIVING)

### The Keystone: one task, two optimizers, one blind spot

_(working title — not for profile/publication.)_

> **NUMBERING NOTE (confirm).** This doc uses the **project scheme**: P1 = deductive-ground-truth eval (clock-kill), P2 = GRPO/RLVR study (gridiron), **P3 = this keystone synthesis**, P4 = cross-domain deductive-ground-truth suite (separate paper). The flagship spec you drafted used a **+1 scheme** (its "Paper 4" = this doc; its "Paper 3" = our P2; its "Paper 5" = a future ε*-derivation). Keeping the project scheme avoids renumbering five docs. The ε*-derivation is a **downstream theory paper** (call it P5); the **cross-domain suite stays P4 and is NOT folded in here**. To adopt the spec's numbers instead = a global renumber across the master brief, both charters, LinkedIn, and the OSS list.

---

## HOW TO USE THIS DOC

Read this first every session. **Update the Status block and Update Log as you work.** "Upstream (frozen)" summarizes Papers 1 & 2 — read-only; their source of truth is their own Cowork projects.

This is the **synthesis / keystone** — after it, Papers 1 and 2 are case studies _of_ a named phenomenon. The primary instrument changed (Jun 2026): from a real-model scale-divergence ladder to a **controlled synthetic task on a tiny from-scratch transformer**, with the **NeMo scale ladder repositioned as the real-model replication arm** (NeMo is load-bearing — see that workstream).

---

## STATUS — UPDATE EACH SESSION

- **Phase:** Spec reconciled (synthetic keystone + NeMo replication). Pre-experiment.
- **Done:** Mechanism reframed to measure/density-below-resolution; `team_points` diagnosed as **group-density** (not never-sampled); synthetic task family specced; substrates chosen.
- **Largely settled:** the old one-vs-two-mechanism question → **one phenomenon (measure below resolution), with optimizer-specific proximate mechanisms now separable on one cell.**
- **Next (confirm → build → port → scale):** (1) tiny-transformer harness + the ε × sharpness × G × objective sweep on the 3080 → the headline figure. (2) Confirm `team_points` group-density on the current gridiron stack (settle the n=10 worry). (3) Port gridiron → NeMo-RL at 1.5B (shakedown). (4) NeMo scale ladder 7B+ as the real-model replication arm.
- **Last updated:** _[fill in]_

---

## ONE-LINER

Expectation-optimizers are blind to low-measure sharp structure. Average-loss SFT can't localize a cell that's ε of the loss; on-policy GRPO can't reinforce a cell whose correct answer is below the group's sampling resolution. **Same blind spot, two faces.** One synthetic task puts both optimizers on the same deductive cell, shows the recovery curves collapse together as the cell gets rarer, and shows that supplying the rule (or a label that localizes the cell) recovers it for both. Papers 1 & 2 are the real-world case studies; this is the controlled demonstration.

---

## THESIS & FALSIFIABLE PREDICTIONS

**Thesis:** expectation-optimizers do not localize low-measure sharp structure. The clock-kill deductive residual (Paper 1, SFT side) and the gridiron composite wall (Paper 2, GRPO side) are two faces of it. Dependent variable: held-out **cell recovery**; ε = cell rarity.

- **P1 — joint blindness.** Recovery → 0 as ε → 0 for **both** SFT and GRPO; the curves collapse together; oracle stays at 1.
- **P2 — not representation.** Handing the learner the exact trigger variable the rule thresholds on does **not** recover the cell at small ε. (= clock-kill kneel-margin control.)
- **P3 — not neighborhood.** Up-weighting a rule-free _region_ containing the cell drifts to the region's average, not the deductive answer. (= clock-kill region control.)
- **P4 — structure supplied recovers + bleed.** Localizing the exact cell (label up-weight) or applying the rule (oracle) recovers it for **both** objectives; label-localization reproduces the clock-kill **bleed** (overshoots the near-cell band). Reproducing the bleed is an internal-consistency link to Paper 1.
- **P5 — sets up the theory paper.** There's a critical ε*(sharpness, smoothing) below which recovery collapses. This paper *measures* the curve; a downstream paper *derives* ε*.

**What kills the thesis:** GRPO recovers where SFT doesn't (or vice versa) on the same cell at the same ε; recovery does _not_ fall with ε in the generalizing regime; or the exact-feature control (P2) recovers at small ε (then it's representation, not measure).

---

## THE SYNTHETIC TASK (the headline instrument)

Fully deductive, supports both objectives, **no football** (no pretrained-prior confound, and inherently more main-track-legible).

- **Inputs:** a, b ~ Uniform{0..99}, rendered `a=37 b=52 =>`; model emits the answer token(s).
- **Bulk rule:** `h(a,b) = (a+b) mod 100` — the common relationship; learned from the majority, generalizes everywhere.
- **Margin & cell:** `m(a,b) = (a−b) mod 100`; cell `C = { m ∈ [K, K+w) }`. **ε = w/100** — the single rarity knob.
- **Deductive override on the cell:** `r(a,b) = (a×b) mod 100` — exactly computable, distinct from bulk, unsignalled by any smooth feature except the threshold on m.
- **Target:** `y = r if m ∈ [K,K+w) else h`. **Oracle** = compute the predicate, apply r or h.

**Knobs:** ε (band width w); **sharpness s** (hard step vs. an override probability ramping 0→1 across a transition zone — tests that _soft_ cells ARE recovered); **smoothing** (weight decay / model size / data size — the generalizing-vs-memorizing regime).

---

## THE TWO OPTIMIZERS (matched)

- **SFT:** cross-entropy on (prompt → y); minimizes average loss, ε of which is the cell.
- **GRPO:** verifiable 0/1 reward (emit == y); maximizes expected reward over the same prompt distribution.
- **Shared warm-start:** both branch from one checkpoint trained to competence on the **bulk only** — so GRPO can emit valid answers (no cold-start artifact) and it matches the real SFT→RLVR pipeline. Match steps / tokens / data across arms.

---

## MECHANISM DECOMPOSITION — THE NEW RESULT (and the team_points fix)

This is where the reconciliation bites. The spec's original framing — "GRPO never samples r on the cell → 0 reward → the exact gridiron wall" — is the **pre-reconciliation** story and it's too coarse. The real gridiron `team_points` wall is **not** never-sampled: the base lands it ~30% at pass@64; the wall is **group-level density** (≈3.8% in-group positive rate at G=8 → groups almost always all-negative → zero advantage). So GRPO's blindness factors into **three** regimes, not two — and the synthetic task can turn each knob independently. Treat the decomposition as the **hypothesis the sweeps test**; demonstrating the knobs move recovery _independently_ is the contribution.

1. **Exploration (never-sample).** In-cell `P(emit r) = 0` → no positive ever → unrecoverable at any G. _This is the spec's warm-start-on-bulk-only default — the clean measure-zero limit._
2. **Group-resolution (density).** In-cell `P(r)` small but > 0 (the `team_points` regime, ~0.5–4%) → positives exist but a group of G is mostly all-negative → starved gradient. Non-degenerate-group fraction = `1 − (1−P(r))^G`. **Recedes as G grows or P(r) grows.** _This is the actual `team_points` wall, reproduced as a curve._
3. **Measure (ε-weighting).** Even with positives in groups, cell prompts are ε of the distribution → the cell's gradient is rare and must fight the bulk's smooth extrapolation toward h. **The shared axis with SFT.**

SFT's blindness is the analogous pair: **measure** (cell is ε of the loss) + **smoothing** (the average-loss bias that smears a sub-resolution cell onto its neighbors — the bias that produces the clock-kill bleed).

**The unification:** a **shared measure axis (ε)** — both collapse as ε→0 (P1) — plus an **optimizer-specific resolution mechanism** (smoothing for SFT; exploration + group-resolution for GRPO). That settles one-vs-two: **one phenomenon, separable proximate mechanisms.**

**Sweeps that isolate them:**

- Sweep **in-cell sampling rate** (warm-start residual P(r) / temperature) → separates exploration (1).
- Sweep **group size G** → separates group-resolution (2): the wall recedes with G. _(This is the charter's group-size sweep, ported into the synthetic task — same `1−(1−p)^G` curve.)_
- Sweep **ε** (cell frequency) → isolates measure (3): recovery falls with ε even when (1) and (2) are handled.

This three-way separation is **new** — not in Paper 1 or Paper 2 — and **subsumes the team_points group-density finding as the measurable middle regime** between never-sample and pure measure.

---

## RECOVERY LADDER (run every rung under BOTH objectives)

1. **No structure** — raw task. (Predict: blind, ε→0.)
2. **+ exact trigger feature** — also give m. (P2: still blind.)
3. **+ region emphasis (no rule)** — up-weight a coarse band `[K−Δ, K+w+Δ]`, Δ≫w. (P3: drifts.)
4. **+ exploration / group-resolution fix (GRPO only)** — raise in-cell P(r) and/or G; isolate (1) and (2) from (3).
5. **+ cell localization (label)** — up-weight exactly C. (P4: recovers + bleed.)
6. **Oracle (rule at inference).** Ceiling = 1.

---

## THE NeMo-RL BUILD — REAL-MODEL REPLICATION ARM (load-bearing, non-negotiable)

NeMo stays in, repositioned. The tiny transformer is the clean headline; **the NeMo scale ladder is the real-model replication that (a) shows the synthetic result transfers to real LLMs/RLVR and (b) carries the RE artifact** — distributed training on NVIDIA's own stack. This restores the "one build, three payoffs" convergence a synthetic-only paper would have broken.

**The scale axis IS the in-cell-density knob on real models.** As the base scales, P(emit r) on cell prompts rises → at fixed G the group-resolution wall (regime 2) recedes → predict the crossover. So the original scale-divergence ladder isn't dropped — it's reinterpreted as moving knobs (1)/(2) on real models, complementing the synthetic task's direct control.

**Sequencing — confirm → port → scale:**

- **Step 0 — confirm on the current gridiron stack (single GPU).** Re-confirm `team_points` group-density; settle the n=10 pass@8 worry before the port.
- **Step 1 — port gridiron → NeMo-RL at 1.5B (DTensor path).** Port shakedown; debug NCCL/config cheaply. _Constraint: NeMo-RL LoRA GRPO is **DTensor-backend-only**; Megatron-Core is SFT-only (RL-LoRA "coming soon") — use DTensor._
- **Step 2 — scale ladder 7B → 32B → 70B on NeMo-RL; Megatron multi-node** at the top. Replicate the synthetic headline on real models; show the group-resolution wall receding with scale.
- **Artifact (for RE):** "the keystone's real-model arm runs on NeMo-RL, single-GPU through multi-node," + a **profiling writeup** (MFU, bottleneck, fix). The line that flips a recruiter from SWE to RE — falls out for free.

---

## SUBSTRATES

- **Primary — tiny transformer from scratch (~1–10M params).** Cleanest: **no pretrained-prior confound**, so the recovery collapse is a property of the optimizer, not the prior. Minutes on the 3080 → the full ε × sharpness × smoothing × G × objective × rung sweep, 3+ seeds. **Produces the headline figure.**
- **Secondary — NeMo-RL real-model ladder (1.5B → 7B+).** Replicate the headline; show transfer to the LLM/RLVR regime and the link to Papers 1 & 2. Reuse the gridiron harness (verifiable reward, pass@k, CIs, McNemar, fail-loud guard). _(This IS the NeMo arm above — substrate and RE artifact in one.)_

---

## METRICS & HEADLINE FIGURE

- **Cell recovery** (held-out C accuracy) — dependent variable.
- **Bulk accuracy** (off-cell) — sanity; both objectives must nail the bulk.
- **Near-cell bleed** — accuracy on the adjacent band; quantifies the smoothing overshoot (P4).
- **Representability check** — a model trained _only_ on C learns it (isolates localization from capacity; the gridiron pass@k / clock-kill "learnable once localized" analog).
- **Headline figure:** cell-recovery vs ε (≡ density), SFT and GRPO overlaid, **collapsing together as ε→0**, oracle flat at 1. Secondary panels: sharpness sweep (soft cells recovered), the bleed, the group-size / in-cell-rate sweeps.

---

## CONTROLS / THREATS

- **Generalizing regime, not memorization.** Pick capacity/weight-decay so the learner _generalizes_ the bulk; show recovery collapses with ε _there_. Over-capacity + no reg can memorize a tiny cell — that's overfitting, and it pays in the near-cell bleed (the point, not a defect).
- **Matched compute/data** across SFT and GRPO.
- **Bulk learnable; cell answer not guessable** from the bulk pattern.
- **`team_points` pass@8 is n=10** — confirm the 3.8% is real (not 1–2 lucky samples) before the real-model arm leans on it. (Step 0.)
- **3+ seeds** with CIs on every curve (reuse `src/eval/stats.py`).

---

## UPSTREAM (FROZEN — canonical source in the Paper-1 / Paper-2 projects)

### Paper 1 — Deductive-Ground-Truth Evaluation ("Forecasting With a Known Answer") — _the SFT-side case study_

- **Setting:** NFL clock-killing first down; outcome fixed by kneel-out arithmetic over clock/score/timeouts; labeled without WP/EP models.
- **Result:** univariate family ~0.970 (residual ~0.03); covariate access doesn't lift it (GBM raw 0.960; **exact kneel-margin 0.959**; Chronos-2 covariate 0.968); region-upweight 0.960; **only** label-upweight (0.992) or oracle (1.000) recover.
- **Mechanism = rarity below loss resolution (measure + smoothing):** a ~0.1% cell pays no rent in average loss; the kneel-margin control proves it's localization, not capability.
- **Caveats:** existence (model-independent) vs magnitude (~0.03, nflfastR-specific); n=38 degenerate set; near-miss audit (704 plays) shows a 3.9pt bleed; no standard calibration diagnostic flags it.

### Paper 2 — RLVR on Structured-Reasoning (the `gridiron-grpo` study) — _the GRPO-side case study_

- **Setting:** GRPO post-training, verifiable 0/1 reward; TRL + vLLM + LoRA, single-GPU; multi-family SmolLM2-1.7B vs Qwen2.5-1.5B.
- **Findings:** RLVR amplifies reachable headroom (SmolLM2 +14.4 / +11.0pp, McNemar p<0.001 _within-run_; Qwen flat +0.9 ns). **`team_points` wall — CORRECTED:** base **does** sample it (per-sample ≈0.5%, pass@8 ≈3.8%, pass@64 ≈30%, frac_never 70%, **n=10**); the wall is **in-group density below the group scale (G≈8 → ~3.8% non-degenerate)**, NOT absent capability. **+0.0 across both recipes and both models.** Contrast `hundred_yd_rec`: same pass@64 ≈30% but pass@8 ≈10.7% → +20.3. **Robust claim = the invariance, not any single reachability number; pass@64 overstates what's learnable.**
- **Caveats:** small per-kind n (`team_points` pass@8 ≈ 3 of ~80 hits — thin; invariance carries it, raise n); seeds pending; Qwen cells ran in a degenerate long-completion regime (saturation holds via pass@k; airtight Qwen needs termination fix + re-run). Qwen confound = Spurious-Rewards in-domain (justifies multi-family). Q2: no evidence Dr. GRPO beats naive GRPO (confounded by mid-matrix EOS fix).

---

## OPEN QUESTIONS / RISKS

- **Numbering** (top) — confirm scheme.
- **Does the synthetic collapse transfer to real models?** The NeMo arm is the test. If it doesn't, the tiny-transformer result is weaker (a property of small models, not optimizers in general).
- **`team_points` pass@8 robustness** (n=10) — Step 0.
- **Generalizing-regime tuning** — the result only holds where the learner generalizes the bulk; rule out the memorization regime per cell.
- **Clean three-way separation** — whether (1)/(2)/(3) actually separate empirically is itself a result; if they don't cleanly separate, report the entanglement honestly.
- **Venue:** tiny + real-model + the three-way decomposition is a plausible main-track / TMLR shot; synthetic-only is a strong workshop. No-football framing helps main-track legibility. High-ceiling, high-variance.
- The ε\*-derivation is a **separate downstream paper** — don't scope-creep it in here.

---

## UPDATE LOG

- _[date4]_ — **Reconciled with the flagship synthetic spec.** Adopted the synthetic task (tiny from-scratch transformer) as the headline instrument; repositioned the NeMo scale ladder as the real-model replication arm (NeMo stays load-bearing). Corrected the GRPO-wall framing from "never-sample" to the three-way decomposition (exploration / group-resolution / measure), subsuming `team_points` group-density as the middle regime. Folded the group-size sweep into the synthetic task. Added P1–P5 predictions, recovery ladder, substrates, metrics, controls from the spec. Flagged numbering reconciliation; cross-domain suite stays a separate P4.
- _[date3]_ — Added the NeMo-RL Build workstream (confirm→port→scale).
- _[date2]_ — Mechanism reframed (density-vs-resolution); `team_points` corrected to group-density.
- _[date1]_ — Charter created (two failure modes, scale-divergence core; superseded).
