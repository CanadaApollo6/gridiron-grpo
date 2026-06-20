# The journey: from "run a model on a rented GPU" to a controlled RLVR study

This is the narrative record of how this repo got to where it is — the bugs, the
dead ends, the results, the diagnosis, the literature check, and the pivot. It
doubles as project memory and as raw material for the write-up. Read top to bottom
and you'll see a fairly honest picture of what doing RL post-training actually
feels like in practice.

---

## TL;DR

We set out to *run* GRPO once on a rented GPU. Getting it to run took clearing a
gauntlet of version/infra bugs (each caught cheaply by a smoke test). Once it ran,
GRPO **consistently failed to help** — flat-to-negative on Qwen2.5-1.5B and 0.5B.
The training logs explained why: **KL divergence exploded ~15× while accuracy stayed
flat** (drift, not learning), completions degenerated to the length cap, and the base
model was already competent (no headroom). A literature review showed every one of
those phenomena is already published — so a single result isn't novel. But the
*setting* (structured/tabular verifiable reasoning, multiple model families, the
Qwen "spurious rewards" confound) is under-explored, so we pivoted to a **controlled
multi-family study** (see [`EXPERIMENTS.md`](EXPERIMENTS.md)).

---

## Act I — Making it run (the bug gauntlet)

The original ask was simple: train the model on a rented GPU (HF Jobs, since the user
has an HF Pro account). The code was complete; *running* it was the adventure. We
adopted a discipline that paid for itself many times over: **validate the cheapest
way possible before spending on a full run** — first CPU-only locally, then a 3080 in
WSL, then a few-cents `SMOKE_ONLY` job on the real A100. Each layer caught a distinct
bug:

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | `transformers 5.x` + `trl 0.17` mismatch | loose version ranges | pin a coherent stack | `78caf5c` |
| 2 | `import trl` fails on Windows | trl≥0.16 hard-imports vLLM, which has no Windows build | move to Linux/WSL; pin `trl 0.19` | `78caf5c`, `e15edfe` |
| 3 | flat/None grads with LoRA | gradient checkpointing didn't propagate to adapters | `enable_input_require_grads()` | `e15edfe` |
| 4 | `KeyError: 'RANK'` | vLLM colocate needs torch-distributed env vars | launch via `torchrun`, not `accelerate --num_processes 1` (simple_launcher) | `234a2b1`, `32e90a8` |
| 5 | `TypeError: NoneType not iterable` | TRL 0.19 colocate does `generation_kwargs.update(None)` | pass `generation_kwargs={}` | `0776501` |
| 6 | `marked ready twice` in backward | DDP + reentrant gradient checkpointing | `use_reentrant=False` | `026094b` |
| 7 | PowerShell mangled the job command | bash `\` line-continuations + `$(curl)` | one-line shell-agnostic command (container clones + runs the script) | `d585256` |

Two infra notes worth remembering: vLLM colocate runs single-GPU in-process only on
**trl ≥ 0.18** (0.17 is server-mode only), and the whole run is **configuration-driven**
on a stock `pytorch/pytorch:2.6.0-cuda12.4` image — no custom Docker.

The pre-flight failures that caught bugs #4–6 on the real A100 cost a grand total of
**$0.42** — cheap insurance against a full-priced run dying partway.

---

## Act II — The results (and an eval bug that hid the truth)

Three full Qwen2.5 runs, all evaluated by task kind:

| Run | Model | Completion cap | Prompt | Base | Tuned | Δ |
|---|---|---|---|---|---|---|
| 1 | 1.5B | 512 | verbose | 39.4% | 39.0% | −0.4 |
| 2 | 1.5B | 512 | **brevity** | 28.0% | 28.9% | +0.9 |
| 3 | 1.5B | 1024 | verbose | **57.5%** | 56.2% | −1.2 |

The brevity-prompt experiment (Run 2) **backfired**: forcing 2–3 sentences amputated
the multi-step reasoning these tasks need, tanking *both* base and tuned. Its only
"win" (+0.9) was relative to a depressed baseline — a cautionary tale about reading
deltas without absolutes.

Then a `do_sample`/right-padding warning tipped us off to a real **eval bug**: decoder-
only models must be **left-padded** for batched generation, and eval was also capped at
512 tokens. Fixing both (`87e1281`, `b7de5a7`) revealed the true baseline was **57.5%**,
not 39% — the bug had been hiding ~18 points and adding noise. Lesson: *a tokenizer
default nearly made us report a number that was badly wrong.*

Even with trustworthy eval, **GRPO did not help** (57.5% → 56.2%). The per-task pattern
was consistent across all runs: the easiest task (`td_or_fg`, a decision) improved,
while the hardest multi-step/set tasks (`hundred_yd_rec`, `team_points`) regressed.

---

## Act III — The diagnosis (read the training logs)

We pulled the full 1.5B training log (`job_15b.log`) and parsed the trajectory. The
story was unambiguous:

- **correctness reward: flat** (~0.29 throughout, noisy, no trend)
- **format reward: flat** at ~0.176 (already ~88% compliant; no gaming)
- **KL divergence: 0.06 → 0.86 (~15×)** — the policy drifted enormously from the base
- **`clipped_ratio` is a misleading metric** in colocate mode (mean length varied
  518–751 well under the 1024 cap) — truncation was never the real problem

The 0.5B run (with a *stronger* KL anchor, `beta=0.1`, `lr=5e-7`) was **worse**: KL hit
0.97 even faster, `grad_norm` was huge and erratic (55–165), and **every** completion
ran to exactly 1024 tokens (the small model never emits a stop token at temp 0.9).

**Diagnosis:** GRPO was *destabilizing* the model, not teaching it. KL ran away while
accuracy stayed flat — drift without learning — and the strong 1.5B base had no headroom
to gain anyway.

---

## Act IV — The literature check (are we onto something?)

Before sinking credits into "fixing" it, we did an honest lit review. Verdict: **every
phenomenon we hit is already published**, often with named fixes.

- Non-terminating / ever-longer completions → **GRPO length bias**, fixed by **Dr. GRPO**
  ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)).
- KL blow-up / instability / collapse → documented naive-GRPO behavior
  ([policy-divergence work](https://arxiv.org/abs/2602.05494)).
- Strong base doesn't improve; RLVR amplifies rather than teaches; narrows pass@k
  ([Yue et al.](https://openreview.net/forum?id=4OsgYD7em5), [Limit of RLVR](https://limit-of-rlvr.github.io/)).
- **The Qwen confound:** on Qwen, even random/incorrect rewards "work" (RLVR elicits
  pretrained behaviors); the same rewards fail on Llama/OLMo
  ([Spurious Rewards](https://openreview.net/forum?id=4NeiwxQ2Bp)). **This means any
  Qwen-only RLVR result is not credible.**

So: as a single reproduction, **not novel**. But the *gap* the literature leaves —
structured/tabular reasoning, studied across model families with the known fixes — is
real and reachable.

---

## Act V — The pivot (a controlled study)

We reframed the work as: **"When does RLVR help on structured/tabular verifiable
reasoning, across model families?"** Three controls the existing work skips:

1. **Multiple families** (Qwen, Llama-3.2, SmolLM2, OLMo-2) to defuse the Qwen confound.
2. **The published fixes** (Dr. GRPO, DAPO stability) applied and measured in this domain.
3. **A compositional-depth task taxonomy** — do the per-task effects track task depth,
   consistently across families?

TRL 0.19 supports all the fixes natively, so we exposed them as `hf_job.sh` env vars
(`LOSS_TYPE=dr_grpo`, `NO_SCALE_REWARDS`, `MASK_TRUNCATED`, `EPSILON_HIGH`, plus
`NO_FORMAT_REWARD`, `SEED`) and made LoRA portable across architectures
(`target_modules="all-linear"`). Full design in [`EXPERIMENTS.md`](EXPERIMENTS.md)
(commit `609236c`).

---

## Act VI — Hardening the study (turning a reproduction into a contribution)

A code review (see [`REVIEW.md`](REVIEW.md)) flagged that the pivot's headline — the
per-task compositional-depth taxonomy — was **confounded by task design**, and that
single-run per-task deltas were **inside the noise floor**. We fixed the foundation
before spending Phase 2 budget:

- **Rebalanced the data.** `td_or_fg` was 74% "TD" by construction (a model could score
  74% by never reasoning); it is now ~52/48. Boxes were physically contradictory in ~86%
  of cases (team TDs disagreed with the player rows); team points are now derived from the
  players, fully consistent. Last names are unique per box, so "which player" is well-posed.
- **Made every number defensible.** Eval now reports a naive **best-constant floor** per
  task, **Wilson 95% CIs** on every cell, a paired **McNemar** test base-vs-tuned, per-class
  recall (to catch majority-class collapse), and a **measured** terminated fraction (the
  colocate length telemetry is unreliable). `src/eval/stats.py` is unit-tested against
  Monte-Carlo and textbook values.
- **Added the learnability probe.** `src/eval/pass_at_k.py` measures base pass@1/8/64 per
  task. GRPO can only reinforce what the base samples, so this predicts which matrix cells
  can move *before* paying for them — it is now Phase 1.
- **Exposed the knobs the diagnosis implicated.** LR schedule + warmup (the old run decayed
  LR to ~0), a `beta` sweep (is the KL runaway a discipline problem?), DAPO dynamic sampling,
  and a partial-credit numeric reward for the sparsity problem. Each run writes a
  self-describing `recipe.json`.

One honest correction from the review itself: the set task's *exploitable* baseline was ~38%
(always "none"), not the ~62% first quoted — 62% was the non-empty fraction, which no
constant answer can capture. The decision task (74%) was the real confound.

**Net:** the reproduction stands, and the study around it is now built to produce a result
that survives scrutiny rather than a table that doesn't.

## Cost ledger (HF Jobs, A100-large @ $2.50/hr)

| Item | Cost |
|---|---|
| Pre-flight debugging failures (caught 3 bugs) | $0.42 |
| Run 1 — Qwen 1.5B, 512, verbose | $6.54 |
| Run 2 — Qwen 1.5B, 512, brevity (overnight) | $5.54 |
| **Subtotal through Run 2** | **$12.50** |
| Run 3 (1.5B, 1024) + Run 4 (0.5B) + Phase-0 pilots | accruing |

A single full 1200-step run is **~$5–6**. The entire debugging+experiment arc so far
is on the order of **$25** — versus the README's conservative "well under $200."

---

## Lessons (the practitioner takeaways)

1. **The infrastructure is the easy part; landing the optimization is the hard part.**
   Running GRPO is cheap and quick; making it *improve* the model is the real work.
2. **Validate cheaply, in layers.** CPU → 3080/WSL → few-cents A100 smoke. The $0.42 of
   failed pre-flights saved multiple full-priced crashes.
3. **Read the per-task breakdown and the training curves, not the headline average.**
   The KL curve told us more than the accuracy number.
4. **Eval bugs masquerade as model results.** A tokenizer padding default hid ~18 points.
5. **Know whether you have a "when" problem or a "how" problem** before reaching for RL.
6. **Beware the Qwen confound** — it invalidates a huge amount of single-model RLVR work.

---

## Current status & next

- All four model families are accessible (Llama license accepted).
- Phase 0 (infra pilots for SmolLM2 + OLMo-2) is running.
- Next: Phase 1 decisive run (Dr. GRPO on a non-Qwen base — does KL stay bounded and
  does accuracy move?), then the Phase 2 family × recipe matrix with a results
  aggregator. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the plan.
