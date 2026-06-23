# Paper outline — RLVR on structured/tabular reasoning

**Working title:** *Reinforcement Learning with Verifiable Rewards Amplifies Reachable
Skills and Stalls at a Composite-Arithmetic Wall: a Controlled Multi-Family Study on
Tabular Reasoning.*

**Target:** TMLR or a reasoning/RLHF workshop. Framing leads with the two novel pieces
(the decomposable ceiling + the pass@k-gating protocol); the confirmations of Yue/Shao are
support, stated as such.

## The contribution (state it plainly, up front)
1. **A decomposable, partly model-invariant ceiling.** A controlled task suite over
   compositional depth localizes *where* RLVR's benefit stops: simple arithmetic is a
   **model-dependent** threshold (Qwen clears it, SmolLM2 doesn't); composite arithmetic
   (`team_points`) is **invariant** — no base samples it, no recipe teaches it.
2. **A pass@k learnability gate.** Measuring base pass@k *before* training predicts which
   task×family cells can move; we show it predicting the matrix. Cheap, practical protocol.
3. (Support) Replication of the headroom-gating (Yue et al.) and the single-family/Qwen
   confound (Shao et al.) in an under-studied domain: structured/tabular verifiable reasoning.

## Section plan
1. **Abstract / Intro.** Problem: when is post-training (RLVR) worth it on structured data?
   Contributions 1–3. The honest one-line result: *RLVR sharpens what the base can already
   sample and cannot cross a composite-arithmetic wall; single-family eval hides both.*
2. **Related work.** GRPO; Dr. GRPO (length bias) / DAPO (stability); RLVR amplifies-not-
   teaches + pass@k narrowing (Yue 2025); the Qwen spurious-reward confound (Shao 2025);
   the structured/tabular gap.
3. **Task suite & verifiable rewards.** 6 kinds over compositional depth 1–4; the
   de-confounding invariants (label rebalance to ~50/50; internally-consistent boxes;
   unique last names) with before/after numbers; the exact 0/1 reward.
4. **Method.** GRPO+LoRA+vLLM; recipes R0 (naive) / R1 (Dr. GRPO) / R2 (DAPO); the pass@k
   protocol; eval with Wilson CIs, paired McNemar, naive-baseline floors, per-class checks.
5. **Results.**
   (a) **Base pass@k = the gate** (Fig 1): per task × family; `frac_never`.
   (b) **Matrix** (Table 1/2): family × recipe overall Δ (multi-seed, mean±std, McNemar) +
   per-kind Δ vs floor.
   (c) **Headroom predicts gain** (Fig 2, the money figure): Δaccuracy vs base pass@8 —
   reachable tasks move, the arithmetic wall doesn't, consistently across families.
   (d) Confound: Qwen-only would read "no effect"; SmolLM2 reveals the real gain.
   (e) Q2: published fixes (Dr. GRPO/DAPO) do not beat naive GRPO here.
6. **Discussion.** The decomposable ceiling; pass@k-as-a-gate; relation to amplification/
   narrowing; *negative result + predictive tool* framing.
7. **Threats to validity.** Scale (1–1.7B, LoRA — scope it); synthetic (mitigated by a
   real-NFL held-out eval); seeds; #families; single domain (mitigated by a 2nd generator).
8. **Conclusion + artifact** (the harness, the labelers, the guard).
- **Appendix:** RLVR engineering pitfalls done right — colocate telemetry is unreliable
  (`clipped_ratio`), a NaN/no-op run that masqueraded as a clean +0.0, the fail-loud guard,
  the EOS-termination fix. (This is a credibility asset — reviewers trust careful authors.)

## Figures/tables that carry the paper
- **Fig 1** base pass@k per task×family (the gate).
- **Fig 2** Δacc vs base pass@8 — the predictive relationship (the headline figure).
- **Table 1** family×recipe overall Δ, mean±std over seeds, McNemar, vs floor.
- **Table 2** per-kind Δ matrix.
- **Table 3** de-confounding before/after (the methodology contribution made concrete).

## To clear the bar (≈$150–250, ~2 weekends)
Multi-seed (3+) clean-regime matrix · real-NFL held-out eval · 4 families (add Llama-3.2,
OLMo-2-SFT) · a 2nd structured domain (invoices/telemetry) · sharpen framing to 1+2.
