# Blog post outline — "I taught a 1.5B model to reason about football with RL, on one GPU"

Target: 1,500–2,200 words. Voice: practitioner showing the work, not a vendor pitch.
This doubles as a "sample of prior work" for the NVIDIA application AND a dry run
for the Dayton AI Day talk (the "when do you reach for post-training vs. prompting"
angle is the same).

---

## 1. Hook (150 words)
- The gap: everyone fine-tunes with SFT; far fewer have actually run RL post-training.
- The claim you're going to back up: a 1.5B model, GRPO, one rented GPU, under $200, measurable reasoning gain — fully reproducible.
- Show the before/after chart up top. Lead with the result.

## 2. Why verifiable rewards are the unlock (250 words)
- RLHF needs a learned reward model + preference data. That's the expensive, fragile part.
- For tasks with a checkable answer, the reward is just `is_correct()`. No reward model.
- Football box scores as the test bed: one right answer, legible, fun.
- Name the trade: synthetic data = perfect labels + infinite volume, at the cost of realism (address realism in the eval section).

## 3. The task and the data (250 words)
- Show one real generated example (data block + question + ground truth).
- The six task kinds and what each stresses (arithmetic, argmax, set membership, rule-based decision).
- The `<think>/<answer>` contract and why it matters for parsing the reward.

## 4. GRPO in plain terms (300 words)
- One paragraph, no equations: sample a group of answers per question, score them, push toward the better ones relative to the group average. No value network, no reward model.
- Contrast with PPO (heavier) and DPO (needs preference pairs).
- The two reward functions; why the format bonus exists and the trap it creates (format gaming).

## 5. Making it run on one GPU (300 words)
- LoRA, bf16, Flash Attention, vLLM for rollouts.
- The honest bottleneck: generation, not gradients.
- Exact hardware + cost. Reproduce-it-yourself instructions.
- NVIDIA-stack framing: this is the accelerated-computing story in miniature.

## 6. Results, including what broke (350 words)
- The per-kind table, not just the average — this is the credible part.
- Where it improved most (likely the multi-step numeric + decision tasks).
- At least one honest failure: a task that didn't move, or the format-gaming episode and how you caught it via the per-kind breakdown.
- This section is what separates a practitioner from a tutorial.

## 7. Wrapping it as an agent (200 words)
- The tuned model as a callable tool in NeMo Agent Toolkit.
- Why "fine-tuned model → agent capability" is the interesting direction.

## 8. When would you actually do this? (200 words)
- The real lesson: post-training earns its cost when you have a verifiable objective and prompting plateaus. Otherwise prompt.
- This is the bridge to the Dayton AI Day / Cortex talk: structured data + LLMs, and knowing which tool fits.

## 9. Repro + links (50 words)
- Repo, one-command quickstart, license. Invite people to swap in their own domain.

---

## Distribution checklist
- [ ] Repo public on github.com/CanadaApollo6 with the README above
- [ ] Before/after chart committed
- [ ] 3–5 min Loom/YouTube walkthrough (JD explicitly asks for video samples)
- [ ] Post on LinkedIn + X + relevant Discords/forums (the advocacy motion itself)
- [ ] Link it in the NVIDIA application as a work sample
