<!--
  RESULTS TEMPLATE — paste this into the "## Results" section of README.md after your run.
  The <!-- ... --> blocks are editing guidance; they DON'T render on GitHub, so the
  published page stays clean. Delete them once you've filled the section if you like.

  Fastest path: run `python src/eval/results_to_md.py results/baseline.json results/grpo.json`
  to auto-print the table rows below, then write the prose by hand.
-->

## Results

<!-- One or two sentences. Lead with the headline number. Be specific and quantified.
     e.g. "GRPO lifted overall verifiable accuracy from 41% to 68% (+27pp) on a held-out
     set of 800 questions, on a single H100 in ~Xh for ~$Y." -->
GRPO lifted overall verifiable accuracy from **__%** to **__%** (**+__pp**) on a held-out set of **800** questions, trained on one **H100** in **~__h** for **~$__**.

| Metric | Base model | GRPO-tuned | Δ |
|---|---|---|---|
| **Overall** | **__._%** | **__._%** | **+__._pp** |
| Yards from scrimmage (numeric) | __._% | __._% | +__._pp |
| Team points (numeric) | __._% | __._% | +__._pp |
| Total touchdowns (numeric) | __._% | __._% | +__._pp |
| Most scrimmage yards (argmax) | __._% | __._% | +__._pp |
| 100+ yard receivers (set) | __._% | __._% | +__._pp |
| TD-or-FG (decision) | __._% | __._% | +__._pp |

![Base vs. GRPO accuracy](results/before_after.png)

### What moved, and what didn't

<!-- This is the credibility section. Hand-waving here reads as a tutorial; specifics read
     as a practitioner. Hit all three beats: -->

<!-- BEAT 1 — biggest win. Which task improved most and a one-line hypothesis why.
     Multi-step numeric and the decision task are the usual movers, since the verifiable
     reward rewards getting the chain right, not just the format. -->
- **Biggest gain:** ____ went from __% to __% — likely because ____.

<!-- BEAT 2 — an honest flat spot or regression. Every real run has one. A task that
     barely moved, or where the base was already saturated, or where the model overfit a
     pattern. Naming it is what makes the rest believable. -->
- **Didn't move:** ____ stayed near __% — ____.

<!-- BEAT 3 — the format-gaming check. State explicitly that you watched for the model
     learning the <think>/<answer> shape WITHOUT improving correctness. Report whether
     format-completion and correctness moved together or diverged. -->
- **Reward-hacking check:** format-completion rose to ~__% while correctness ____ — confirming the gains are real reasoning, not the model gaming the format bonus. <!-- If they diverged, say so and what you did (lowered the format weight, etc.) -->

### Run metadata

<!-- Reproducibility = dev-advocate credibility. Fill every line. -->
- **Base model:** Qwen/Qwen2.5-1.5B-Instruct <!-- or whatever you ran -->
- **Method:** GRPO, LoRA (r=16, α=32), bf16, Flash Attention 2, vLLM rollouts
- **Reward:** correctness (+1.0 verifiable) + format (+0.2)
- **Training:** __ steps · group size 8 · LR 1e-6 · KL β 0.04
- **Data:** 8,000 synthetic train / 800 held-out eval, seed 7
- **Hardware:** 1× H100 80GB (rented, ____) · wall-clock ~__h · ~__ GPU-hours · ~$__
- **Eval realism:** <!-- "synthetic held-out" or, if you swapped it in, "real box scores via nfl_data_py" -->
