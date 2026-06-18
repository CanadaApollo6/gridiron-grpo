# gridiron-grpo

**Teach a small language model to _reason_ over structured data — with reinforcement learning, on a single GPU, for the price of lunch.**

Most "fine-tune an LLM" tutorials stop at supervised fine-tuning: show the model thousands of good answers and hope it imitates them. This repo does something more interesting and far less common in example code — **reinforcement learning with a verifiable reward**. The model generates its own attempts, each attempt is _checked against ground truth_, and the ones that got it right get reinforced. No human labeling. No reward model to train. Just: did you get the right answer, yes or no.

The deeper point, and the reason this is worth your afternoon: when a model is bad at a task, the instinct is to reach for a bigger model. Often the better move is to _teach the one you have_ — with the right objective and a clean signal. This is a small, legible demonstration of that idea end to end.

The domain is football (box scores, late-game situations) because verifiable sports questions are a perfect teaching sandbox — _"which player had the most total yards from scrimmage"_ has exactly one correct answer you can compute. But the pipeline is domain-agnostic. Swap the data generator and you've got a structured-data reasoning trainer for invoices, lab results, telemetry, anything with checkable answers.

## What you'll learn

By reading the code and running it once, you'll come away understanding:

- **How GRPO actually works** — the RL method behind DeepSeek-R1 — without the math-paper overhead.
- **Why verifiable rewards are a cheat code** — when your task has a checkable answer, you skip the single most expensive, fragile part of RLHF (the learned reward model).
- **How to keep RL fine-tuning on one GPU** — LoRA, bf16, and vLLM-accelerated rollouts, with where the real bottleneck hides.
- **How to evaluate honestly** — measuring accuracy _by task type_, and how to catch a model that's gaming your reward instead of actually improving.
- **How to adapt it to your own domain** — the data layer is the only thing you need to touch.

## The result, in one glance

> _Fill in after your run — `python src/eval/results_to_md.py results/baseline.json results/grpo.json` prints the table; the run metadata is your reproducibility receipt._

**‹one-line takeaway: base → tuned overall accuracy, the gain, and the standout task›**

![base vs. GRPO-tuned](results/before_after.png)

| Task            | What it tests                            |  Base |  GRPO | Δ (pp) |
| --------------- | ---------------------------------------- | ----: | ----: | -----: |
| **Overall**     | —                                        | `__%` | `__%` |  `+__` |
| scrimmage_total | single-player rush + rec sum             | `__%` | `__%` |  `+__` |
| team_points     | reconstruct points from the scoring line | `__%` | `__%` |  `+__` |
| total_tds       | sum touchdowns across players            | `__%` | `__%` |  `+__` |
| most_scrimmage  | argmax over the table                    | `__%` | `__%` |  `+__` |
| hundred_yd_rec  | set membership (≥100 rec yds)            | `__%` | `__%` |  `+__` |
| td_or_fg        | rule-based decision                      | `__%` | `__%` |  `+__` |

**Trained on:** `Qwen/Qwen2.5-1.5B-Instruct` · GRPO + LoRA (r=16) · `__` steps · 1× H100 · ~`__h` · **~$`__`**

## Quickstart

Three commands from clone to a charted before/after:

```bash
pip install -r requirements.txt

# 1. Generate verifiable training + eval data (synthetic, seeded, no downloads)
python src/data/build_dataset.py --n_train 8000 --n_eval 800 --seed 7 --out data_out

# 2. Train — runs a 20-step smoke test first, then the real run
bash scripts/run_train.sh Qwen/Qwen2.5-1.5B-Instruct

# 3. Evaluate base vs. tuned, and chart it
bash scripts/run_eval.sh Qwen/Qwen2.5-1.5B-Instruct runs/grpo-qwen15b
```

That's the whole loop. The rest of this README explains _why_ each piece is the way it is.

---

## The idea, taught

### Verifiable rewards: the cheat code

Classic RLHF needs a **reward model** — a second neural network trained on human preference data to score outputs. That's expensive, slow, and a research project in its own right.

But some tasks have a _right answer you can check with code_. "What's 119 + 29?" "Which row has the max?" "Is a field goal enough to take the lead?" For those, the reward function is just a Python function: `is_the_answer_correct() → 1.0 or 0.0`. No reward model, no labeling, no ambiguity. That's what makes RL tractable on a laptop budget — and football box scores are a clean, fun source of exactly these checkable questions.

### GRPO in plain terms

[GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) is refreshingly simple once you strip the notation:

1. For each question, the model generates a **group** of answers (here, 8).
2. Each answer is **scored** by the reward function.
3. The model is nudged **toward the answers that beat the group's average** and away from the ones below it.

No value network, no separate critic, no reward model. The "baseline" each answer is judged against is just _how its siblings did on the same question_. That's the whole trick, and it's why GRPO is the method that made small-model reasoning practical.

This repo uses two reward functions together:

- **correctness** (`+1.0` / `0.0`) — parses the model's `<answer>` and checks it: numeric tolerance, order-insensitive sets, normalized names, paraphrase-tolerant decisions.
- **format** (a small `+0.2`) — rewards a clean `<think>…</think><answer>…</answer>` structure so the early reward signal isn't hopelessly sparse. (This bonus is also a trap — see Caveats.)

### Why a made-up dataset

The training data is **synthetic and seeded**: box scores and game states are generated, and every answer is _computed_ from them, never hand-labeled. Three reasons that's the right call for learning RL:

1. **Infinite data** — generate as much as you want.
2. **Perfect labels** — the ground truth is correct by construction, so the reward is exact.
3. **No licensing or scraping** — nothing to download, nothing to attribute.

When you want a _credible headline number_, point the **eval** set at real stat lines (e.g. `nfl_data_py` / nflverse) formatted to match the generator. "Trained on synthetic verifiable tasks, evaluated on real games" is an honest and strong story.

---

## How it's built

```
synthetic data ──▶ verifiable reward ──▶ GRPO loop ──▶ tuned model ──▶ agent tool
 src/data/          src/rewards/          src/train_grpo.py            agent/
```

| Piece            | File                                         | What it does                                                                              |
| ---------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Data generators  | `src/data/generators.py`, `tasks.py`         | Seeded box scores / game states → 6 verifiable task types                                 |
| Reward functions | `src/rewards/verifiers.py`                   | correctness + format, the contract the model is trained against                           |
| Training         | `src/train_grpo.py`                          | TRL `GRPOTrainer` + LoRA + vLLM rollouts                                                  |
| Eval harness     | `src/eval/evaluate.py`                       | Same correctness check as training → honest accuracy, by task type                        |
| Reporting        | `src/eval/make_chart.py`, `results_to_md.py` | The before/after chart and the markdown table                                             |
| Agent wrapper    | `agent/`                                     | Wraps the tuned model as a callable tool (example integration: NVIDIA NeMo Agent Toolkit) |

## Make it yours

The whole point is that you fork this and point it at _your_ problem. The only file you need to change is the data layer:

1. Replace the generators in `src/data/` with your domain — invoices, support tickets, sensor readings, anything with a checkable answer.
2. Make sure each example yields a `ground_truth` and an `answer_type` (`numeric`, `name`, `set`, or `decision` — or add your own and extend `verifiers.py`).
3. Everything downstream — reward, GRPO loop, eval, charts — works unchanged.

If you build something with it, I'd genuinely love to see it.

## Running it (any GPU)

Vendor-neutral by design — rent an H100 wherever you like:

- A **1.5–3B** model + LoRA + group size 8 fits comfortably on one **H100 80GB**; an A100 40GB works with a smaller group or shorter completions.
- **bf16** throughout, with **Flash Attention 2** when it's installed and an automatic fallback to PyTorch `sdpa` when it isn't (see `src/model_utils.py`); **vLLM** serves the rollouts (the real throughput bottleneck — see below).
- End-to-end reproduction runs **well under $200**, often a fraction of that. RL post-training really has gotten this cheap.

### Run it on Hugging Face Jobs

If you don't want to manage a box, [HF Jobs](https://huggingface.co/docs/hub/jobs) runs the whole pipeline on rented hardware billed by the minute. `scripts/hf_job.sh` is self-contained — it clones this repo inside the container, installs deps, builds data, runs the 20-step smoke test, trains, evaluates, charts, and **pushes the LoRA adapter + results to a Hub repo** (the container's disk is wiped when the Job ends, so this upload is the point).

```bash
# one-time: a HF account with a positive credit balance (Pro unlocks Jobs;
# GPU-minutes are pay-as-you-go). The CLI ships with huggingface_hub:
uvx --from huggingface_hub hf auth login

# launch on a single A100 80GB ($2.50/hr); --secrets HF_TOKEN lets the job push results
uvx --from huggingface_hub hf jobs run \
    --flavor a100-large --timeout 4h --secrets HF_TOKEN \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c "$(curl -fsSL https://raw.githubusercontent.com/CanadaApollo6/gridiron-grpo/main/scripts/hf_job.sh)"
```

Tune via `-e MAX_STEPS=...`, `-e MODEL=...`, `-e REPO_NAME=...`, or `-e NO_VLLM=1`. Cheaper hardware: `--flavor l40sx1` (48 GB, $1.80/hr). Watch it with `hf jobs logs <id>`; pull the result with `hf download <namespace>/gridiron-grpo-qwen15b --local-dir ./out`.

**Validate the Job env for a few cents first.** Add `-e SMOKE_ONLY=1` (and drop `--secrets HF_TOKEN`) to run only the data build + 20-step smoke on the real hardware — a ~5-minute, few-cents confirmation that the pinned stack and TRL GRPO API are happy before you commit to the full 4-hour run.

### Validate locally first (free, Linux/WSL)

> **Native Windows won't work:** vLLM has no Windows build and `trl>=0.16` imports vLLM at load, so GRPO can't even import there. Use **WSL2** (your local GPU is available via the Windows NVIDIA driver — no extra setup) or any Linux box.

On a CUDA GPU (≥8 GB), `bash scripts/smoke_local.sh` builds a uv venv with the pinned stack and runs a 5-step GRPO smoke on Qwen2.5-0.5B. If it prints "saved LoRA adapter", the pipeline works and the real Job is safe to launch. The 3080 can also do scaled-down _real_ runs (0.5B, smaller group); the 1.5B/group-8 headline config wants the rented A100.

## Caveats (read before you spend GPU hours)

I'd rather you learn these from a paragraph than from a wasted run:

- **TRL's GRPO API moves between releases.** Pin the versions in `requirements.txt` and let the 20-step smoke test confirm the `GRPOConfig` / reward-function argument names on _your_ install before the real run.
- **GRPO is rollout-heavy.** Wall-clock is dominated by generation, not the gradient step. This is why vLLM matters and why "RL is slow" is usually really "generation is slow."
- **Small models will game the format reward.** If `<think>/<answer>` adherence shoots to ~100% while _correctness_ stays flat, the model learned the shape without learning to be right. Watch the per-task breakdown, not the average, and lower the format bonus if it happens.
- **The knobs that matter most** are LoRA rank, KL `beta`, and group size. Change one at a time.

---

## About

Built by **Riel St. Amand** as a hands-on companion to talks and writing on agentic AI and small-model reasoning. The accompanying walkthrough is in [`BLOG_OUTLINE.md`](BLOG_OUTLINE.md).

**License:** MIT. Football data is synthetic — no real NFL data ships in this repo.
