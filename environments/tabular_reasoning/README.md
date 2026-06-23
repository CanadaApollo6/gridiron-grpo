# tabular_reasoning (verifiers environment)

A [Prime Intellect `verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
environment for verifiable multi-step reasoning over **structured records** —
read a rendered record (a football box score, or an invoice), then commit one
machine-checkable answer in `<answer>...</answer>`. It loads with
`vf.load_environment("tabular-reasoning")`, the call NVIDIA
[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)'s `verifiers_agent` makes, so it
drops into NeMo Gym for rollout collection and RLVR training. The records are
fully synthetic and procedurally generated, with a controllable
compositional-difficulty ladder and exact (non-LLM) verifiers — so you can dial
difficulty and trust the reward, with no data downloads.

## What makes it worth a separate package

It doesn't re-implement the reward. `load_environment` imports the *exact*
checker (`rewards.verifiers`), prompt format (`prompts`), and procedural task
generators (`data.tasks`, `data.invoices_tasks`) the GRPO trainer in the parent
`gridiron-grpo` repo uses, and wraps each reward as a batch-of-one adapter over
the trainer's public reward functions. One verifier, two runtimes (TRL's
`GRPOTrainer` and NeMo Gym): an accuracy number measured here is the same number
the trainer optimizes, with no second implementation to drift out of sync. The
reward *logic* stays covered by the parent's `tests/test_verifiers.py`.

This is a **monorepo subpackage**, not a standalone Environments Hub package: it
expects the parent `gridiron-grpo` (this repo's `src/`) to be importable. It
finds it two ways — an in-tree path shim in `tabular_reasoning.py` (works
straight from a checkout), or `pip install -e .` at the repo root for an isolated
venv.

## Install and run

NeMo Gym's own guidance is to install verifiers environments in a **separate
venv** from the Gym servers (to avoid dependency conflicts). Same here:

```bash
# from the repo root
python -m venv .venv-env && source .venv-env/bin/activate
pip install -e .                              # parent: provides prompts/rewards/data
pip install -e environments/tabular_reasoning  # this environment
```

Quick check that it loads and scores (no model/server needed):

```bash
pytest environments/tabular_reasoning/tests -q
```

### In NeMo Gym

```bash
# 1) generate a rollout-input JSONL from the environment's dataset
#    (run from a venv that has this env installed)
cd /path/to/Gym/responses_api_agents/verifiers_agent
python scripts/create_dataset.py --env-id tabular-reasoning --size 100 \
    --output data/tabular-reasoning-example.jsonl

# 2) copy the drop-in agent config into the Gym checkout
cp /path/to/gridiron-grpo/environments/tabular_reasoning/configs/tabular_reasoning.yaml \
    responses_api_agents/verifiers_agent/configs/tabular_reasoning.yaml

# 3) start the Gym servers (verifiers_agent + your served policy model)
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/tabular_reasoning.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# 4) collect rollouts
ng_collect_rollouts +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/tabular-reasoning-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/tabular-reasoning-example-rollouts.jsonl \
    +limit=1
```

## `load_environment` arguments

| arg | default | meaning |
|---|---|---|
| `domain` | `"football"` | `"football"` or `"invoices"` (the second structured domain — same pipeline, different data layer) |
| `num_train_examples` | `2000` | examples drawn for the train split (generators are procedural) |
| `num_eval_examples` | `800` | examples drawn for the eval split (disjoint RNG stream) |
| `seed` | `7` | base RNG seed (eval uses `seed + 10_000`, matching `build_dataset.py`) |
| `graded_numeric` | `False` | partial credit for close numeric answers (a training-time densification lever). Keep `False` for eval — eval is always strict 0/1 |
| `use_format_reward` | `True` | include the small `<think>/<answer>` format-shaping reward (0.2 / 0.1 / 0.0), as the trainer does unless `--no_format_reward` |

Pass these through NeMo Gym's `vf_env_args`, e.g. `vf_env_args: {domain: invoices}`.

## Notes

- **Pin.** NeMo Gym currently tracks verifiers `v0.1.14`; this env is written
  against that API (`SingleTurnEnv` / `Rubric` / `Parser`). The lower bound in
  `pyproject.toml` is `>=0.1.8`.
- **License.** MIT, matching the parent repo. If the NeMo Gym PR requires
  Apache-2.0 for contributions, this is small and self-contained enough to
  relicense at that point.
