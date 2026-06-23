"""tabular_reasoning -- a NeMo Gym / verifiers environment.

Wraps the `gridiron-grpo` verifiable-reward task suite (reason over a structured
record -- a football box score, or an invoice -- then commit a single
machine-checkable answer inside <answer>...</answer>) as a Prime Intellect
`verifiers` environment. That makes it loadable by
`vf.load_environment("tabular-reasoning")`, which is exactly what NVIDIA NeMo
Gym's `verifiers_agent` calls to run an environment.

The point of this package is what it does NOT do: it does not reimplement the
reward. It imports the *exact* checker (`rewards.verifiers`), prompt format
(`prompts`), and task generators (`data.tasks`, `data.invoices_tasks`) the GRPO
trainer in this repo uses. One verifier, two runtimes (TRL's GRPOTrainer and
NeMo Gym) -- so an accuracy number measured here is the same number the trainer
optimizes, with no second implementation to drift.

Entry point: `load_environment(...) -> vf.SingleTurnEnv`.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import verifiers as vf

# --- share the trainer's verifier instead of duplicating it ----------------
# This repo uses a flat src/ layout (top-level modules `prompts`, packages
# `rewards`/`data`). When this env is run from inside the monorepo, src/ sits
# two levels up; add it to the path so the imports below resolve. When the
# parent `gridiron-grpo` package is installed separately (the standalone case),
# src/ won't be there and this is a harmless no-op -- the imports resolve from
# the installed package instead.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from prompts import (  # noqa: E402  (path shim must run first)
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    extract_answer,
)
from rewards import verifiers as _trl  # noqa: E402  (path shim must run first)

# domain -> module exposing `sample_one(rng) -> Sample`. Mirrors the --domain
# dispatch in src/data/build_dataset.py; both domains share the Sample shape.
_DOMAINS = {"football": "data.tasks", "invoices": "data.invoices_tasks"}


def _sample_one_for(domain: str):
    if domain not in _DOMAINS:
        raise ValueError(f"unknown domain {domain!r} (expected one of {sorted(_DOMAINS)})")
    return importlib.import_module(_DOMAINS[domain]).sample_one


def _build_split(sample_one, n: int, seed: int):
    """Materialize `n` examples into a verifiers-format HF dataset.

    Columns follow the verifiers contract:
    - `question`: the user turn; SingleTurnEnv wraps it with SYSTEM_PROMPT into
      the chat prompt (identical to prompts.build_prompt).
    - `answer`:   the reference answer (string).
    - `task`:     the task kind, so eval can slice per-kind (the Q3 taxonomy).
    - `info`:     per-example metadata. `answer_type` drives the checker, exactly
      as the `answer_type` column does in training.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = sample_one(rng)
        rows.append(
            {
                "question": USER_TEMPLATE.format(context=s.context, question=s.question),
                "answer": s.answer,
                "task": s.kind,
                "info": {"answer_type": s.answer_type, "kind": s.kind, "depth": s.depth},
            }
        )
    return Dataset.from_list(rows)


def load_environment(
    domain: str = "football",
    num_train_examples: int = 2000,
    num_eval_examples: int = 800,
    seed: int = 7,
    graded_numeric: bool = False,
    use_format_reward: bool = True,
) -> vf.Environment:
    """Build the tabular_reasoning verifiers environment.

    The args mirror the trainer's knobs so the reward surface matches:
    - `domain`: "football" (default) or "invoices" (the second structured
      domain -- same pipeline, different data layer).
    - `num_train_examples` / `num_eval_examples`: how many synthetic examples to
      generate per split (the generators are procedural, so this is just a draw
      count; train and eval use disjoint RNG streams as in build_dataset.py).
    - `seed`: base RNG seed (eval uses seed + 10_000, matching build_dataset.py).
    - `graded_numeric`: partial credit for close numeric answers (a training-time
      densification lever). Leave False for eval -- the repo's policy is that
      eval is always strict 0/1.
    - `use_format_reward`: include the small <think>/<answer> format-shaping
      reward (0.2 / 0.1 / 0.0), as the trainer does unless --no_format_reward.
    """
    sample_one = _sample_one_for(domain)

    def build_dataset():
        return _build_split(sample_one, num_train_examples, seed)

    def build_eval_dataset():
        return _build_split(sample_one, num_eval_examples, seed + 10_000)

    # One parser instance, shared by the env and the rubric (verifiers warns if
    # the two differ). Its extract_fn IS the trainer's answer extractor, so the
    # environment's notion of "the answer" agrees with the checker's.
    parser = vf.Parser(extract_fn=lambda text: extract_answer(text) or "")

    # The reward functions below are thin batch-of-one adapters over the repo's
    # *public* training rewards. verifiers scores one rollout at a time and wants
    # a scalar; TRL scores a batch and wants a list. So we wrap each completion
    # in a length-1 list and take element [0] -- the reward is byte-identical to
    # what the GRPOTrainer optimizes, and it stays covered by tests/test_verifiers.py.
    def correctness_reward(completion, answer, info, **kwargs) -> float:
        """1.0 iff the extracted <answer> matches ground truth (strict)."""
        return _trl.correctness_reward(
            None, [completion], ground_truth=[answer], answer_type=[info["answer_type"]]
        )[0]

    def correctness_reward_graded(completion, answer, info, **kwargs) -> float:
        """Strict 1.0, else up to 0.5 partial credit for close numeric answers."""
        return _trl.correctness_reward_graded(
            None, [completion], ground_truth=[answer], answer_type=[info["answer_type"]]
        )[0]

    def format_reward(completion, **kwargs) -> float:
        """0.2 for a clean <think>..</think><answer>..</answer>, 0.1 for an
        <answer> block alone, else 0.0 -- the trainer's format-shaping reward."""
        return _trl.format_reward(None, [completion])[0]

    correctness = correctness_reward_graded if graded_numeric else correctness_reward
    funcs = [correctness]
    weights = [1.0]
    if use_format_reward:
        funcs.append(format_reward)
        weights.append(1.0)  # format's small magnitude already encodes the shaping

    rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
