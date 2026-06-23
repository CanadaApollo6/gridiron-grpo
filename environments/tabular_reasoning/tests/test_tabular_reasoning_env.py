"""Smoke tests for the tabular_reasoning verifiers environment.

These need `verifiers` installed (and the parent gridiron-grpo src/ importable),
so they `importorskip` -- the repo's main suite (tests/) stays green without
verifiers. Run them in the env venv:

    pip install -e . && pip install -e environments/tabular_reasoning
    pytest environments/tabular_reasoning/tests -q

The reward functions are closures inside load_environment(), so we reach them
through `env.rubric._get_reward_funcs()` (verifiers' own flattened accessor) and
call them directly with a synthetic completion -- no model or server needed.
That is enough to prove the verifier is wired to the verifiers API correctly;
the reward *logic* itself is covered by the parent's tests/test_verifiers.py
(these adapters call those exact functions).
"""

import pytest

vf = pytest.importorskip("verifiers")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tabular_reasoning  # noqa: E402


def _completion(text: str):
    """A verifiers chat completion: a single assistant message."""
    return [{"role": "assistant", "content": text}]


# verifiers wraps a passed Rubric in a RubricGroup, so reach the flattened
# reward funcs/names through the accessors verifiers itself uses to score
# (`_get_reward_funcs` / `_get_reward_func_names`) -- robust to that wrapping.
def _rubric_funcs(env):
    return {f.__name__: f for f in env.rubric._get_reward_funcs()}


def _reward_names(env):
    return set(env.rubric._get_reward_func_names())


def test_load_environment_returns_singleturn_env():
    env = tabular_reasoning.load_environment(num_train_examples=8, num_eval_examples=8)
    assert isinstance(env, vf.SingleTurnEnv)
    # default reward surface: correctness + format shaping (the env also adds its
    # own built-in `num_turns` metric, so check our funcs are a subset).
    assert {"correctness_reward", "format_reward"} <= _reward_names(env)


def test_dataset_has_verifiers_columns_and_system_prompt():
    env = tabular_reasoning.load_environment(num_train_examples=8, num_eval_examples=8)
    ds = env.get_dataset(n=4)
    assert {"prompt", "answer", "info", "task"} <= set(ds.column_names)
    first = ds[0]
    # SingleTurnEnv built the chat prompt: system (our SYSTEM_PROMPT) + user
    assert first["prompt"][0]["role"] == "system"
    assert first["prompt"][-1]["role"] == "user"
    assert "answer_type" in first["info"]


def test_correctness_reward_strict_match():
    env = tabular_reasoning.load_environment(num_train_examples=8, num_eval_examples=8)
    correctness = _rubric_funcs(env)["correctness_reward"]
    good = _completion("<think>3 + 4 = 7</think><answer>137</answer>")
    bad = _completion("<think>hmm</think><answer>999</answer>")
    info = {"answer_type": "numeric"}
    assert correctness(completion=good, answer="137", info=info) == 1.0
    assert correctness(completion=bad, answer="137", info=info) == 0.0


def test_format_reward_tiers():
    env = tabular_reasoning.load_environment(num_train_examples=8, num_eval_examples=8)
    fmt = _rubric_funcs(env)["format_reward"]
    clean = _completion("<think>reasoning</think><answer>7</answer>")
    answer_only = _completion("<answer>7</answer>")
    neither = _completion("the answer is 7")
    assert fmt(completion=clean) == pytest.approx(0.2)
    assert fmt(completion=answer_only) == pytest.approx(0.1)
    assert fmt(completion=neither) == pytest.approx(0.0)


def test_graded_numeric_partial_credit():
    env = tabular_reasoning.load_environment(num_train_examples=8, num_eval_examples=8, graded_numeric=True)
    correctness = _rubric_funcs(env)["correctness_reward_graded"]
    info = {"answer_type": "numeric"}
    # within 10% of 100 -> partial credit in (0, 0.5]; exact -> 1.0
    close = _completion("<think>~</think><answer>105</answer>")
    exact = _completion("<think>~</think><answer>100</answer>")
    assert correctness(completion=exact, answer="100", info=info) == 1.0
    r = correctness(completion=close, answer="100", info=info)
    assert 0.0 < r <= 0.5


def test_invoices_domain_loads():
    env = tabular_reasoning.load_environment(domain="invoices", num_train_examples=8, num_eval_examples=8)
    ds = env.get_dataset(n=4)
    assert {"prompt", "answer", "info", "task"} <= set(ds.column_names)


def test_no_format_reward_drops_shaping_term():
    env = tabular_reasoning.load_environment(
        num_train_examples=8, num_eval_examples=8, use_format_reward=False
    )
    names = _reward_names(env)
    assert "correctness_reward" in names and "format_reward" not in names
