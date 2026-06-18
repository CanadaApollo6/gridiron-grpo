"""Reward functions for GRPO.

TRL's GRPOTrainer calls each reward function as:
    reward_func(prompts, completions, **kwargs) -> list[float]
where any extra dataset columns (here: ground_truth, answer_type) are passed
through **kwargs as parallel lists. `completions` is a list of either strings
(standard format) or chat message-lists (conversational format); _to_text
handles both.

IMPORTANT: the GRPO reward-function calling convention has shifted across TRL
versions. Pin TRL (see requirements.txt) and sanity-check the signature with a
tiny run before a long one.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import extract_answer, has_reasoning_format  # noqa: E402


def _to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{"role": "assistant", "content": "..."}]
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def _parse_number(s: str):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else None


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace(".", ""))


def _norm_set(s: str) -> set[str]:
    if s is None:
        return set()
    raw = re.split(r"[,\n;]", s)
    items = {_norm_name(x) for x in raw if x.strip()}
    items.discard("none")
    return items


def _check(pred: str | None, gt: str, answer_type: str) -> bool:
    if pred is None:
        return False
    if answer_type == "numeric":
        pv, gv = _parse_number(pred), _parse_number(gt)
        return pv is not None and gv is not None and abs(pv - gv) < 1e-6
    if answer_type == "name":
        return _norm_name(pred) == _norm_name(gt)
    if answer_type == "set":
        return _norm_set(pred) == _norm_set(gt)
    if answer_type == "decision":
        p = _norm_name(pred)
        g = _norm_name(gt)
        # accept common phrasings
        if g == "fg":
            return ("fg" in p) or ("field goal" in p)
        if g == "td":
            return ("td" in p) or ("touchdown" in p)
        return p == g
    return False


def correctness_reward(prompts, completions, ground_truth=None, answer_type=None, **kwargs):
    """+1.0 if the extracted <answer> matches ground truth, else 0.0."""
    out = []
    for i, comp in enumerate(completions):
        text = _to_text(comp)
        pred = extract_answer(text)
        gt = ground_truth[i]
        at = answer_type[i]
        out.append(1.0 if _check(pred, gt, at) else 0.0)
    return out


def format_reward(prompts, completions, **kwargs):
    """Small shaping reward: 0.2 for a clean <think>...</think><answer>...</answer>
    structure, 0.1 if at least an <answer> block is present, else 0.0.
    Keeps reward signal less sparse early in training."""
    out = []
    for comp in completions:
        text = _to_text(comp)
        if has_reasoning_format(text):
            out.append(0.2)
        elif extract_answer(text) is not None:
            out.append(0.1)
        else:
            out.append(0.0)
    return out
