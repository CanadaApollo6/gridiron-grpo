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


def _name_match(pred: str, gt: str) -> bool:
    """Exact normalized match, OR a correct last-name-only answer. The generator
    guarantees unique last names within a box, so 'Smith' unambiguously identifies
    'A. Smith' -- accepting it avoids undercounting otherwise-correct answers."""
    p, g = _norm_name(pred), _norm_name(gt)
    if not g:
        return False
    if p == g:
        return True
    g_last = g.split()[-1]
    p_last = p.split()[-1] if p else ""
    return bool(p_last) and p_last == g_last


def _check(pred: str | None, gt: str, answer_type: str) -> bool:
    if pred is None:
        return False
    if answer_type == "numeric":
        pv, gv = _parse_number(pred), _parse_number(gt)
        return pv is not None and gv is not None and abs(pv - gv) < 1e-6
    if answer_type == "name":
        return _name_match(pred, gt)
    if answer_type == "set":
        return _norm_set(pred) == _norm_set(gt)
    if answer_type == "decision":
        p = _norm_name(pred)
        g = _norm_name(gt)
        if g == "fg":
            return ("fg" in p) or ("field goal" in p)
        if g == "td":
            return ("td" in p) or ("touchdown" in p)
        return p == g
    return False


def _numeric_partial(pred: str | None, gt: str) -> float:
    """Graded credit for a *close* numeric answer (training only; eval stays
    strict). Exact handled by the caller. Within 10% of the truth earns up to
    0.5, scaling linearly to 0 at the band edge. The cap (0.5 << 1.0 exact) and
    the tight band keep a 'guess the mean' policy from farming it."""
    pv, gv = _parse_number(pred), _parse_number(gt)
    if pv is None or gv is None:
        return 0.0
    band = max(1.0, abs(gv) * 0.10)
    err = abs(pv - gv)
    if err >= band:
        return 0.0
    return 0.5 * (1.0 - err / band)


def correctness_reward(prompts, completions, ground_truth=None, answer_type=None, **kwargs):
    """+1.0 if the extracted <answer> matches ground truth (strict), else 0.0."""
    out = []
    for i, comp in enumerate(completions):
        pred = extract_answer(_to_text(comp))
        out.append(1.0 if _check(pred, ground_truth[i], answer_type[i]) else 0.0)
    return out


def correctness_reward_graded(prompts, completions, ground_truth=None, answer_type=None, **kwargs):
    """Like correctness_reward but with partial credit on *numeric* tasks, to
    densify the signal where the strict 0/1 reward is sparse (REVIEW.md sparsity
    lever). 1.0 exact; up to 0.5 for numeric answers within 10%; set/name/decision
    stay strict. Opt-in via --graded_numeric; eval is always strict."""
    out = []
    for i, comp in enumerate(completions):
        pred = extract_answer(_to_text(comp))
        gt, at = ground_truth[i], answer_type[i]
        if _check(pred, gt, at):
            out.append(1.0)
        elif at == "numeric":
            out.append(_numeric_partial(pred, gt))
        else:
            out.append(0.0)
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
