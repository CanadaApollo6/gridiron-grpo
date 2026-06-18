"""Prompt construction and output parsing.

The model is trained to think inside <think>...</think> and then commit to a
final, machine-checkable answer inside <answer>...</answer>. The reward
functions in src/rewards/verifiers.py parse the <answer> block, so the format
contract here and the parsing there must stay in sync.
"""

import re

SYSTEM_PROMPT = (
    "You are a precise football analytics assistant. You are given a block of "
    "structured data (a box score or a game situation) and a question.\n"
    "First reason step by step inside <think> </think>. Then give ONLY the final "
    "answer inside <answer> </answer>.\n"
    "Rules for the answer:\n"
    "- Numeric answers: a single number, no units, no commas (e.g. <answer>137</answer>).\n"
    "- 'Which player' answers: the player's full name exactly as written in the data.\n"
    "- List answers: comma-separated names (e.g. <answer>A. Smith, J. Doe</answer>).\n"
    "- Decision answers: a single word (e.g. <answer>TD</answer> or <answer>FG</answer>).\n"
)

USER_TEMPLATE = "DATA:\n{context}\n\nQUESTION: {question}"


def build_prompt(context: str, question: str) -> list[dict]:
    """Return chat-format messages. The trainer/evaluator applies the
    tokenizer's chat template to this."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_FORMAT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE
)


def extract_answer(text: str) -> str | None:
    """Pull the last <answer>...</answer> payload (last, so trailing restated
    answers win). Returns None if absent."""
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def has_reasoning_format(text: str) -> bool:
    """True if the completion has a <think> block followed by an <answer> block."""
    return bool(_FORMAT_RE.search(text or ""))
