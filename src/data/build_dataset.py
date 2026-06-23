"""Build train/eval datasets as JSONL.

Each row:
  {
    "prompt": [ {role, content}, ... ],   # chat format; trainer applies template
    "ground_truth": "...",
    "answer_type": "numeric|name|set|decision",
    "kind": "<task kind>",
    "depth": <int>                          # compositional depth (Q3 taxonomy)
  }

Usage:
  python src/data/build_dataset.py --n_train 8000 --n_eval 800 --seed 7 --out data_out
  python src/data/build_dataset.py --domain invoices --n_train 8000 --n_eval 800 --out data_out

Domains: the pipeline is domain-agnostic -- the data layer is the only thing
that changes. Each domain exposes its own `sample_one(rng)` (a single source of
truth for that domain's ALL_TASKS); `--domain` just picks which one to draw
from. Everything downstream (prompt format, rewards, eval) is shared and
UNCHANGED. Default is football, and the football output is byte-identical to
before this flag existed.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import build_prompt  # noqa: E402


def get_sample_one(domain: str):
    """Return the domain's `sample_one` factory. Each domain keeps a single
    source of truth for its ALL_TASKS / sample_one; this just dispatches."""
    if domain == "football":
        from data.tasks import sample_one  # noqa: E402
        return sample_one
    if domain == "invoices":
        from data.invoices_tasks import sample_one  # noqa: E402
        return sample_one
    raise ValueError(f"unknown domain: {domain!r} (expected 'football' or 'invoices')")


def build(n: int, rng: random.Random, sample_one) -> list[dict]:
    rows = []
    for _ in range(n):
        s = sample_one(rng)
        rows.append({
            "prompt": build_prompt(s.context, s.question),
            "ground_truth": s.answer,
            "answer_type": s.answer_type,
            "kind": s.kind,
            "depth": s.depth,
        })
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, default="football",
                    choices=["football", "invoices"],
                    help="which data layer to draw from (default: football)")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_eval", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data_out")
    args = ap.parse_args()

    sample_one = get_sample_one(args.domain)

    out = Path(args.out)
    # separate RNG streams so eval is disjoint and reproducible
    train = build(args.n_train, random.Random(args.seed), sample_one)
    eval_ = build(args.n_eval, random.Random(args.seed + 10_000), sample_one)

    write_jsonl(train, out / "train.jsonl")
    write_jsonl(eval_, out / "eval.jsonl")

    from collections import Counter
    dist = Counter(r["kind"] for r in eval_)
    print(f"wrote {len(train)} train, {len(eval_)} eval to {out}/ (domain={args.domain})")
    print("eval kind distribution:", dict(dist))


if __name__ == "__main__":
    main()
