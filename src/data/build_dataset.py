"""Build train/eval datasets as JSONL.

Each row:
  {
    "prompt": [ {role, content}, ... ],   # chat format; trainer applies template
    "ground_truth": "...",
    "answer_type": "numeric|name|set|decision",
    "kind": "<task kind>"
  }

Usage:
  python src/data/build_dataset.py --n_train 8000 --n_eval 800 --seed 7 --out data_out
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.tasks import sample_one  # noqa: E402
from prompts import build_prompt  # noqa: E402


def build(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for _ in range(n):
        s = sample_one(rng)
        rows.append({
            "prompt": build_prompt(s.context, s.question),
            "ground_truth": s.answer,
            "answer_type": s.answer_type,
            "kind": s.kind,
        })
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_eval", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data_out")
    args = ap.parse_args()

    out = Path(args.out)
    # separate RNG streams so eval is disjoint and reproducible
    train = build(args.n_train, random.Random(args.seed))
    eval_ = build(args.n_eval, random.Random(args.seed + 10_000))

    write_jsonl(train, out / "train.jsonl")
    write_jsonl(eval_, out / "eval.jsonl")

    # quick distribution print
    from collections import Counter
    dist = Counter(r["kind"] for r in eval_)
    print(f"wrote {len(train)} train, {len(eval_)} eval to {out}/")
    print("eval kind distribution:", dict(dist))


if __name__ == "__main__":
    main()
