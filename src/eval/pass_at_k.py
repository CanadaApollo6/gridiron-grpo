"""Base-model pass@k per task -- the cheap, decisive learnability measurement.

GRPO can only reinforce reasoning the base model already samples with nonzero
probability. So before spending on the recipe matrix, measure, per task kind, the
unbiased pass@1 / pass@8 / pass@64 of the *base* model (Chen et al., 2021). A task
where base pass@8 ~ 0 is effectively unlearnable by GRPO -- every rollout group is
all-wrong, the advantage is zero, and no recipe (Dr. GRPO, DAPO, any beta) can
help. This reframes Q1 from "did GRPO fail?" to "was there anything to learn?".

  python src/eval/pass_at_k.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data data_out/eval.jsonl --label base --n_samples 64 --temperature 0.9
Writes results/passk_<label>.json.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import extract_answer  # noqa: E402
from rewards.verifiers import _check  # noqa: E402
from eval.stats import pass_at_k  # noqa: E402


# ---- pure core (unit-testable without a model) ----------------------------
def passk_summary(rows: list[dict], samples_per_row: list[list[str]],
                  ks=(1, 8, 64)) -> dict:
    """samples_per_row[i] = list of completion texts for rows[i].
    Returns overall + per-kind unbiased pass@k, plus the fraction of items the
    base model never solves (c==0) and always solves (c==n)."""
    per_kind_c = defaultdict(list)   # kind -> list of (n, c)
    overall = []
    for r, samples in zip(rows, samples_per_row):
        n = len(samples)
        c = sum(1 for s in samples if _check(extract_answer(s), r["ground_truth"], r["answer_type"]))
        per_kind_c[r["kind"]].append((n, c))
        overall.append((n, c, r["kind"]))

    def agg(pairs):
        out = {}
        for k in ks:
            out[f"pass@{k}"] = round(sum(pass_at_k(n, c, k) for n, c in pairs) / len(pairs), 4)
        out["frac_never"] = round(sum(1 for n, c in pairs if c == 0) / len(pairs), 4)
        out["frac_always"] = round(sum(1 for n, c in pairs if c == n) / len(pairs), 4)
        out["n_items"] = len(pairs)
        return out

    result = {"overall": agg([(n, c) for n, c, _ in overall]),
              "by_kind": {k: agg(v) for k, v in per_kind_c.items()}}
    # learnability flag: low pass@1 but high pass@k = headroom GRPO could exploit;
    # pass@k ~ 0 = unlearnable.
    return result


# ---- GPU sampling path -----------------------------------------------------
def sample_completions(model, tok, prompts, n_samples, temperature, max_new_tokens, bs=8):
    import torch
    out = [[] for _ in prompts]
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            rendered = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                        for p in chunk]
            enc = tok(rendered, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=temperature, top_p=1.0,
                                  num_return_sequences=n_samples, pad_token_id=tok.pad_token_id)
            start = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                for s in range(n_samples):
                    seq = gen[j * n_samples + s][start:]
                    out[i + j].append(tok.decode(seq, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data_out/eval.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N rows (pass@64 is costly).")
    args = ap.parse_args()

    from eval.evaluate import load_model
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    model, tok = load_model(args.model, args.adapter)
    samples = sample_completions(model, tok, [r["prompt"] for r in rows],
                                 args.n_samples, args.temperature, args.max_new_tokens)
    result = passk_summary(rows, samples)
    result.update({"label": args.label, "model": args.model, "adapter": args.adapter,
                   "n_samples": args.n_samples, "temperature": args.temperature})
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"passk_{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
