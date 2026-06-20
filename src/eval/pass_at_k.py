"""Base-model pass@k per task -- the cheap, decisive learnability measurement.

GRPO can only reinforce reasoning the base model already samples with nonzero
probability. So before spending on the recipe matrix, measure, per task kind, the
unbiased pass@1 / pass@8 / pass@64 of the *base* model (Chen et al., 2021). A task
where base pass@8 ~ 0 is effectively unlearnable by GRPO.

Memory: sampling is CHUNKED to `--gen_batch` sequences at a time (default 16), so
peak VRAM is bounded regardless of --n_samples. This fits a 10GB GPU (RTX 3080)
for 1-1.7B models; raise --gen_batch on a big GPU for speed. Inference only --
imports no vLLM/trl, so it runs on native Windows or WSL with a CUDA PyTorch.

  python src/eval/pass_at_k.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data data_out/eval.jsonl --label base --n_samples 64 --gen_batch 16 --limit 120
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
    per_kind_c = defaultdict(list)
    overall = []
    for r, samples in zip(rows, samples_per_row):
        n = len(samples)
        c = sum(1 for s in samples if _check(extract_answer(s), r["ground_truth"], r["answer_type"]))
        per_kind_c[r["kind"]].append((n, c))
        overall.append((n, c))

    def agg(pairs):
        out = {}
        for k in ks:
            out[f"pass@{k}"] = round(sum(pass_at_k(n, c, k) for n, c in pairs) / len(pairs), 4)
        out["frac_never"] = round(sum(1 for n, c in pairs if c == 0) / len(pairs), 4)
        out["frac_always"] = round(sum(1 for n, c in pairs if c == n) / len(pairs), 4)
        out["n_items"] = len(pairs)
        return out

    return {"overall": agg(overall), "by_kind": {k: agg(v) for k, v in per_kind_c.items()}}


# ---- GPU sampling path (chunked for bounded VRAM) --------------------------
def sample_completions(model, tok, prompts, n_samples, temperature, max_new_tokens,
                       gen_batch=16):
    """Generate n_samples completions per prompt, but never more than `gen_batch`
    sequences in flight at once -- so peak VRAM is set by gen_batch, not n_samples.
    One prompt at a time keeps memory predictable on small GPUs."""
    import torch
    out = []
    with torch.no_grad():
        for p in prompts:
            rendered = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            enc = tok([rendered], return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            start = enc["input_ids"].shape[1]
            samples, remaining = [], n_samples
            while remaining > 0:
                k = min(gen_batch, remaining)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=temperature, top_p=1.0,
                                     num_return_sequences=k, pad_token_id=tok.pad_token_id)
                for s in range(k):
                    samples.append(tok.decode(gen[s][start:], skip_special_tokens=True))
                remaining -= k
            out.append(samples)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data_out/eval.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--gen_batch", type=int, default=16,
                    help="sequences generated at once (caps VRAM). 16 fits ~10GB for "
                         "1-1.7B; raise it on a big GPU for speed.")
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
                                 args.n_samples, args.temperature, args.max_new_tokens,
                                 gen_batch=args.gen_batch)
    result = passk_summary(rows, samples)
    result.update({"label": args.label, "model": args.model, "adapter": args.adapter,
                   "n_samples": args.n_samples, "temperature": args.temperature})
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"passk_{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
