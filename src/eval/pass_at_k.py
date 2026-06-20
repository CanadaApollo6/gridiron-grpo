"""Base-model pass@k per task -- the cheap, decisive learnability measurement.

GRPO can only reinforce reasoning the base model already samples with nonzero
probability. So before spending on the recipe matrix, measure, per task kind, the
unbiased pass@1 / pass@8 / pass@64 of the *base* model (Chen et al., 2021). A task
where base pass@8 ~ 0 is effectively unlearnable by GRPO.

Memory: sampling is CHUNKED to `--gen_batch` sequences at a time (default 16), so
peak VRAM is bounded regardless of --n_samples. Fits a 10GB GPU (RTX 3080) for
1-1.7B models. Inference only -- imports no vLLM/trl, so it runs on native Windows
or WSL with a CUDA PyTorch.

SPEED: a batched generate() runs until its SLOWEST member stops, so one rambling
sample drags a whole chunk to --max_new_tokens. On small models that dominates
wall-clock; cap --max_new_tokens (256-512 is plenty for these short answers) to
keep it bounded. Progress + a per-prompt avg length + ETA print as it runs.

  python src/eval/pass_at_k.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data data_out/eval.jsonl --label base --n_samples 64 --max_new_tokens 512 --limit 120
Writes results/passk_<label>.json.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import extract_answer  # noqa: E402
from rewards.verifiers import _check  # noqa: E402
from eval.stats import pass_at_k  # noqa: E402


# ---- pure core (unit-testable without a model) ----------------------------
def passk_summary(rows: list[dict], samples_per_row: list[list[str]],
                  ks=(1, 8, 64)) -> dict:
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


# ---- GPU sampling path (chunked for bounded VRAM; prints progress) ---------
def sample_completions(model, tok, prompts, n_samples, temperature, max_new_tokens,
                       gen_batch=16, progress=True):
    """Generate n_samples completions per prompt, <= gen_batch sequences in flight.
    Prints per-prompt progress with average generated length (watch for it pinning
    at max_new_tokens = rambling) and a running ETA."""
    import torch
    out = []
    n_prompts = len(prompts)
    t0 = time.time()
    eos = tok.eos_token_id
    with torch.no_grad():
        for i, p in enumerate(prompts):
            rendered = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            enc = tok([rendered], return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            start = enc["input_ids"].shape[1]
            samples, gen_lens, remaining = [], [], n_samples
            while remaining > 0:
                k = min(gen_batch, remaining)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=temperature, top_p=1.0,
                                     num_return_sequences=k, pad_token_id=tok.pad_token_id)
                for s in range(k):
                    new = gen[s][start:]
                    # generated length = up to first EOS, else full (rambled to cap)
                    nz = (new == eos).nonzero()
                    glen = int(nz[0]) + 1 if len(nz) else int(new.shape[0])
                    gen_lens.append(glen)
                    samples.append(tok.decode(new, skip_special_tokens=True))
                remaining -= k
            out.append(samples)
            if progress:
                done = i + 1
                el = time.time() - t0
                eta = (el / done) * (n_prompts - done)
                avg_len = sum(gen_lens) / len(gen_lens)
                print(f"[passk] {done}/{n_prompts} prompts | avg_len={avg_len:.0f}tok"
                      f"{' (CAP-bound!)' if avg_len > 0.9 * max_new_tokens else ''} | "
                      f"elapsed {el/60:.1f}m | eta ~{eta/60:.1f}m", flush=True)
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
                    help="sequences generated at once (caps VRAM). 16 fits ~10GB for 1-1.7B.")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="cap on generated tokens. These tasks have short answers; 256-512 "
                         "is plenty and keeps the slow-tail bounded. 1024 can be very slow.")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N rows (pass@64 is costly).")
    ap.add_argument("--quiet", action="store_true", help="suppress per-prompt progress")
    args = ap.parse_args()

    from eval.evaluate import load_model
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[passk] {args.model} | {len(rows)} prompts x {args.n_samples} samples "
          f"| max_new={args.max_new_tokens} gen_batch={args.gen_batch}", flush=True)
    model, tok = load_model(args.model, args.adapter)
    samples = sample_completions(model, tok, [r["prompt"] for r in rows],
                                 args.n_samples, args.temperature, args.max_new_tokens,
                                 gen_batch=args.gen_batch, progress=not args.quiet)
    result = passk_summary(rows, samples)
    result.update({"label": args.label, "model": args.model, "adapter": args.adapter,
                   "n_samples": args.n_samples, "temperature": args.temperature,
                   "max_new_tokens": args.max_new_tokens})
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"passk_{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
