"""Evaluation harness.

Scores a model on the held-out eval set using the SAME strict correctness check
the reward uses, so the headline number is honest. Beyond a single accuracy, it
reports the things that make per-task claims defensible (see REVIEW.md):

  * Wilson 95% CIs on overall and per-kind accuracy (at n~130/kind, a 5pp delta
    is inside the noise -- show the bar).
  * best-constant / majority baselines per kind, so "accuracy moved" is judged
    against the naive floor, not against 0. (Crucial for the imbalanced decision
    and set tasks.)
  * per-class breakdown for the decision and set tasks, to catch majority-class
    collapse masquerading as learning.
  * terminated fraction (did the model emit EOS, or run into the cap?) -- the
    native colocate length telemetry is unreliable, so we measure it here.
  * a per-item dump (kind, depth, correct) so compare.py can run a paired
    McNemar test base-vs-tuned.

Example:
  python src/eval/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data data_out/eval.jsonl --label baseline
  python src/eval/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter runs/grpo-qwen15b --data data_out/eval.jsonl --label grpo
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.stats import wilson  # noqa: E402 (pass_at_k re-exported for convenience)
from prompts import extract_answer  # noqa: E402
from rewards.verifiers import _check, _norm_name, _norm_set  # noqa: E402

try:
    from data.tasks import KIND_DEPTH  # depth fallback for older eval files
except Exception:
    KIND_DEPTH = {}


# ---------------------------------------------------------------------------
# Pure scoring (no model) -- unit-testable on synthetic completions.
# ---------------------------------------------------------------------------
def summarize(rows: list[dict], preds: list[str], terminated: list[bool] | None = None) -> dict:
    """rows: eval rows (ground_truth, answer_type, kind, [depth]).
    preds: model completion text per row. terminated: optional EOS flags.
    Returns the full results dict (minus run metadata)."""
    n = len(rows)
    items = []
    by_kind_hits, by_kind_tot = defaultdict(int), defaultdict(int)
    by_depth_hits, by_depth_tot = defaultdict(int), defaultdict(int)
    hits = 0
    for r, c in zip(rows, preds, strict=False):
        pred = extract_answer(c)
        ok = bool(_check(pred, r["ground_truth"], r["answer_type"]))
        kind = r["kind"]
        depth = r.get("depth", KIND_DEPTH.get(kind))
        hits += ok
        by_kind_hits[kind] += ok
        by_kind_tot[kind] += 1
        by_depth_hits[depth] += ok
        by_depth_tot[depth] += 1
        items.append(
            {"kind": kind, "depth": depth, "correct": ok, "pred": pred, "gt": r["ground_truth"]}
        )

    def ci(h, t):
        lo, hi = wilson(h, t)
        return [round(lo, 4), round(hi, 4)]

    # best-constant baseline per kind = accuracy of always guessing that kind's
    # most common ground-truth answer (the naive, no-reasoning floor).
    best_constant = {}
    for kind in by_kind_tot:
        gts = [r["ground_truth"] for r in rows if r["kind"] == kind]
        top = Counter(gts).most_common(1)[0][1]
        best_constant[kind] = round(top / len(gts), 4)
    # overall majority baseline: an oracle that knows the kind and always guesses
    # its majority answer (weighted by kind frequency).
    majority_overall = round(sum(best_constant[k] * by_kind_tot[k] for k in by_kind_tot) / n, 4)

    result = {
        "n": n,
        "accuracy": round(hits / n, 4),
        "accuracy_ci95": ci(hits, n),
        "by_kind": {k: round(by_kind_hits[k] / by_kind_tot[k], 4) for k in by_kind_tot},
        "by_kind_ci95": {k: ci(by_kind_hits[k], by_kind_tot[k]) for k in by_kind_tot},
        "by_kind_n": dict(by_kind_tot),
        "by_depth": {
            str(d): round(by_depth_hits[d] / by_depth_tot[d], 4)
            for d in sorted(by_depth_tot, key=lambda x: (x is None, x))
        },
        "best_constant_by_kind": best_constant,
        "majority_baseline_overall": majority_overall,
        "per_class": _per_class(rows, preds),
        "items": items,
    }
    if terminated is not None and len(terminated) == n:
        result["terminated_fraction"] = round(sum(terminated) / n, 4)
    return result


def _per_class(rows: list[dict], preds: list[str]) -> dict:
    """Catch majority-class collapse on the imbalanced tasks."""
    out = {}
    # decision task: per-class recall + the model's predicted-class distribution
    dec = [
        (r, extract_answer(c)) for r, c in zip(rows, preds, strict=False) if r["kind"] == "td_or_fg"
    ]
    if dec:
        recall, pred_dist = {}, Counter()
        for cls in ("TD", "FG"):
            sub = [(r, p) for r, p in dec if r["ground_truth"] == cls]
            if sub:
                recall[cls] = round(
                    sum(_check(p, r["ground_truth"], "decision") for r, p in sub) / len(sub), 4
                )
        for r, p in dec:
            pn = _norm_name(p or "")
            pred_dist[
                "FG"
                if ("fg" in pn or "field goal" in pn)
                else "TD"
                if ("td" in pn or "touchdown" in pn)
                else "other"
            ] += 1
        out["td_or_fg"] = {"recall": recall, "pred_dist": dict(pred_dist)}
    # set task: none vs non-empty accuracy + mean Jaccard
    st = [
        (r, extract_answer(c))
        for r, c in zip(rows, preds, strict=False)
        if r["kind"] == "hundred_yd_rec"
    ]
    if st:
        none_sub = [(r, p) for r, p in st if r["ground_truth"] == "none"]
        ne_sub = [(r, p) for r, p in st if r["ground_truth"] != "none"]

        def acc(sub):
            return (
                round(sum(_check(p, r["ground_truth"], "set") for r, p in sub) / len(sub), 4)
                if sub
                else None
            )

        jac = []
        for r, p in st:
            g, pr = _norm_set(r["ground_truth"]), _norm_set(p or "")
            u = g | pr
            jac.append(1.0 if not u else len(g & pr) / len(u))
        out["hundred_yd_rec"] = {
            "none_acc": acc(none_sub),
            "nonempty_acc": acc(ne_sub),
            "n_none": len(none_sub),
            "n_nonempty": len(ne_sub),
            "mean_jaccard": round(sum(jac) / len(jac), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Model loading + generation (GPU path).
# ---------------------------------------------------------------------------
def load_model(model_name: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_utils import pick_attn_impl

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Decoder-only models MUST be left-padded for batched generation.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=pick_attn_impl(),
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def generate_batch(model, tok, prompts, max_new_tokens=1024, bs=16):
    """Greedy generation. Returns (texts, terminated_flags) where terminated is
    True iff the model emitted EOS (vs. running into max_new_tokens)."""
    import torch

    texts, terminated = [], []
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            chunk = prompts[i : i + bs]
            rendered = [
                tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in chunk
            ]
            enc = tok(
                rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
            )
            start = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                new = gen[j][start:]
                terminated.append(bool((new == tok.eos_token_id).any().item()))
                texts.append(tok.decode(new, skip_special_tokens=True))
    return texts, terminated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data_out/eval.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--bs", type=int, default=16, help="generation batch size; lower (e.g. 4-8) for a 10GB GPU."
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="cap on generated tokens at eval; default matches the "
        "study's 1024 completion budget so eval isn't truncated.",
    )
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    model, tok = load_model(args.model, args.adapter)
    preds, terminated = generate_batch(
        model, tok, [r["prompt"] for r in rows], max_new_tokens=args.max_new_tokens, bs=args.bs
    )

    result = summarize(rows, preds, terminated)
    result.update(
        {
            "label": args.label,
            "model": args.model,
            "adapter": args.adapter,
            "recipe": {
                "max_new_tokens": args.max_new_tokens,
                "decoding": "greedy",
                "data": args.data,
            },
        }
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    # console: the headline + the floor it must beat
    print(
        f"{args.label}: acc={result['accuracy']:.1%} "
        f"CI{tuple(round(x * 100, 1) for x in result['accuracy_ci95'])} | "
        f"majority floor={result['majority_baseline_overall']:.1%} | "
        f"terminated={result.get('terminated_fraction')}"
    )
    for k in sorted(result["by_kind"]):
        print(
            f"  {k:16s} {result['by_kind'][k]:.1%}  floor={result['best_constant_by_kind'][k]:.1%}"
        )


if __name__ == "__main__":
    main()
