"""Evaluation harness.

Scores a model on the held-out eval set using the SAME correctness check the
reward uses, so the headline number is honest. Reports overall accuracy and a
per-task-kind breakdown, and writes results/<label>.json.

Run the base model and the trained model with different --label values, then
make_chart.py turns the two JSONs into the before/after bar chart.

Example:
  python src/eval/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data data_out/eval.jsonl --label baseline
  python src/eval/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter runs/grpo-qwen15b --data data_out/eval.jsonl --label grpo
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_utils import pick_attn_impl  # noqa: E402
from prompts import extract_answer  # noqa: E402
from rewards.verifiers import _check  # noqa: E402


def load_model(model_name: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation=pick_attn_impl(), device_map="auto",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens=512, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data_out/eval.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    model, tok = load_model(args.model, args.adapter)

    completions = generate_batch(model, tok, [r["prompt"] for r in rows])

    by_kind_hits = defaultdict(int)
    by_kind_tot = defaultdict(int)
    hits = 0
    for r, c in zip(rows, completions):
        ok = _check(extract_answer(c), r["ground_truth"], r["answer_type"])
        hits += int(ok)
        by_kind_hits[r["kind"]] += int(ok)
        by_kind_tot[r["kind"]] += 1

    result = {
        "label": args.label,
        "model": args.model,
        "adapter": args.adapter,
        "n": len(rows),
        "accuracy": hits / len(rows),
        "by_kind": {k: by_kind_hits[k] / by_kind_tot[k] for k in by_kind_tot},
    }

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
