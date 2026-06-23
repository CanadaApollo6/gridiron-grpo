"""GRPO training entrypoint (TRL + PEFT/LoRA + vLLM rollouts).

Runs on a single H100 80GB for a ~1.5-3B model with LoRA. This file is written
to spec against TRL's GRPOTrainer; the GRPO API has moved across releases, so:
  1. install the pinned versions in requirements.txt,
  2. do a 20-step smoke run first (--max_steps 20) to confirm arg names,
  3. then launch the real run.

Example:
  python src/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data_out/train.jsonl --out runs/grpo-qwen15b --max_steps 1200
"""

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import pick_attn_impl  # noqa: E402
from rewards.verifiers import (  # noqa: E402
    correctness_reward,
    correctness_reward_graded,
    format_reward,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data_out/train.jsonl")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument(
        "--num_generations", type=int, default=8, help="rollouts per prompt (the 'group' in GRPO)"
    )
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        help="HF LR schedule. 'cosine'/'linear' decay toward 0; use "
        "'constant_with_warmup' for a fair does-it-learn probe so "
        "the back half of training still updates (REVIEW.md).",
    )
    ap.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="fraction of steps spent warming the LR up from 0",
    )
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--max_prompt_len", type=int, default=640)
    ap.add_argument(
        "--max_completion_len",
        type=int,
        default=1024,
        help="completion token budget; default matches the study design "
        "and the eval cap so train and eval agree.",
    )
    ap.add_argument("--per_device_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--no_vllm", action="store_true")
    ap.add_argument(
        "--vllm_mode",
        default="colocate",
        help="colocate = vLLM shares the training GPU (single-GPU runs); "
        "server = expects a separate `trl vllm-serve` process",
    )
    ap.add_argument(
        "--vllm_gpu_mem_util",
        type=float,
        default=0.30,
        help="fraction of GPU memory vLLM reserves for KV cache in "
        "colocate mode. Lower if training OOMs; raise if rollouts "
        "are KV-starved.",
    )
    ap.add_argument(
        "--seed", type=int, default=7, help="training seed (set per run for multi-seed rigor)"
    )
    # --- GRPO objective variants (the "known fixes" research axis) ----------
    ap.add_argument(
        "--loss_type",
        default="bnpo",
        choices=["grpo", "bnpo", "dr_grpo"],
        help="bnpo = TRL default; grpo = original length-normalized "
        "(has length bias); dr_grpo = length-bias-free (Dr. GRPO)",
    )
    ap.add_argument(
        "--no_scale_rewards",
        action="store_true",
        help="disable std reward scaling (the other half of Dr. GRPO; removes the difficulty bias)",
    )
    ap.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        help="don't penalize completions that hit the length cap (DAPO)",
    )
    ap.add_argument(
        "--epsilon_high",
        type=float,
        default=None,
        help="asymmetric clip upper bound (DAPO recommends 0.28)",
    )
    ap.add_argument(
        "--dynamic_sampling",
        action="store_true",
        help="DAPO dynamic sampling: drop zero-advantage groups so every "
        "step carries signal. Forward-compatible: applied only if the "
        "installed TRL's GRPOConfig supports it (reported at startup).",
    )
    ap.add_argument(
        "--graded_numeric",
        action="store_true",
        help="use partial-credit numeric reward in training to densify the "
        "sparse 0/1 signal (eval stays strict). Reward ablation arm.",
    )
    ap.add_argument(
        "--no_format_reward",
        action="store_true",
        help="train on the correctness reward only (reward ablation)",
    )
    ap.add_argument(
        "--lora_target",
        default="all-linear",
        help="LoRA target modules; 'all-linear' is portable across "
        "model families (avoids per-architecture module names)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn_impl = pick_attn_impl()
    print(f"using attn_implementation={attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    # With LoRA + gradient checkpointing, the checkpointed blocks only build a
    # backward graph if at least one input requires grad. Without this the input
    # embeddings are detached, no gradient reaches the LoRA params, and training
    # silently does nothing. Harmless if checkpointing is off.
    model.enable_input_require_grads()

    dataset = load_dataset("json", data_files=args.data, split="train")

    target = "all-linear" if args.lora_target == "all-linear" else args.lora_target.split(",")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target,
    )

    # The GRPOConfig field names have moved across TRL releases. Keep the desired
    # config here and filter it to what the installed GRPOConfig accepts, warning
    # on anything dropped; the 20-step smoke run confirms the survivors behave.
    # Make rollouts terminate. Qwen2.5-Instruct emits <|im_end|> (151645) but the
    # default vLLM stop can miss it -> completions run to the cap (clipped_ratio=1.0),
    # which then lets mask_truncated zero the whole batch (grad_norm=0). Eval via HF
    # .generate already stops fine; this fixes the TRAINING/vLLM path. General across
    # families (reads each model's declared EOS ids).
    # Collect EVERY plausible stop id: the model's declared EOS list, the tokenizer
    # EOS, and known chat-turn terminators that some configs omit from eos_token_id
    # (Qwen's <|im_end|>, Llama's <|eot_id|>, ...). Passing all of them to vLLM makes
    # rollouts terminate; the narrower generation_config-only version missed <|im_end|>
    # on Qwen2.5-Instruct, so its rollouts ran to the cap.
    _eos = set()
    _gc = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(_gc, int):
        _eos.add(_gc)
    elif _gc:
        _eos.update(_gc)
    if tok.eos_token_id is not None:
        _eos.add(tok.eos_token_id)
    for _t in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>"):
        _tid = tok.convert_tokens_to_ids(_t)
        if isinstance(_tid, int) and _tid >= 0 and _tid != tok.unk_token_id:
            _eos.add(_tid)
    gen_kwargs = {"stop_token_ids": sorted(_eos)} if _eos else {}
    print(f"vLLM stop_token_ids = {gen_kwargs.get('stop_token_ids')}")

    desired = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        max_steps=args.max_steps,
        temperature=0.9,
        seed=args.seed,
        loss_type=args.loss_type,
        scale_rewards=not args.no_scale_rewards,
        mask_truncated_completions=args.mask_truncated_completions,
        epsilon_high=args.epsilon_high,
        bf16=True,
        gradient_checkpointing=True,
        # DDP (even 1 GPU via torchrun) is incompatible with *reentrant* grad
        # checkpointing -- it double-marks the LoRA params. Non-reentrant is safe.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        use_vllm=not args.no_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
        # TRL 0.19.0's colocate path does generation_kwargs.update(None) by
        # default and crashes; pass an empty dict so the update is a no-op.
        generation_kwargs=gen_kwargs,
        report_to="none",
    )
    if args.dynamic_sampling:
        # Field name only present on TRL builds that ship DAPO dynamic sampling.
        desired["dynamic_sampling"] = True

    accepted = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(set(desired) - accepted)
    if unknown:
        print(
            f"[warn] GRPOConfig in this TRL build does not accept: {unknown} "
            f"-- dropping them (confirm the smoke run still trains as intended)."
        )
    if args.dynamic_sampling:
        print(
            f"[research] dynamic_sampling -> "
            f"{'ENABLED' if 'dynamic_sampling' in accepted else 'NOT SUPPORTED by this TRL build (ignored); bump TRL to use it'}"
        )
    config = GRPOConfig(**{k: v for k, v in desired.items() if k in accepted})

    correctness = correctness_reward_graded if args.graded_numeric else correctness_reward
    reward_funcs = [correctness] if args.no_format_reward else [correctness, format_reward]
    print(
        f"reward_funcs: {[f.__name__ for f in reward_funcs]} | "
        f"loss_type={args.loss_type} scale_rewards={not args.no_scale_rewards} "
        f"mask_truncated={args.mask_truncated_completions} lr_sched={args.lr_scheduler_type} "
        f"graded_numeric={args.graded_numeric} seed={args.seed}"
    )

    # Persist the exact recipe BEFORE training so a crashed run is still
    # self-describing (bookkeeping for the Phase-2 aggregator).
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "recipe.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "reward_funcs": [f.__name__ for f in reward_funcs],
                "grpo_config_set": {k: v for k, v in desired.items() if k in accepted},
                "grpo_config_dropped": unknown,
            },
            indent=2,
            default=str,
        )
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()

    # Fail loud on divergence: a NaN in loss/kl/grad means the policy blew up and
    # the LoRA adapter never learned (stays at init -> silent no-op -> a fake +0.0
    # eval that masquerades as "the recipe had no effect"). Refuse to save garbage.
    _hist = trainer.state.log_history
    nan_steps = [
        h
        for h in _hist
        if any(
            isinstance(h.get(k), float) and math.isnan(h.get(k))
            for k in ("loss", "kl", "grad_norm", "reward")
        )
    ]
    _gn = [
        h["grad_norm"]
        for h in _hist
        if isinstance(h.get("grad_norm"), (int, float)) and not math.isnan(h["grad_norm"])
    ]
    never_moved = bool(_gn) and max(_gn) < 1e-8
    if nan_steps or never_moved:
        print(
            f"[FATAL] no valid update: nan_steps={len(nan_steps)}, "
            f"max_grad_norm={max(_gn) if _gn else 'NA'}. The adapter would be a "
            f"no-op (e.g. the Qwen clipped_ratio=1.0 + mask_truncated failure). "
            f"NOT saving. Check rollout termination + recipe, then re-run."
        )
        sys.exit(1)

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter + tokenizer to {args.out}")


if __name__ == "__main__":
    main()
