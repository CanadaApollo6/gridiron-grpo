"""GRPO training entrypoint (TRL + PEFT/LoRA + vLLM rollouts).

Runs on a single H100 80GB for a ~1.5-3B model with LoRA. This file is written
to spec against TRL's GRPOTrainer; the GRPO API has moved across releases, so:
  1. install the pinned versions in requirements.txt,
  2. do a 20-step smoke run first (--max_steps 20) to confirm arg names,
  3. then launch the real run.

Example:
  python src/train_grpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data_out/train.jsonl \
    --out runs/grpo-qwen15b \
    --max_steps 1200
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import pick_attn_impl  # noqa: E402
from rewards.verifiers import correctness_reward, format_reward  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data_out/train.jsonl")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--num_generations", type=int, default=8,
                    help="rollouts per prompt (the 'group' in GRPO)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--max_prompt_len", type=int, default=640)
    ap.add_argument("--max_completion_len", type=int, default=512)
    ap.add_argument("--per_device_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--no_vllm", action="store_true")
    ap.add_argument("--vllm_mode", default="colocate",
                    help="colocate = vLLM shares the training GPU (single-GPU runs); "
                         "server = expects a separate `trl vllm-serve` process")
    ap.add_argument("--vllm_gpu_mem_util", type=float, default=0.30,
                    help="fraction of GPU memory vLLM reserves for KV cache in "
                         "colocate mode. Lower it if training OOMs; raise it if "
                         "rollouts are KV-starved. On a busy/small GPU this is "
                         "often what needs tuning.")
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
    # backward graph if at least one of their inputs requires grad. Without this,
    # the input embeddings are detached, no gradient reaches the LoRA params, and
    # training silently does nothing (you'd see "None of the inputs have
    # requires_grad=True. Gradients will be None"). This registers the hook that
    # makes the embedding output require grad. Harmless if checkpointing is off.
    model.enable_input_require_grads()

    # Dataset already has a chat-format `prompt` column + the reward columns
    # (ground_truth, answer_type) that TRL forwards to the reward functions.
    dataset = load_dataset("json", data_files=args.data, split="train")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # The GRPOConfig field names have moved across TRL releases (this is the #1
    # gotcha called out in the README). Rather than pass a fixed kwargs blob that
    # might crash on a renamed/removed field, we keep the *desired* config here and
    # filter it down to what the installed GRPOConfig actually accepts, warning on
    # anything dropped. The 20-step smoke run then confirms the survivors behave.
    desired = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        max_steps=args.max_steps,
        temperature=0.9,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        use_vllm=not args.no_vllm,           # vLLM-accelerated rollouts
        vllm_mode=args.vllm_mode,            # colocate: vLLM shares the single GPU
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
        # TRL 0.19.0's colocate path does `generation_kwargs.update(self.args.
        # generation_kwargs)`, which crashes when this defaults to None. Pass an
        # empty dict so the update is a harmless no-op.
        generation_kwargs={},
        report_to="none",                    # swap to "wandb" to log curves
    )
    import dataclasses
    accepted = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(set(desired) - accepted)
    if unknown:
        print(f"[warn] GRPOConfig in this TRL build does not accept: {unknown} "
              f"-- dropping them (check the smoke run still trains as intended).")
    config = GRPOConfig(**{k: v for k, v in desired.items() if k in accepted})

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter + tokenizer to {args.out}")


if __name__ == "__main__":
    main()
