"""Agent layer: the tuned model as a callable tool.

`ReasoningTool` wraps the GRPO-tuned model as one well-defined function,
`reason_over_structured_data(context, question) -> answer`. It is real and
runnable on its own -- see the `__main__` demo at the bottom (load the model,
ask a box-score question, get a parsed answer back).

The tool is framework-agnostic. The block at the end of this file shows the
*pattern* for registering it inside NVIDIA's NeMo Agent Toolkit so an agent can
call it. Keeping the claim honest: what actually runs here is the ReasoningTool;
the NeMo Agent Toolkit registration is a documented integration sketch, not a
shipped, wired-up integration -- pinning it to an installed NeMo Agent Toolkit
version and registering the function is the remaining step.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prompts import build_prompt, extract_answer  # noqa: E402


class ReasoningTool:
    """Loads the GRPO-tuned model and answers structured-data questions."""

    def __init__(self, base_model: str, adapter: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()

    @torch.no_grad()
    def run(self, context: str, question: str, max_new_tokens: int = 512) -> str:
        msgs = build_prompt(context, question)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id,
        )
        out = self.tok.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return extract_answer(out) or out


# --- NeMo Agent Toolkit integration (pattern; not wired in this repo) ------
# Register ReasoningTool.run as a tool/function in NeMo Agent Toolkit so an
# agent can invoke it. Pin to the NeMo Agent Toolkit version you install and
# follow its current tool-registration API (function decorator + YAML workflow
# config). Pseudocode shape:
#
#   from nat.builder.function_info import FunctionInfo   # name per installed version
#
#   tool = ReasoningTool(base_model="Qwen/Qwen2.5-1.5B-Instruct",
#                        adapter="runs/grpo-qwen15b")
#
#   def football_data_reasoner(context: str, question: str) -> str:
#       return tool.run(context, question)
#
#   # ...register football_data_reasoner with the toolkit and reference it from
#   # the agent's workflow YAML. See NVIDIA NeMo Agent Toolkit docs.

if __name__ == "__main__":
    tool = ReasoningTool("Qwen/Qwen2.5-1.5B-Instruct", adapter=None)
    ctx = (
        "Player | RushAtt RushYds RushTD | Rec RecYds RecTD\n"
        "E. Thomas | 22 119 1 | 2 29 0\nC. Jackson | 10 43 2 | 12 91 0"
    )
    print(tool.run(ctx, "Which player had the most total yards from scrimmage?"))
