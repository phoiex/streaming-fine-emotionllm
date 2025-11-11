
#!/usr/bin/env python3
"""
Quick testrun for a local Llama-2-7b-chat-hf checkout.

Examples (from repo root):
  python code/scripts/testrun.py --prompt "你好！用一句话介绍你自己。"
  python code/scripts/testrun.py --device-map cpu --dtype float32 --max-new-tokens 64 \
         --prompt "Explain Llama 2 in one sentence."
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-2-7b-chat-hf quick testrun")
    # Default to repo-root/resources/models/Llama-2-7b-chat-hf
    root = Path(__file__).resolve().parents[2]
    default_repo = str(root / "resources/models/Llama-2-7b-chat-hf")
    p.add_argument("--repo", default=default_repo, help="Path to local model repository")
    p.add_argument(
        "--prompt",
        default="你能理解中文吗，你能输出中文吗，输出一个中文句子试试",
        help="User message to test",
    )
    p.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="Optional system message for chat template",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024, dest="max_new_tokens")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, dest="top_p", default=None)
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for weights",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help="Device map (e.g., auto, cpu, cuda)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-chat-template",
        dest="use_chat_template",
        action="store_false",
        help="Use raw prompt instead of chat template",
    )
    p.set_defaults(use_chat_template=True)
    return p.parse_args()


def to_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"[testrun] Loading tokenizer and model from: {args.repo}")
    dtype = to_dtype(args.dtype)
    try:
        tok = AutoTokenizer.from_pretrained(args.repo)
    except Exception as e:
        print(f"[error] Failed to load tokenizer: {e}")
        return 2

    try:
        # Prefer new 'dtype' kwarg; fall back to 'torch_dtype' for older transformers
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.repo, dtype=dtype, device_map=args.device_map
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.repo, torch_dtype=dtype, device_map=args.device_map
            )
    except Exception as e:
        print(
            "[error] Failed to load model. Consider using --device-map cpu and/or --dtype float32.\n"
            f"Details: {e}"
        )
        return 3

    # Ensure pad token is defined to avoid warnings/errors during generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Build inputs
    device = next(model.parameters()).device
    if args.use_chat_template:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt.strip()},
        ]
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        generate_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        enc = tok(args.prompt.strip(), return_tensors="pt")
        generate_inputs = {k: v.to(device) for k, v in enc.items()}  # includes attention_mask

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    # Respect optional overrides; otherwise rely on generation_config.json
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    gen_kwargs.setdefault("pad_token_id", tok.pad_token_id)
    gen_kwargs.setdefault("eos_token_id", tok.eos_token_id)

    print(
        f"[testrun] Generating (max_new_tokens={gen_kwargs['max_new_tokens']}, "
        f"temperature={gen_kwargs.get('temperature','default')}, top_p={gen_kwargs.get('top_p','default')})..."
    )
    with torch.no_grad():
        out = model.generate(**generate_inputs, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n===== Generated Text =====\n")
    print(text)
    print("\n===== End =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
