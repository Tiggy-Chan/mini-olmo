"""使用中文 chat 模板与 mini-OLMo V1 模型对话。"""

from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from generate import generate, load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a mini-OLMo Chinese V1 checkpoint")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer_zh_v1.json")
    parser.add_argument("--system-prompt", type=str, default="你是一个中文助手，请尽量清晰、简洁、直接地回答问题。")
    parser.add_argument("--user-prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    parts = []
    if system_prompt.strip():
        parts.append(f"系统：{system_prompt.strip()}\n")
    parts.append(f"用户：{user_prompt.strip()}\n助手：")
    return "".join(parts)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mini-olmo] 使用设备: {device}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.ckpt_path, device)
    prompt = build_chat_prompt(args.system_prompt, args.user_prompt)

    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()

    print("[mini-olmo] User:")
    print(args.user_prompt)
    print("\n[mini-olmo] Assistant:")
    print(answer)


if __name__ == "__main__":
    main()
