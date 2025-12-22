import argparse
import os
from typing import List

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from mini_olmo.models.config import MiniOlmoConfig
from mini_olmo.models.transformer import MiniOlmoModel


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_tokenizer() -> Tokenizer:
    root = get_project_root()
    tok_path = os.path.join(root, "..", "tokenizer", "tokenizer.json")
    tok_path = os.path.abspath(tok_path)
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"未找到 tokenizer 文件: {tok_path}")
    return Tokenizer.from_file(tok_path)


def load_model(ckpt_path: str, device: torch.device) -> MiniOlmoModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config")
    if cfg_dict is not None:
        config = MiniOlmoConfig(**cfg_dict)
    else:
        config = MiniOlmoConfig()
    model = MiniOlmoModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    # logits: (vocab_size,)
    if temperature <= 0:
        # greedy
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        return int(indices[next_idx].item())

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return int(next_id)


def generate(
    model: MiniOlmoModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    enc = tokenizer.encode(prompt)
    input_ids: List[int] = enc.ids

    max_seq_len = model.config.max_seq_len
    if len(input_ids) >= max_seq_len:
        input_ids = input_ids[-(max_seq_len - 1) :]

    for _ in range(max_new_tokens):
        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids_tensor)
            logits = out["logits"]
            next_logits = logits[0, -1, :]

        next_id = sample_next_token(next_logits, temperature, top_k)
        input_ids.append(next_id)

        if len(input_ids) >= max_seq_len:
            break

    text = tokenizer.decode(input_ids)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from mini-OLMo checkpoint")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, this is mini-OLMo.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mini-olmo] 使用设备: {device}")

    tokenizer = load_tokenizer()
    model = load_model(args.ckpt_path, device)

    print("[mini-olmo] Prompt:")
    print(args.prompt)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("\n[mini-olmo] Generation:")
    print(text)


if __name__ == "__main__":
    main()
