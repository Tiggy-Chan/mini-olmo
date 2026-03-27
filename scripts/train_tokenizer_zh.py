"""训练第一版中文 tokenizer。

默认从 `data/raw/zh_corpus_v1/{train,validation,test}.txt` 读取数据，
输出到 `tokenizer/tokenizer_zh_v1.json`。

说明：
- `zh_corpus_v1` 保持为 tokenizer / smoke 用的小规模默认语料
- 主训练语料默认已经切到更大的 `zh_corpus_v3_elite_20gb`
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese-first tokenizer for mini-OLMo")
    parser.add_argument("--corpus-name", type=str, default="zh_corpus_v1")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-path", type=str, default="tokenizer/tokenizer_zh_v1.json")
    return parser.parse_args()


def iter_corpus_files(corpus_name: str) -> List[str]:
    root = get_project_root()
    data_dir = os.path.join(root, "data", "raw", corpus_name)
    files = [
        os.path.join(data_dir, name)
        for name in ("train.txt", "validation.txt", "test.txt")
        if os.path.exists(os.path.join(data_dir, name))
    ]
    if not files:
        raise FileNotFoundError(
            f"未找到语料文件，请先运行 scripts/prepare_corpus_zh_v1.py 生成 {data_dir}/*.txt。"
        )
    return files


def train_bpe_tokenizer(
    files: Iterable[str],
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    print("[mini-olmo] 初始化中文 BPE tokenizer …")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    files = list(files)
    print("[mini-olmo] 在以下文件上训练中文 tokenizer：")
    for path in files:
        print(f"  - {path}")

    tokenizer.train(files, trainer)
    print(f"[mini-olmo] 训练完成，词表大小: {tokenizer.get_vocab_size()}")
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_path: str) -> str:
    root = get_project_root()
    abs_path = os.path.join(root, output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tokenizer.save(abs_path)
    print(f"[mini-olmo] 已保存 tokenizer 到 {abs_path}")
    return abs_path


def preview_tokenization(tokenizer: Tokenizer) -> None:
    samples = [
        "你好，请介绍一下你自己。",
        "什么是机器学习？请用简单中文解释。",
        "请总结一下这段文字的核心观点，并给出三个要点。",
        "RTX 4060 Laptop 可以训练多大的中文模型？",
    ]
    print("[mini-olmo] tokenizer 编码预览：")
    for text in samples:
        enc = tokenizer.encode(text)
        print(f"  TEXT: {text}")
        print(f"  TOKENS: {enc.tokens[:24]}")
        print(f"  LENGTH: {len(enc.ids)}")


def main() -> None:
    args = parse_args()
    files = iter_corpus_files(args.corpus_name)
    tokenizer = train_bpe_tokenizer(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    save_tokenizer(tokenizer, args.output_path)
    preview_tokenization(tokenizer)


if __name__ == "__main__":
    main()
