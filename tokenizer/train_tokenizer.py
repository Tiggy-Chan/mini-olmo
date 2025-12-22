"""在 Wikitext 语料上训练一个 BPE tokenizer，并保存到本目录下。

当前实现基于 `tokenizers` 库，从 `data/raw/wikitext/*.txt` 读取文本，
训练出一个 32k 词表大小的 BPE tokenizer，保存为 `tokenizer.json`。
"""

import os
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def get_project_root() -> str:
    """返回项目根目录（包含 README.md 的那一层）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iter_corpus_files() -> List[str]:
    """返回用于训练 tokenizer 的语料文件列表。"""
    root = get_project_root()
    data_dir = os.path.join(root, "data", "raw", "wikitext")
    files = [
        os.path.join(data_dir, name)
        for name in ("train.txt", "validation.txt", "test.txt")
        if os.path.exists(os.path.join(data_dir, name))
    ]
    if not files:
        raise FileNotFoundError(
            f"未找到任何语料文件，请先运行 scripts/prepare_wikitext.py 生成 data/raw/wikitext/*.txt。"
        )
    return files


def train_bpe_tokenizer(
    files: Iterable[str],
    vocab_size: int = 32_000,
    min_frequency: int = 2,
) -> Tokenizer:
    """在给定语料文件上训练一个 BPE tokenizer。"""
    print("[mini-olmo] 初始化 BPE tokenizer（空词表）…")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    files = list(files)
    print("[mini-olmo] 开始在以下文件上训练 tokenizer：")
    for path in files:
        print(f"  - {path}")

    tokenizer.train(files, trainer)

    print("[mini-olmo] 训练完成，词表大小：", tokenizer.get_vocab_size())
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer) -> str:
    """将训练好的 tokenizer 保存到当前目录下的 tokenizer.json。"""
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
    tokenizer.save(out_path)
    print(f"[mini-olmo] 已保存 tokenizer 到 {out_path}")
    return out_path


def main() -> None:
    corpus_files = iter_corpus_files()
    tokenizer = train_bpe_tokenizer(corpus_files)
    save_tokenizer(tokenizer)


if __name__ == "__main__":
    main()
