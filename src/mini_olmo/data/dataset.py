import os
from typing import Dict, Iterable, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from mini_olmo.models.config import MiniOlmoConfig

Split = Literal["train", "validation", "test"]


def get_project_root() -> str:
    """返回项目根目录（包含 README.md 的那一层）。"""
    # 当前文件位于: <project_root>/src/mini_olmo/data/dataset.py
    # 逐级回溯 4 次可到达项目根目录
    path = os.path.abspath(__file__)
    for _ in range(4):
        path = os.path.dirname(path)
    return path


def _get_tokenizer() -> Tokenizer:
    """加载我们之前训练好的 tokenizer。"""
    root = get_project_root()
    tok_path = os.path.join(root, "tokenizer", "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"未找到 tokenizer 文件: {tok_path}，请先运行 tokenizer/train_tokenizer.py。"
        )
    return Tokenizer.from_file(tok_path)


def _get_cache_path(split: Split, corpus_name: str) -> str:
    root = get_project_root()
    cache_dir = os.path.join(root, "data", "tokenized")
    os.makedirs(cache_dir, exist_ok=True)
    if corpus_name == "wikitext":
        return os.path.join(cache_dir, f"wikitext_{split}_ids.pt")
    corpus_dir = os.path.join(cache_dir, corpus_name)
    os.makedirs(corpus_dir, exist_ok=True)
    return os.path.join(corpus_dir, f"{corpus_name}_{split}_ids.pt")


def _get_raw_text_path(split: Split, corpus_name: str) -> str:
    root = get_project_root()
    if corpus_name == "wikitext":
        return os.path.join(root, "data", "raw", "wikitext", f"{split}.txt")
    return os.path.join(root, "data", "raw", corpus_name, f"{split}.txt")


def build_or_load_token_ids(split: Split, corpus_name: str = "wikitext") -> torch.Tensor:
    """为给定 split 构建或加载一维 token id 序列。

    - 优先从 data/tokenized/wikitext_{split}_ids.pt 加载
    - 如果不存在，则从 data/raw/wikitext/{split}.txt 读取文本，
      使用 tokenizer 编码，并缓存到上述路径。
    """
    cache_path = _get_cache_path(split, corpus_name)
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    text_path = _get_raw_text_path(split, corpus_name)
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"未找到原始文本: {text_path}。")

    tokenizer = _get_tokenizer()
    eos_id = tokenizer.token_to_id("<eos>")

    all_ids: Iterable[int] = []
    ids_list = []

    print(f"[mini-olmo] 正在为 {split} 编码文本（这一步只需执行一次）…")
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer.encode(line)
            ids = enc.ids
            if eos_id is not None:
                ids.append(eos_id)
            ids_list.extend(ids)

    token_ids = torch.tensor(ids_list, dtype=torch.long)
    torch.save(token_ids, cache_path)
    print(f"[mini-olmo] 已将 {split} token ids 保存到 {cache_path}，共 {token_ids.numel()} 个 token。")
    return token_ids


class LMDataset(Dataset):
    """将长 token 流切分为固定长度序列，以用于自回归语言建模。

    每个样本包含：
      - input_ids: 长度为 seq_len
      - labels:    长度为 seq_len，对应 input_ids 的下一个 token
    """

    def __init__(
        self,
        split: Split,
        config: MiniOlmoConfig,
        seq_len: int | None = None,
        corpus_name: str = "wikitext",
    ) -> None:
        super().__init__()
        self.split = split
        self.config = config
        self.seq_len = seq_len or config.max_seq_len
        self.corpus_name = corpus_name

        if self.seq_len > config.max_seq_len:
            raise ValueError(f"seq_len={self.seq_len} 不应超过 config.max_seq_len={config.max_seq_len}")

        token_ids = build_or_load_token_ids(split, corpus_name=corpus_name)
        # 为了构造 (input, label) 对，每个样本需要 seq_len+1 个 token
        num_tokens = token_ids.numel()
        self.num_sequences = (num_tokens - 1) // self.seq_len
        usable_tokens = self.num_sequences * self.seq_len + 1
        self.token_ids = token_ids[:usable_tokens]

        print(
            f"[mini-olmo] 构建 LMDataset(split={split}, corpus={corpus_name})：seq_len={self.seq_len}, "
            f"num_tokens={num_tokens}, num_sequences={self.num_sequences}"
        )

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        seq = self.token_ids[start:end]
        input_ids = seq[:-1]
        labels = seq[1:]
        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    split: Split,
    config: MiniOlmoConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    corpus_name: str = "wikitext",
) -> DataLoader:
    """创建用于训练/验证的 DataLoader。"""
    dataset = LMDataset(split, config, corpus_name=corpus_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
