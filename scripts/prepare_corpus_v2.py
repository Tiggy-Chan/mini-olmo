import os
from typing import Dict, Iterable, List

from datasets import load_dataset, Dataset


def get_project_root() -> str:
    """返回项目根目录（包含 README.md 的那一层）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dataset_to_text(ds: Dataset, text_key: str = "text") -> List[str]:
    return [str(t) for t in ds[text_key] if isinstance(t, str) and t.strip()]


def prepare_corpus_v2() -> Dict[str, str]:
    """构建一个比 Wikitext 更大的混合语料 corpus_v2。

    目前简单组合：
      - 现有的 Wikitext train/validation/test 文本
      - 英文 Wikipedia 的一个子集（例如 1%）
      - BookCorpus Open 的一个子集

    最终写入 data/raw/corpus_v2/{train,validation,test}.txt
    """

    root = get_project_root()
    raw_dir = os.path.join(root, "data", "raw")
    wikitext_dir = os.path.join(raw_dir, "wikitext")
    out_dir = os.path.join(raw_dir, "corpus_v2")
    _ensure_dir(out_dir)

    paths: Dict[str, str] = {}

    # 1) 读取已有 Wikitext 文本
    wiki_texts: Dict[str, List[str]] = {}
    for split in ("train", "validation", "test"):
        wt_path = os.path.join(wikitext_dir, f"{split}.txt")
        if os.path.exists(wt_path):
            with open(wt_path, "r", encoding="utf-8") as f:
                wiki_texts[split] = [line.rstrip("\n") for line in f if line.strip()]
        else:
            wiki_texts[split] = []

    # 2) 加载部分 Wikipedia 英文语料（子集，避免过大）
    wiki_text_list: List[str] = []
    try:
        print("[mini-olmo] 下载 Wikipedia 子集 …")
        try:
            wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en")
        except Exception:
            wiki_ds = load_dataset("wikipedia", "20220301.en")
        wiki_train = wiki_ds["train"].shuffle(seed=42).select(range(int(len(wiki_ds["train"]) * 0.02)))
        wiki_text_list = _dataset_to_text(wiki_train, text_key="text")
    except Exception as e:  # noqa: BLE001
        print(f"[mini-olmo] 加载 Wikipedia 失败，将跳过该语料，仅使用 Wikitext + BookCorpusOpen。错误: {e}")

    # 3) 加载 BookCorpus Open 子集
    book_text_list: List[str] = []
    try:
        print("[mini-olmo] 下载 BookCorpusOpen 子集 …")
        book_ds = load_dataset("bookcorpusopen")
        book_train = book_ds["train"].shuffle(seed=42).select(range(min(200000, len(book_ds["train"]))))
        book_text_list = _dataset_to_text(book_train, text_key="text")
    except Exception as e:  # noqa: BLE001
        print(f"[mini-olmo] 加载 BookCorpusOpen 失败，将只使用 Wikitext 和（若可用）Wikipedia。错误: {e}")

    # 简单划分：
    #   - 绝大部分放入 train
    #   - 少量划给 validation/test
    def split_list(data: List[str], val_ratio: float = 0.01, test_ratio: float = 0.01):
        n = len(data)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test
        return data[:n_train], data[n_train : n_train + n_val], data[n_train + n_val :]

    wiki_train_l, wiki_val_l, wiki_test_l = split_list(wiki_text_list)
    book_train_l, book_val_l, book_test_l = split_list(book_text_list)

    # 4) 按 split 合并 Wikitext + Wikipedia + BookCorpusOpen
    for split in ("train", "validation", "test"):
        merged: List[str] = []

        merged.extend(wiki_texts.get(split, []))

        if split == "train":
            merged.extend(wiki_train_l)
            merged.extend(book_train_l)
        elif split == "validation":
            merged.extend(wiki_val_l)
            merged.extend(book_val_l)
        else:
            merged.extend(wiki_test_l)
            merged.extend(book_test_l)

        # 简单打乱
        import random

        random.seed(42)
        random.shuffle(merged)

        out_path = os.path.join(out_dir, f"{split}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(merged))

        paths[split] = out_path
        print(f"[mini-olmo] corpus_v2 {split} 已保存到 {out_path}，共 {len(merged)} 段文本。")

    return paths


def main() -> None:
    paths = prepare_corpus_v2()
    print("[mini-olmo] 完成 corpus_v2 数据准备：")
    for split, path in paths.items():
        print(f"  - {split}: {path}")


if __name__ == "__main__":
    main()
