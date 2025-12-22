import os
from typing import Dict

from datasets import load_dataset


def get_project_root() -> str:
    """返回项目根目录（包含 README.md 的那一层）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_wikitext(dataset_name: str = "wikitext", config: str = "wikitext-103-raw-v1") -> Dict[str, str]:
    """下载 Wikitext 数据集，并将 train/validation/test 各自保存为一个纯文本文件。

    返回一个 dict，键是 split 名称，值是对应保存的文件路径。
    """
    root = get_project_root()
    out_dir = os.path.join(root, "data", "raw", "wikitext")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[mini-olmo] 下载数据集 {dataset_name}/{config} …")
    dataset = load_dataset(dataset_name, config)

    paths: Dict[str, str] = {}
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        split_ds = dataset[split]
        texts = split_ds["text"]
        # Wikitext 自身已经是按段落分好的文本，这里简单按两个换行拼接
        merged = "\n\n".join(texts)
        out_path = os.path.join(out_dir, f"{split}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(merged)
        paths[split] = out_path
        print(f"[mini-olmo] 已保存 {split} 到 {out_path}，共 {len(texts)} 条。")

    return paths


def main() -> None:
    paths = prepare_wikitext()
    print("[mini-olmo] 完成 Wikitext 数据准备：")
    for split, path in paths.items():
        print(f"  - {split}: {path}")


if __name__ == "__main__":
    main()
