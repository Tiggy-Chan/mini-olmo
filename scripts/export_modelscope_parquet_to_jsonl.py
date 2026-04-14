"""把本地 ModelScope parquet 语料导出成当前训练流水线可读的 jsonl。

输出格式兼容 `scripts/prepare_corpus_zh_v1.py --extra-text-dir`：
- 每行一个 JSON 对象
- 至少包含 `text` 字段

示例：

```bash
conda run -n mini-olmo python scripts/export_modelscope_parquet_to_jsonl.py \
  --input-dir data/downloads/zh_corpora/chinese_cosmopedia/data \
  --output-dir data/local/chinese_cosmopedia_jsonl \
  --max-files 17
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from datasets import load_dataset


PREFERRED_TEXT_KEYS = (
    "text",
    "content",
    "markdown",
    "body",
    "article",
    "document",
    "passage",
    "response",
    "output",
    "answer",
)

SKIP_FALLBACK_KEYS = {
    "id",
    "uuid",
    "url",
    "domain",
    "lang",
    "language",
    "source",
    "category",
    "topic",
    "title",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local ModelScope parquet shards to jsonl files compatible with --extra-text-dir"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing local parquet files, e.g. data/downloads/zh_corpora/chinese_cosmopedia/data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write exported jsonl files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.parquet",
        help="Glob pattern for parquet files. Default: *.parquet",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only export the first N parquet files after sorting. 0 means all.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional row limit per parquet file. 0 means no limit.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help="Skip exported samples shorter than this many characters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing jsonl files instead of skipping them.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10_000,
        help="Print progress every N exported samples.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\u3000", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_text_from_row(row: Dict[str, object]) -> str | None:
    for key in PREFERRED_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for key in ("messages", "conversations"):
        value = row.get(key)
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                maybe_text = item.get("content") or item.get("value")
                if isinstance(maybe_text, str) and maybe_text.strip():
                    parts.append(maybe_text)
            if parts:
                return "\n".join(parts)

    candidate_strings: List[str] = []
    for key, value in row.items():
        if key in SKIP_FALLBACK_KEYS:
            continue
        if isinstance(value, str) and value.strip():
            candidate_strings.append(value)
    if candidate_strings:
        return max(candidate_strings, key=len)
    return None


def iter_parquet_rows(parquet_path: Path) -> Iterator[Dict[str, object]]:
    data_file = str(parquet_path)
    try:
        dataset = load_dataset("parquet", data_files=data_file, split="train", streaming=True)
        for row in dataset:
            if isinstance(row, dict):
                yield row
        return
    except Exception:
        dataset = load_dataset("parquet", data_files=data_file, split="train")
        for row in dataset:
            if isinstance(row, dict):
                yield row


def discover_parquet_files(input_dir: Path, pattern: str, max_files: int) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    if max_files > 0:
        files = files[:max_files]
    return [path for path in files if path.is_file()]


def export_parquet_file(
    parquet_path: Path,
    output_path: Path,
    *,
    min_chars: int,
    max_rows_per_file: int,
    log_interval: int,
) -> int:
    exported = 0
    seen = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in iter_parquet_rows(parquet_path):
            seen += 1
            text = extract_text_from_row(row)
            if not isinstance(text, str):
                continue
            text = normalize_text(text)
            if len(text) < min_chars:
                continue

            payload = {
                "text": text,
                "source": f"modelscope::{parquet_path.name}",
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            exported += 1

            if log_interval > 0 and exported % log_interval == 0:
                print(
                    f"[mini-olmo] {parquet_path.name}: 已导出 {exported} 条有效样本 "
                    f"(已扫描 {seen} 行原始记录)"
                )

            if max_rows_per_file > 0 and exported >= max_rows_per_file:
                break

    return exported


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"未找到 parquet 目录: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = discover_parquet_files(input_dir, args.glob, args.max_files)
    if not parquet_files:
        raise FileNotFoundError(f"在 {input_dir} 下未找到匹配 {args.glob!r} 的 parquet 文件。")

    total_exported = 0
    converted_files = 0
    skipped_files = 0

    print(f"[mini-olmo] 发现 {len(parquet_files)} 个 parquet 文件待导出。")
    for parquet_path in parquet_files:
        output_path = output_dir / f"{parquet_path.stem}.jsonl"
        if output_path.exists() and not args.overwrite:
            skipped_files += 1
            print(f"[mini-olmo] 已存在，跳过: {output_path}")
            continue

        print(f"[mini-olmo] 开始导出: {parquet_path} -> {output_path}")
        exported = export_parquet_file(
            parquet_path,
            output_path,
            min_chars=args.min_chars,
            max_rows_per_file=args.max_rows_per_file,
            log_interval=args.log_interval,
        )
        converted_files += 1
        total_exported += exported
        print(f"[mini-olmo] 完成 {parquet_path.name}: 导出 {exported} 条样本")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "glob": args.glob,
        "max_files": args.max_files,
        "converted_files": converted_files,
        "skipped_files": skipped_files,
        "total_exported_samples": total_exported,
        "output_files": [str(output_dir / f"{path.stem}.jsonl") for path in parquet_files],
    }
    summary_path = output_dir / "export_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[mini-olmo] parquet -> jsonl 导出完成：")
    print(f"  - converted_files: {converted_files}")
    print(f"  - skipped_files: {skipped_files}")
    print(f"  - total_exported_samples: {total_exported}")
    print(f"  - summary: {summary_path}")
    print(
        "[mini-olmo] 下一步可把输出目录直接传给 "
        "`scripts/prepare_corpus_zh_v1.py --extra-text-dir`。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
