"""构建中文优先预训练语料。

当前主线默认口径是：
- 中文 Wikipedia（通过 Hugging Face datasets）
- 中文 FineWeb Edu 高质量分桶抽样

脚本仍然支持额外来源：
- 中文 Cosmopedia（合成教材/知识文本，可选）
- 用户自行准备的本地中文文本目录

输出格式与现有数据管线兼容：
- data/raw/<corpus_name>/train.txt
- data/raw/<corpus_name>/validation.txt
- data/raw/<corpus_name>/test.txt

每行一条样本，便于直接复用当前 tokenizer / dataset 流程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Iterator, List, Sequence

from datasets import load_dataset


@dataclass(frozen=True)
class TextSample:
    text: str
    source: str


class DigestStore:
    def __init__(self, backend: str, db_path: str | None = None) -> None:
        self.backend = backend
        self._memory_store: set[str] | None = None
        self._conn: sqlite3.Connection | None = None
        self._pending = 0

        if backend == "memory":
            self._memory_store = set()
            return
        if backend != "sqlite":
            raise ValueError(f"不支持的去重后端: {backend}")

        if not db_path:
            raise ValueError("sqlite 去重后端需要提供数据库路径。")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS digests (digest TEXT PRIMARY KEY)")

    def add_if_new(self, digest: str) -> bool:
        if self.backend == "memory":
            assert self._memory_store is not None
            if digest in self._memory_store:
                return False
            self._memory_store.add(digest)
            return True

        assert self._conn is not None
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO digests(digest) VALUES (?)",
            (digest,),
        )
        self._pending += 1
        if self._pending >= 5000:
            self._conn.commit()
            self._pending = 0
        return cursor.rowcount == 1

    def close(self) -> None:
        if self._conn is not None:
            if self._pending > 0:
                self._conn.commit()
                self._pending = 0
            self._conn.close()


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the first Chinese-heavy pretraining corpus")
    parser.add_argument("--corpus-name", type=str, default="zh_corpus_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.60)
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=0.0,
        help="Stop once the written corpus reaches this many GiB of UTF-8 text. 0 means no byte target.",
    )
    parser.add_argument(
        "--dedupe-backend",
        type=str,
        default="auto",
        choices=("auto", "memory", "sqlite"),
        help="Use sqlite for large corpus builds to avoid keeping all digests in RAM.",
    )
    parser.add_argument(
        "--dedupe-db-path",
        type=str,
        default="",
        help="Optional sqlite path for digest dedupe. Defaults to data/raw/<corpus_name>/dedupe.sqlite3 when needed.",
    )
    parser.add_argument("--skip-wikipedia", action="store_true")
    parser.add_argument("--max-wikipedia-docs", type=int, default=300_000)
    parser.add_argument("--include-cosmopedia", action="store_true")
    parser.add_argument("--max-cosmopedia-docs", type=int, default=150_000)
    parser.add_argument("--include-fineweb", action="store_true")
    parser.add_argument("--max-fineweb-docs", type=int, default=100_000)
    parser.add_argument(
        "--fineweb-data-dir",
        action="append",
        default=[],
        help="FineWeb Edu Chinese score bucket directory, e.g. 4_5, 3_4, 2_3. Can be passed multiple times.",
    )
    parser.add_argument("--streaming-buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--extra-text-dir",
        action="append",
        default=[],
        help="Optional local directory containing .txt/.md/.jsonl Chinese text files. Can be passed multiple times.",
    )
    parser.add_argument("--log-interval", type=int, default=10_000)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\u3000", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
    )


def cjk_ratio(text: str) -> float:
    cjk_count = 0
    latin_count = 0
    for ch in text:
        if is_cjk_char(ch):
            cjk_count += 1
        elif ("a" <= ch.lower() <= "z"):
            latin_count += 1
    denom = cjk_count + latin_count
    if denom == 0:
        return 0.0
    return cjk_count / denom


def accept_text(text: str, min_chars: int, max_chars: int, min_ratio: float) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False
    ratio = cjk_ratio(text)
    return ratio >= min_ratio


def hash_bucket(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def choose_split(text: str, validation_ratio: float, test_ratio: float) -> str:
    bucket = hash_bucket(text)
    test_threshold = int(test_ratio * 10_000)
    val_threshold = int((test_ratio + validation_ratio) * 10_000)
    if bucket < test_threshold:
        return "test"
    if bucket < val_threshold:
        return "validation"
    return "train"


def extract_text_from_row(row: Dict[str, object]) -> str | None:
    for key in ("text", "content", "markdown", "body", "article", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for key in ("messages", "conversations"):
        value = row.get(key)
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    maybe_text = item.get("content") or item.get("value")
                    if isinstance(maybe_text, str) and maybe_text.strip():
                        parts.append(maybe_text)
            if parts:
                return "\n".join(parts)
    return None


def iter_wikipedia_samples(max_docs: int, seed: int, buffer_size: int) -> Iterator[TextSample]:
    print("[mini-olmo] 加载中文 Wikipedia …")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.zh",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=buffer_size)

    count = 0
    for row in dataset:
        text = extract_text_from_row(row)
        if isinstance(text, str):
            yield TextSample(text=text, source="zh_wikipedia")
            count += 1
            if count >= max_docs:
                break


def iter_cosmopedia_samples(max_docs: int, seed: int, buffer_size: int) -> Iterator[TextSample]:
    print("[mini-olmo] 加载中文 Cosmopedia 抽样 …")
    dataset = load_dataset(
        "opencsg/chinese-cosmopedia",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=buffer_size)

    count = 0
    for row in dataset:
        text = extract_text_from_row(row)
        if isinstance(text, str):
            yield TextSample(text=text, source="chinese_cosmopedia")
            count += 1
            if count >= max_docs:
                break


def iter_fineweb_samples(max_docs: int, seed: int, data_dir: str, buffer_size: int) -> Iterator[TextSample]:
    print(f"[mini-olmo] 加载中文 FineWeb Edu 抽样（data_dir={data_dir}）…")
    try:
        dataset = load_dataset(
            "opencsg/Fineweb-Edu-Chinese-V2.1",
            data_dir=data_dir,
            split="train",
            streaming=True,
        )
    except Exception:
        dataset = load_dataset(
            "opencsg/chinese-fineweb-edu-v2",
            split="train",
            streaming=True,
        )

    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    count = 0
    for row in dataset:
        text = extract_text_from_row(row)
        if isinstance(text, str):
            source_name = f"fineweb_edu_zh::{data_dir}"
            yield TextSample(text=text, source=source_name)
            count += 1
            if count >= max_docs:
                break


def iter_local_text_samples(directories: Sequence[str]) -> Iterator[TextSample]:
    for directory in directories:
        abs_dir = os.path.abspath(directory)
        if not os.path.isdir(abs_dir):
            raise FileNotFoundError(f"未找到目录: {abs_dir}")

        print(f"[mini-olmo] 读取本地目录语料: {abs_dir}")
        for root, _, files in os.walk(abs_dir):
            for name in sorted(files):
                path = os.path.join(root, name)
                ext = os.path.splitext(name)[1].lower()
                if ext in {".txt", ".md"}:
                    yield from iter_text_file(path)
                elif ext == ".jsonl":
                    yield from iter_jsonl_file(path)


def iter_text_file(path: str) -> Iterator[TextSample]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = re.split(r"\n\s*\n+", content)
    source = f"local::{os.path.basename(path)}"
    for chunk in chunks:
        if chunk.strip():
            yield TextSample(text=chunk, source=source)


def iter_jsonl_file(path: str) -> Iterator[TextSample]:
    source = f"local::{os.path.basename(path)}"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record.get("text"), str):
                yield TextSample(text=record["text"], source=source)
                continue
            messages = record.get("messages")
            if isinstance(messages, list):
                parts: List[str] = []
                for item in messages:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    content = item.get("content")
                    if isinstance(role, str) and isinstance(content, str):
                        parts.append(f"{role}: {content}")
                if parts:
                    yield TextSample(text="\n".join(parts), source=source)


def build_sources(args: argparse.Namespace) -> List[Iterable[TextSample]]:
    sources: List[Iterable[TextSample]] = []
    if not args.skip_wikipedia:
        sources.append(
            iter_wikipedia_samples(
                args.max_wikipedia_docs,
                args.seed,
                args.streaming_buffer_size,
            )
        )
    if args.include_cosmopedia:
        sources.append(
            iter_cosmopedia_samples(
                args.max_cosmopedia_docs,
                args.seed,
                args.streaming_buffer_size,
            )
        )
    if args.include_fineweb:
        fineweb_data_dirs = args.fineweb_data_dir or ["4_5"]
        for data_dir in fineweb_data_dirs:
            sources.append(
                iter_fineweb_samples(
                    args.max_fineweb_docs,
                    args.seed,
                    data_dir,
                    args.streaming_buffer_size,
                )
            )
    if args.extra_text_dir:
        sources.append(iter_local_text_samples(args.extra_text_dir))
    if not sources:
        raise ValueError("没有启用任何语料源。请至少启用 Wikipedia、FineWeb、Cosmopedia 或本地文本目录。")
    return sources


def ensure_output_dir(corpus_name: str) -> str:
    root = get_project_root()
    output_dir = os.path.join(root, "data", "raw", corpus_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def resolve_dedupe_backend(args: argparse.Namespace) -> str:
    if args.dedupe_backend != "auto":
        return args.dedupe_backend
    if args.target_size_gb >= 5.0:
        return "sqlite"
    return "memory"


def resolve_dedupe_db_path(args: argparse.Namespace, output_dir: str) -> str | None:
    backend = resolve_dedupe_backend(args)
    if backend != "sqlite":
        return None
    if args.dedupe_db_path:
        return os.path.abspath(args.dedupe_db_path)
    return os.path.join(output_dir, "dedupe.sqlite3")


def write_corpus(args: argparse.Namespace) -> Dict[str, int]:
    if args.validation_ratio + args.test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio 必须小于 1.0")

    output_dir = ensure_output_dir(args.corpus_name)
    dedupe_backend = resolve_dedupe_backend(args)
    dedupe_db_path = resolve_dedupe_db_path(args, output_dir)
    target_bytes = int(args.target_size_gb * (1024 ** 3)) if args.target_size_gb > 0 else 0
    output_paths = {
        split: os.path.join(output_dir, f"{split}.txt")
        for split in ("train", "validation", "test")
    }

    writers = {
        split: open(path, "w", encoding="utf-8")
        for split, path in output_paths.items()
    }

    digest_store = DigestStore(dedupe_backend, dedupe_db_path)
    split_counts = Counter()
    split_bytes = Counter()
    source_counts = Counter()
    source_split_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    total_seen = 0
    total_kept = 0
    total_written_bytes = 0
    stop_requested = False

    try:
        for source_iter in build_sources(args):
            for sample in source_iter:
                total_seen += 1

                text = normalize_text(sample.text)
                if not accept_text(
                    text=text,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    min_ratio=args.min_cjk_ratio,
                ):
                    continue

                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if not digest_store.add_if_new(digest):
                    continue

                split = choose_split(
                    text=text,
                    validation_ratio=args.validation_ratio,
                    test_ratio=args.test_ratio,
                )
                writers[split].write(text + "\n")
                bytes_written = len(text.encode("utf-8")) + 1

                total_kept += 1
                total_written_bytes += bytes_written
                split_counts[split] += 1
                split_bytes[split] += bytes_written
                source_counts[sample.source] += 1
                source_split_counts[sample.source][split] += 1

                if total_kept % args.log_interval == 0:
                    print(
                        f"[mini-olmo] 已保留 {total_kept} 条样本，"
                        f"约 {total_written_bytes / (1024 ** 3):.2f} GiB，"
                        f"train={split_counts['train']}, "
                        f"validation={split_counts['validation']}, "
                        f"test={split_counts['test']}"
                    )
                if target_bytes and total_written_bytes >= target_bytes:
                    stop_requested = True
                    print(
                        f"[mini-olmo] 已达到目标体量 {args.target_size_gb:.2f} GiB，"
                        "停止继续收集样本。"
                    )
                    break
            if stop_requested:
                break
    finally:
        digest_store.close()
        for writer in writers.values():
            writer.close()

    stats = {
        "total_seen": total_seen,
        "total_kept": total_kept,
        "total_written_bytes": total_written_bytes,
        "total_written_gib": total_written_bytes / (1024 ** 3),
        "splits": dict(split_counts),
        "split_bytes": dict(split_bytes),
        "sources": dict(source_counts),
        "source_splits": {key: dict(value) for key, value in source_split_counts.items()},
        "dedupe_backend": dedupe_backend,
        "dedupe_db_path": dedupe_db_path,
        "args": vars(args),
        "output_paths": output_paths,
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[mini-olmo] 中文语料准备完成：")
    for split, path in output_paths.items():
        print(
            f"  - {split}: {path} "
            f"({split_counts[split]} 条, {split_bytes[split] / (1024 ** 3):.2f} GiB)"
        )
    print(f"  - total: {total_written_bytes / (1024 ** 3):.2f} GiB")
    print(f"  - stats: {stats_path}")

    return split_counts


def main() -> None:
    args = parse_args()
    write_corpus(args)


if __name__ == "__main__":
    main()
