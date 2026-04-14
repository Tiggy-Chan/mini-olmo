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
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Iterable, Iterator, List, Sequence

from datasets import load_dataset

SPLIT_NAMES = ("train", "validation", "test")
RESUME_STATE_FILENAME = "resume_state.json"


@dataclass(frozen=True)
class TextSample:
    text: str
    source: str


class DigestStore:
    def __init__(
        self,
        backend: str,
        db_path: str | None = None,
        commit_interval: int = 5000,
    ) -> None:
        self.backend = backend
        self._memory_store: set[str] | None = None
        self._conn: sqlite3.Connection | None = None
        self._pending = 0
        self._commit_interval = max(1, commit_interval)

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
        if self._pending >= self._commit_interval:
            self._conn.commit()
            self._pending = 0
        return cursor.rowcount == 1

    def reset(self) -> None:
        if self.backend == "memory":
            assert self._memory_store is not None
            self._memory_store.clear()
            return

        assert self._conn is not None
        self._conn.execute("DELETE FROM digests")
        self._conn.commit()
        self._pending = 0

    def close(self) -> None:
        if self._conn is not None:
            if self._pending > 0:
                self._conn.commit()
                self._pending = 0
            self._conn.close()


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_resume_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, RESUME_STATE_FILENAME)


def write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json_if_exists(path: str) -> Dict[str, object] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"JSON 文件格式不正确: {path}")


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
        "--disable-shuffle",
        action="store_true",
        help="Skip streaming shuffle for faster iteration. This improves throughput but weakens sample mixing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted build by appending to existing outputs and rebuilding dedupe state.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Delete existing outputs for this corpus_name and rebuild from scratch. Use with care.",
    )
    parser.add_argument(
        "--sqlite-commit-interval",
        type=int,
        default=5_000,
        help="Commit sqlite dedupe state every N insert attempts. Higher is faster but less crash-resilient.",
    )
    parser.add_argument(
        "--source-retries",
        type=int,
        default=8,
        help="Retry count for transient remote dataset streaming failures before aborting.",
    )
    parser.add_argument(
        "--source-retry-delay-seconds",
        type=float,
        default=3.0,
        help="Seconds to wait before reopening a remote streaming source after a transient error.",
    )
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


def maybe_shuffle_dataset(
    dataset: Iterable[Dict[str, object]],
    seed: int,
    buffer_size: int,
    disable_shuffle: bool,
) -> Iterable[Dict[str, object]]:
    if disable_shuffle:
        return dataset
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)


def maybe_skip_dataset(
    dataset: Iterable[Dict[str, object]],
    skip_rows: int,
) -> Iterable[Dict[str, object]]:
    if skip_rows <= 0:
        return dataset
    return dataset.skip(skip_rows)


def iter_remote_samples_with_retries(
    *,
    source_label: str,
    source_name: str,
    max_docs: int,
    max_retries: int,
    retry_delay_seconds: float,
    dataset_factory: Callable[[int], Iterable[Dict[str, object]]],
) -> Iterator[TextSample]:
    yielded_docs = 0
    rows_seen = 0
    attempt = 0
    first_open = True

    while yielded_docs < max_docs:
        if first_open:
            print(f"[mini-olmo] 加载{source_label} …")
            first_open = False
        elif rows_seen > 0:
            print(
                f"[mini-olmo] 重新打开{source_label}（从第 {rows_seen} 条原始记录附近继续，"
                f"当前已产出 {yielded_docs} 条样本）…"
            )
        else:
            print(f"[mini-olmo] 重新打开{source_label} …")

        dataset = dataset_factory(rows_seen)

        try:
            for row in dataset:
                rows_seen += 1
                text = extract_text_from_row(row)
                if not isinstance(text, str):
                    continue

                yield TextSample(text=text, source=source_name)
                yielded_docs += 1
                if yielded_docs >= max_docs:
                    return
            return
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"{source_label} 在读取过程中连续失败，超过 {max_retries} 次脚本级重试上限。"
                ) from exc

            print(
                f"[mini-olmo] {source_label} 读取异常：{exc!r}。"
                f" {retry_delay_seconds:.1f}s 后进行脚本级重试 ({attempt}/{max_retries})。"
            )
            time.sleep(max(0.0, retry_delay_seconds))


def iter_wikipedia_samples(
    max_docs: int,
    seed: int,
    buffer_size: int,
    disable_shuffle: bool,
    max_retries: int,
    retry_delay_seconds: float,
) -> Iterator[TextSample]:
    def dataset_factory(skip_rows: int) -> Iterable[Dict[str, object]]:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.zh",
            split="train",
            streaming=True,
        )
        dataset = maybe_shuffle_dataset(dataset, seed, buffer_size, disable_shuffle)
        return maybe_skip_dataset(dataset, skip_rows)

    yield from iter_remote_samples_with_retries(
        source_label="中文 Wikipedia",
        source_name="zh_wikipedia",
        max_docs=max_docs,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        dataset_factory=dataset_factory,
    )


def iter_cosmopedia_samples(
    max_docs: int,
    seed: int,
    buffer_size: int,
    disable_shuffle: bool,
    max_retries: int,
    retry_delay_seconds: float,
) -> Iterator[TextSample]:
    def dataset_factory(skip_rows: int) -> Iterable[Dict[str, object]]:
        dataset = load_dataset(
            "opencsg/chinese-cosmopedia",
            split="train",
            streaming=True,
        )
        dataset = maybe_shuffle_dataset(dataset, seed, buffer_size, disable_shuffle)
        return maybe_skip_dataset(dataset, skip_rows)

    yield from iter_remote_samples_with_retries(
        source_label="中文 Cosmopedia 抽样",
        source_name="chinese_cosmopedia",
        max_docs=max_docs,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        dataset_factory=dataset_factory,
    )


def iter_fineweb_samples(
    max_docs: int,
    seed: int,
    data_dir: str,
    buffer_size: int,
    disable_shuffle: bool,
    max_retries: int,
    retry_delay_seconds: float,
) -> Iterator[TextSample]:
    def dataset_factory(skip_rows: int) -> Iterable[Dict[str, object]]:
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

        dataset = maybe_shuffle_dataset(dataset, seed, buffer_size, disable_shuffle)
        return maybe_skip_dataset(dataset, skip_rows)

    yield from iter_remote_samples_with_retries(
        source_label=f"中文 FineWeb Edu 抽样（data_dir={data_dir}）",
        source_name=f"fineweb_edu_zh::{data_dir}",
        max_docs=max_docs,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        dataset_factory=dataset_factory,
    )


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
                args.disable_shuffle,
                args.source_retries,
                args.source_retry_delay_seconds,
            )
        )
    if args.include_cosmopedia:
        sources.append(
            iter_cosmopedia_samples(
                args.max_cosmopedia_docs,
                args.seed,
                args.streaming_buffer_size,
                args.disable_shuffle,
                args.source_retries,
                args.source_retry_delay_seconds,
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
                    args.disable_shuffle,
                    args.source_retries,
                    args.source_retry_delay_seconds,
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


def collect_existing_artifact_paths(
    output_paths: Dict[str, str],
    state_path: str,
    stats_path: str,
    dedupe_db_path: str | None,
) -> List[str]:
    candidates = list(output_paths.values()) + [state_path, stats_path]
    if dedupe_db_path:
        candidates.extend(
            [
                dedupe_db_path,
                f"{dedupe_db_path}-wal",
                f"{dedupe_db_path}-shm",
            ]
        )
    return [path for path in candidates if os.path.exists(path)]


def clear_existing_artifacts(paths: Sequence[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def guard_or_reset_existing_outputs(
    *,
    args: argparse.Namespace,
    output_paths: Dict[str, str],
    state_path: str,
    stats_path: str,
    dedupe_db_path: str | None,
) -> None:
    existing_paths = collect_existing_artifact_paths(
        output_paths=output_paths,
        state_path=state_path,
        stats_path=stats_path,
        dedupe_db_path=dedupe_db_path,
    )
    if not existing_paths or args.resume:
        return

    if args.overwrite_existing:
        print("[mini-olmo] 检测到已有语料产物，按 --overwrite-existing 清理后从头构建。")
        for path in existing_paths:
            print(f"  - 删除: {path}")
        clear_existing_artifacts(existing_paths)
        return

    details = "\n".join(f"  - {path}" for path in existing_paths[:10])
    raise ValueError(
        "检测到当前 corpus_name 已存在历史语料产物；为避免误覆盖，脚本已停止。\n"
        "如果你是想继续上次中断的构建，请在相同参数上加 `--resume`。\n"
        "如果你是想丢弃旧结果并从头重建，请显式传 `--overwrite-existing`。\n"
        f"{details}"
    )


def build_resume_config(
    args: argparse.Namespace,
    dedupe_backend: str,
    dedupe_db_path: str | None,
) -> Dict[str, object]:
    return {
        "corpus_name": args.corpus_name,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "min_cjk_ratio": args.min_cjk_ratio,
        "skip_wikipedia": args.skip_wikipedia,
        "max_wikipedia_docs": args.max_wikipedia_docs,
        "include_cosmopedia": args.include_cosmopedia,
        "max_cosmopedia_docs": args.max_cosmopedia_docs,
        "include_fineweb": args.include_fineweb,
        "max_fineweb_docs": args.max_fineweb_docs,
        "fineweb_data_dir": list(args.fineweb_data_dir),
        "extra_text_dir": [os.path.abspath(path) for path in args.extra_text_dir],
        "dedupe_backend": dedupe_backend,
        "dedupe_db_path": dedupe_db_path,
    }


def validate_resume_config(
    previous_state: Dict[str, object] | None,
    current_config: Dict[str, object],
) -> None:
    if previous_state is None:
        return

    previous_config = previous_state.get("resume_config")
    if not isinstance(previous_config, dict):
        return

    mismatches: List[str] = []
    for key, current_value in current_config.items():
        previous_value = previous_config.get(key)
        if previous_value != current_value:
            mismatches.append(f"{key}: 上次={previous_value!r}, 本次={current_value!r}")

    if mismatches:
        details = "\n".join(f"  - {item}" for item in mismatches[:10])
        raise ValueError(
            "检测到 `--resume` 的关键语料参数和已有断点状态不一致。\n"
            "请保持相同的数据配置继续续跑，或换一个新的 --corpus-name。\n"
            f"{details}"
        )


def trim_incomplete_last_line(path: str) -> int:
    if not os.path.exists(path):
        return 0

    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return 0

        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0

        chunk_size = 8192
        position = size
        newline_pos = -1
        while position > 0 and newline_pos < 0:
            read_size = min(chunk_size, position)
            position -= read_size
            f.seek(position)
            chunk = f.read(read_size)
            idx = chunk.rfind(b"\n")
            if idx >= 0:
                newline_pos = position + idx

        truncate_pos = 0 if newline_pos < 0 else newline_pos + 1
        removed_bytes = size - truncate_pos
        f.truncate(truncate_pos)
        return removed_bytes


def rebuild_digest_store_from_outputs(
    output_paths: Dict[str, str],
    digest_store: DigestStore,
    log_interval: int,
) -> tuple[Counter, Counter]:
    split_counts = Counter()
    split_bytes = Counter()
    restored_lines = 0

    for split in SPLIT_NAMES:
        path = output_paths[split]
        removed_bytes = trim_incomplete_last_line(path)
        if removed_bytes:
            print(
                f"[mini-olmo] 恢复模式：检测到 {split} 文件末尾存在未写完的半行，"
                f"已裁剪 {removed_bytes} 字节。"
            )
        if not os.path.exists(path):
            continue

        print(f"[mini-olmo] 恢复模式：扫描已有 {split} 文件 {path} …")
        with open(path, "rb") as f:
            for raw_line in f:
                text_bytes = raw_line.rstrip(b"\r\n")
                digest = hashlib.sha1(text_bytes).hexdigest()
                digest_store.add_if_new(digest)
                split_counts[split] += 1
                split_bytes[split] += len(raw_line)
                restored_lines += 1

                if log_interval > 0 and restored_lines % log_interval == 0:
                    print(f"[mini-olmo] 恢复模式：已重建 {restored_lines} 条已有样本的去重索引。")

    return split_counts, split_bytes


def restore_source_counters(
    previous_state: Dict[str, object] | None,
    split_counts: Counter,
    split_bytes: Counter,
) -> tuple[Counter, DefaultDict[str, Counter], int, bool]:
    source_split_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    restored_total_kept = sum(split_counts.values())
    if restored_total_kept == 0:
        return Counter(), source_split_counts, 0, True

    if previous_state is None:
        return Counter(), source_split_counts, 0, False

    saved_splits = Counter(
        {
            split: int(value)
            for split, value in previous_state.get("splits", {}).items()
            if split in SPLIT_NAMES
        }
    )
    saved_split_bytes = Counter(
        {
            split: int(value)
            for split, value in previous_state.get("split_bytes", {}).items()
            if split in SPLIT_NAMES
        }
    )
    if saved_splits != split_counts or saved_split_bytes != split_bytes:
        return Counter(), source_split_counts, 0, False

    source_counts = Counter(
        {name: int(value) for name, value in previous_state.get("sources", {}).items()}
    )
    raw_source_splits = previous_state.get("source_splits", {})
    if isinstance(raw_source_splits, dict):
        for source_name, split_map in raw_source_splits.items():
            if not isinstance(split_map, dict):
                continue
            source_split_counts[source_name] = Counter(
                {split: int(value) for split, value in split_map.items() if split in SPLIT_NAMES}
            )

    total_seen = int(previous_state.get("total_seen", 0))
    return source_counts, source_split_counts, total_seen, True


def build_run_snapshot(
    args: argparse.Namespace,
    dedupe_backend: str,
    dedupe_db_path: str | None,
    output_paths: Dict[str, str],
    split_counts: Counter,
    split_bytes: Counter,
    source_counts: Counter,
    source_split_counts: DefaultDict[str, Counter],
    total_seen: int,
    total_written_bytes: int,
    resume_config: Dict[str, object],
    resume_info: Dict[str, object],
    source_stats_complete: bool,
) -> Dict[str, object]:
    return {
        "total_seen": total_seen,
        "total_kept": sum(split_counts.values()),
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
        "resume_config": resume_config,
        "resume_info": resume_info,
        "source_stats_complete": source_stats_complete,
    }


def write_corpus(args: argparse.Namespace) -> Dict[str, int]:
    if args.validation_ratio + args.test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio 必须小于 1.0")
    if args.resume and args.overwrite_existing:
        raise ValueError("`--resume` 和 `--overwrite-existing` 不能同时使用。")

    output_dir = ensure_output_dir(args.corpus_name)
    dedupe_backend = resolve_dedupe_backend(args)
    dedupe_db_path = resolve_dedupe_db_path(args, output_dir)
    target_bytes = int(args.target_size_gb * (1024 ** 3)) if args.target_size_gb > 0 else 0
    state_path = get_resume_state_path(output_dir)
    stats_path = os.path.join(output_dir, "stats.json")
    output_paths = {
        split: os.path.join(output_dir, f"{split}.txt")
        for split in SPLIT_NAMES
    }
    guard_or_reset_existing_outputs(
        args=args,
        output_paths=output_paths,
        state_path=state_path,
        stats_path=stats_path,
        dedupe_db_path=dedupe_db_path,
    )
    previous_state = load_json_if_exists(state_path) if args.resume else None
    resume_config = build_resume_config(args, dedupe_backend, dedupe_db_path)
    if args.resume:
        validate_resume_config(previous_state, resume_config)

    digest_store = DigestStore(
        dedupe_backend,
        dedupe_db_path,
        commit_interval=args.sqlite_commit_interval,
    )
    restored_split_counts = Counter()
    restored_split_bytes = Counter()
    if args.resume:
        digest_store.reset()
        restored_split_counts, restored_split_bytes = rebuild_digest_store_from_outputs(
            output_paths,
            digest_store,
            args.log_interval,
        )

    source_counts, source_split_counts, restored_total_seen, source_stats_complete = restore_source_counters(
        previous_state,
        restored_split_counts,
        restored_split_bytes,
    )
    if args.resume and sum(restored_split_counts.values()) > 0:
        if source_stats_complete:
            print("[mini-olmo] 恢复模式：已从断点状态恢复 source 级统计。")
        else:
            print(
                "[mini-olmo] 恢复模式：已有输出文件已恢复，但缺少完全匹配的断点状态；"
                "最终 source 级统计将只覆盖本次续跑新增部分。"
            )

    writers = {
        split: open(path, "ab" if args.resume else "wb")
        for split, path in output_paths.items()
    }

    split_counts = Counter(restored_split_counts)
    split_bytes = Counter(restored_split_bytes)
    total_seen = restored_total_seen
    total_kept = sum(split_counts.values())
    total_written_bytes = sum(split_bytes.values())
    stop_requested = False
    run_status = "running"
    resume_info = {
        "resume_requested": args.resume,
        "restored_total_kept": total_kept,
        "restored_total_written_bytes": total_written_bytes,
        "restored_total_seen": restored_total_seen if source_stats_complete else None,
        "state_path": state_path,
    }
    progress_save_interval = max(1, min(args.log_interval, 5_000))

    def save_resume_state(status: str) -> None:
        snapshot = build_run_snapshot(
            args=args,
            dedupe_backend=dedupe_backend,
            dedupe_db_path=dedupe_db_path,
            output_paths=output_paths,
            split_counts=split_counts,
            split_bytes=split_bytes,
            source_counts=source_counts,
            source_split_counts=source_split_counts,
            total_seen=total_seen,
            total_written_bytes=total_written_bytes,
            resume_config=resume_config,
            resume_info={**resume_info, "status": status},
            source_stats_complete=source_stats_complete,
        )
        write_json_atomic(state_path, snapshot)

    if args.resume and total_kept > 0:
        print(
            f"[mini-olmo] 恢复模式：已恢复 {total_kept} 条样本，"
            f"约 {total_written_bytes / (1024 ** 3):.2f} GiB。"
        )

    if target_bytes and total_written_bytes >= target_bytes:
        stop_requested = True
        print(
            f"[mini-olmo] 恢复模式：当前已有语料已达到目标体量 "
            f"{args.target_size_gb:.2f} GiB，无需继续收集。"
        )

    try:
        if not stop_requested:
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

                    encoded_text = text.encode("utf-8")
                    digest = hashlib.sha1(encoded_text).hexdigest()
                    if not digest_store.add_if_new(digest):
                        continue

                    split = choose_split(
                        text=text,
                        validation_ratio=args.validation_ratio,
                        test_ratio=args.test_ratio,
                    )
                    writers[split].write(encoded_text + b"\n")
                    bytes_written = len(encoded_text) + 1

                    total_kept += 1
                    total_written_bytes += bytes_written
                    split_counts[split] += 1
                    split_bytes[split] += bytes_written
                    source_counts[sample.source] += 1
                    source_split_counts[sample.source][split] += 1

                    if total_kept % progress_save_interval == 0:
                        save_resume_state("running")
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
        run_status = "completed"
    except KeyboardInterrupt:
        run_status = "interrupted"
        raise
    except Exception:
        run_status = "failed"
        raise
    finally:
        for writer in writers.values():
            writer.flush()
            writer.close()
        digest_store.close()
        save_resume_state(run_status)

    stats = build_run_snapshot(
        args=args,
        dedupe_backend=dedupe_backend,
        dedupe_db_path=dedupe_db_path,
        output_paths=output_paths,
        split_counts=split_counts,
        split_bytes=split_bytes,
        source_counts=source_counts,
        source_split_counts=source_split_counts,
        total_seen=total_seen,
        total_written_bytes=total_written_bytes,
        resume_config=resume_config,
        resume_info={**resume_info, "status": run_status},
        source_stats_complete=source_stats_complete,
    )

    write_json_atomic(stats_path, stats)

    print("[mini-olmo] 中文语料准备完成：")
    for split, path in output_paths.items():
        print(
            f"  - {split}: {path} "
            f"({split_counts[split]} 条, {split_bytes[split] / (1024 ** 3):.2f} GiB)"
        )
    print(f"  - total: {total_written_bytes / (1024 ** 3):.2f} GiB")
    print(f"  - stats: {stats_path}")
    if not source_stats_complete:
        print("  - note: 断点恢复时未能完整还原旧的 source 级统计，sources/source_splits 仅对新写入部分精确。")

    return split_counts


def main() -> None:
    args = parse_args()
    write_corpus(args)


if __name__ == "__main__":
    main()
