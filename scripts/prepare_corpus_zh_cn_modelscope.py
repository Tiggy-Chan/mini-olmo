"""下载国内友好的中文高质量语料源。

这个脚本是独立入口，不修改现有训练 / 清洗代码。
它的目标是把“顶级优质语料”的下载入口优先换成：

- ModelScope 魔搭社区数据集
- 只有在显式选择时才使用的其他官方来源

当前仓库已有 `scripts/prepare_corpus_zh_v1.py`，但它更偏向在线流式抓取。
本脚本负责先把原始语料下载到本地，便于后续：

1. 先手工检查质量 / 许可 / 目录结构
2. 再转成 `txt / jsonl` 后传给 `--extra-text-dir`

示例：

```bash
python scripts/prepare_corpus_zh_cn_modelscope.py --list

python scripts/prepare_corpus_zh_cn_modelscope.py \
  --preset elite_cn \
  --output-root data/downloads/zh_corpora

python scripts/prepare_corpus_zh_cn_modelscope.py \
  --preset elite_cn_with_wiki \
  --dry-run
```
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from urllib import error, request


@dataclass(frozen=True)
class DirectFile:
    filename: str
    urls: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class CorpusSource:
    key: str
    title: str
    kind: str
    description: str
    quality: str
    size_hint: str
    reference_urls: tuple[str, ...]
    postprocess_note: str
    files: tuple[DirectFile, ...] = ()
    dataset_id: str = ""
    access_note: str = ""
    modelscope_include: tuple[str, ...] = ()
    estimated_download_gib: float | None = None


def build_cosmopedia_subset_includes(num_shards: int) -> tuple[str, ...]:
    shard_files = tuple(f"data/{index:05d}.parquet" for index in range(num_shards))
    return shard_files


CATALOG: dict[str, CorpusSource] = {
    "wikipedia_zh": CorpusSource(
        key="wikipedia_zh",
        title="中文维基百科官方 dump",
        kind="http_file",
        description="高质量通识知识语料，适合作为中文 base pretrain 的稳定骨架，但默认海外直链通常偏慢。",
        quality="elite",
        size_hint="中等",
        reference_urls=(
            "https://dumps.wikimedia.org/zhwiki/latest/",
        ),
        postprocess_note=(
            "下载结果是 Wikimedia XML dump，需要先抽取纯文本，再传给 "
            "`scripts/prepare_corpus_zh_v1.py --extra-text-dir`。"
        ),
        files=(
            DirectFile(
                filename="zhwiki-latest-pages-articles-multistream.xml.bz2",
                urls=(
                    "https://dumps.wikimedia.org/zhwiki/latest/"
                    "zhwiki-latest-pages-articles-multistream.xml.bz2",
                ),
                description="中文维基百科正文 dump",
            ),
            DirectFile(
                filename="zhwiki-latest-pages-articles-multistream-index.txt.bz2",
                urls=(
                    "https://dumps.wikimedia.org/zhwiki/latest/"
                    "zhwiki-latest-pages-articles-multistream-index.txt.bz2",
                ),
                description="对应正文 dump 的索引文件",
            ),
        ),
        access_note="这是海外官方直链；如果你更在意下载速度，建议优先使用默认的 ModelScope 预设，仅在需要维基料时再单独启用。",
    ),
    "wikisource_zh": CorpusSource(
        key="wikisource_zh",
        title="中文维基文库官方 dump",
        kind="http_file",
        description="公版文献 / 古典文本 / 规范书面文本补充源，适合增强中文长句和书面语风格，但默认海外直链通常偏慢。",
        quality="elite",
        size_hint="中等",
        reference_urls=(
            "https://dumps.wikimedia.org/zhwikisource/latest/",
        ),
        postprocess_note=(
            "下载结果同样是 XML dump，需要先抽取纯文本，再作为本地语料目录传入。"
        ),
        files=(
            DirectFile(
                filename="zhwikisource-latest-pages-articles-multistream.xml.bz2",
                urls=(
                    "https://dumps.wikimedia.org/zhwikisource/latest/"
                    "zhwikisource-latest-pages-articles-multistream.xml.bz2",
                ),
                description="中文维基文库正文 dump",
            ),
            DirectFile(
                filename="zhwikisource-latest-pages-articles-multistream-index.txt.bz2",
                urls=(
                    "https://dumps.wikimedia.org/zhwikisource/latest/"
                    "zhwikisource-latest-pages-articles-multistream-index.txt.bz2",
                ),
                description="对应正文 dump 的索引文件",
            ),
        ),
        access_note="这是海外官方直链；如果你更在意下载速度，建议优先使用默认的 ModelScope 预设，仅在需要文库料时再单独启用。",
    ),
    "chinese_cosmopedia": CorpusSource(
        key="chinese_cosmopedia",
        title="Chinese Cosmopedia 约20GiB子集（ModelScope）",
        kind="modelscope_dataset",
        description="高质量中文合成教材 / 故事 / 教程语料，默认只下载约 20GiB 子集，更适合低参数小模型起步。",
        quality="elite",
        size_hint="约20GiB",
        reference_urls=(
            "https://modelscope.cn/datasets/AI-ModelScope/chinese-cosmopedia",
            "https://community.modelscope.cn/678db5522db35d119533d31b.html",
        ),
        dataset_id="AI-ModelScope/chinese-cosmopedia",
        modelscope_include=build_cosmopedia_subset_includes(17),
        estimated_download_gib=19.3,
        postprocess_note=(
            "下载后通常需要把 `parquet/json/jsonl` 中的文本字段导出成 `txt/jsonl`，"
            "当前仓库现有语料入口不直接读取 parquet。"
        ),
        access_note="默认只拉前 17 个 parquet 分片，约 19-20GiB，避免全量 60GB+ 下载把单机磁盘打满。",
    ),
    "chinese_cosmopedia_full": CorpusSource(
        key="chinese_cosmopedia_full",
        title="Chinese Cosmopedia 全量（ModelScope）",
        kind="modelscope_dataset",
        description="Chinese Cosmopedia 全量下载入口，体量很大，只适合磁盘和带宽都比较宽裕时使用。",
        quality="elite",
        size_hint="60GiB+",
        reference_urls=(
            "https://modelscope.cn/datasets/AI-ModelScope/chinese-cosmopedia",
            "https://community.modelscope.cn/678db5522db35d119533d31b.html",
        ),
        dataset_id="AI-ModelScope/chinese-cosmopedia",
        estimated_download_gib=66.0,
        postprocess_note=(
            "这是全量版本。下载后通常需要把 `parquet/json/jsonl` 中的文本字段导出成 `txt/jsonl`，"
            "当前仓库现有语料入口不直接读取 parquet。"
        ),
        access_note="只有在你明确需要全量数据时才建议使用；单机默认不推荐。",
    ),
    "cci3_data": CorpusSource(
        key="cci3_data",
        title="CCI 3.0（ModelScope）",
        kind="modelscope_dataset",
        description="智源中文互联网语料 3.0，含 1000GB 全量与 498GB 高质量子集，是国内友好的网页类主力扩展源。",
        quality="elite",
        size_hint="超大",
        reference_urls=(
            "https://modelscope.cn/datasets/BAAI/CCI3-Data",
            "https://community.modelscope.cn/6704a27fcd8b2677c3cbdef9.html",
            "http://open.flopsera.com/flopsera-open/data-details/BAAI-CCI3",
        ),
        dataset_id="BAAI/CCI3-Data",
        postprocess_note=(
            "优先使用其中的高质量子集或你自己再筛过的目录。下载后建议先转成 `jsonl/txt`，"
            "再喂给现有 `--extra-text-dir`。"
        ),
        access_note="数据量很大，部分文件可能需要登录或先接受许可后再完整下载。",
    ),
    "cci4_base": CorpusSource(
        key="cci4_base",
        title="CCI 4.0 Base（ModelScope）",
        kind="modelscope_dataset",
        description="更大规模的新一代中英语料底座，适合后续超大扩展阶段，不建议默认全量下载到单卡实验机。",
        quality="massive",
        size_hint="极大",
        reference_urls=(
            "https://modelscope.cn/datasets/BAAI/CCI4.0-M2-Base-v1",
            "https://community.modelscope.cn/682fe7a201ee522510990122.html",
        ),
        dataset_id="BAAI/CCI4.0-M2-Base-v1",
        postprocess_note=(
            "体量非常大，建议只在你确定有足够磁盘、带宽和后处理流程时再启用。"
        ),
        access_note="更适合大规模集群或后续长线版本，不建议作为当前仓库的默认首选下载项。",
    ),
}


PRESETS: dict[str, list[str]] = {
    "elite_cn": [
        "chinese_cosmopedia",
    ],
    "elite_cn_plus_web": [
        "chinese_cosmopedia",
        "cci3_data",
    ],
    "elite_cn_massive": [
        "chinese_cosmopedia_full",
        "cci3_data",
        "cci4_base",
    ],
    "elite_cn_with_wiki": [
        "chinese_cosmopedia",
        "wikipedia_zh",
        "wikisource_zh",
    ],
    "elite_cn_full_mixed": [
        "chinese_cosmopedia_full",
        "cci3_data",
        "wikipedia_zh",
        "wikisource_zh",
    ],
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载国内友好的中文高质量语料源（优先 ModelScope / 国内友好入口）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只打印内置语料源和预设，不执行下载。",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        default=[],
        help="下载一个内置语料预设。可重复传入。",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(CATALOG),
        default=[],
        help="额外下载某个具体语料源。可重复传入。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/downloads/zh_corpora_cn_friendly",
        help="下载根目录，默认相对仓库根目录。",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="download_manifest.json",
        help="落盘的语料清单文件名。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的动作，不真的下载。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果目标文件已存在则覆盖重下。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="单次 HTTP 连接超时秒数。",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=4,
        help="HTTP 下载块大小（MiB）。",
    )
    parser.add_argument(
        "--modelscope-bin",
        type=str,
        default="",
        help="可选：显式指定 modelscope CLI 路径。",
    )
    parser.add_argument(
        "--modelscope-max-workers",
        type=int,
        default=2,
        help="ModelScope 并发下载线程数。默认 2，避免并发过高导致磁盘和临时文件暴涨。",
    )
    return parser.parse_args()


def unique_in_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_output_root(raw_path: str) -> Path:
    root = Path(raw_path)
    if root.is_absolute():
        return root
    return get_project_root() / root


def resolve_source_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []
    if not args.preset and not args.source and not args.list:
        keys.extend(PRESETS["elite_cn"])
    for preset_name in args.preset:
        keys.extend(PRESETS[preset_name])
    keys.extend(args.source)
    return unique_in_order(keys)


def format_bytes(size: int) -> str:
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


def summarize_kind(kind: str) -> str:
    if kind == "http_file":
        return "直链文件"
    if kind == "modelscope_dataset":
        return "ModelScope 数据集"
    return kind


def modelscope_command_hint(source: CorpusSource, dest_dir: Path) -> str:
    parts = [
        "modelscope download",
        f"--dataset {shlex.quote(source.dataset_id)}",
        f"--local_dir {shlex.quote(str(dest_dir))}",
        "--max-workers 2",
    ]
    if source.modelscope_include:
        include_args = " ".join(shlex.quote(item) for item in source.modelscope_include)
        parts.append(f"--include {include_args}")
    return " ".join(parts)


def http_command_hint(file: DirectFile, dest_path: Path) -> str:
    primary_url = file.urls[0]
    return f"curl -L -C - -o {shlex.quote(str(dest_path))} {shlex.quote(primary_url)}"


def print_catalog() -> None:
    print("[mini-olmo] 内置国内友好语料源：")
    for key in sorted(CATALOG):
        source = CATALOG[key]
        print(f"\n- {source.key}: {source.title}")
        print(f"  类型: {summarize_kind(source.kind)}")
        print(f"  级别: {source.quality}")
        print(f"  体量: {source.size_hint}")
        print(f"  说明: {source.description}")
        if source.dataset_id:
            print(f"  Dataset ID: {source.dataset_id}")
        if source.reference_urls:
            print(f"  参考链接: {source.reference_urls[0]}")
        if source.estimated_download_gib is not None:
            print(f"  预计下载量: 约 {source.estimated_download_gib:.1f}GiB")
        if source.access_note:
            print(f"  提示: {source.access_note}")

    print("\n[mini-olmo] 内置预设：")
    for name in sorted(PRESETS):
        joined = ", ".join(PRESETS[name])
        print(f"- {name}: {joined}")

    print("\n[mini-olmo] 默认预设: elite_cn（只走 ModelScope）")


def build_manifest_payload(
    output_root: Path,
    selected_sources: Sequence[CorpusSource],
    args: argparse.Namespace,
) -> dict[str, object]:
    source_payloads: list[dict[str, object]] = []
    for source in selected_sources:
        source_dir = output_root / source.key
        item = asdict(source)
        item["local_dir"] = str(source_dir)
        if source.kind == "modelscope_dataset":
            item["download_hint"] = modelscope_command_hint(source, source_dir)
        else:
            file_hints: dict[str, str] = {}
            for file in source.files:
                file_hints[file.filename] = http_command_hint(file, source_dir / file.filename)
            item["download_hint"] = file_hints
        source_payloads.append(item)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(get_project_root()),
        "output_root": str(output_root),
        "presets": list(args.preset) if args.preset else ["elite_cn"] if not args.source else [],
        "selected_source_keys": [source.key for source in selected_sources],
        "sources": source_payloads,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_source_info(source: CorpusSource, output_dir: Path) -> None:
    write_json(output_dir / "SOURCE_INFO.json", asdict(source))


def resolve_modelscope_bin(explicit_bin: str) -> str:
    if explicit_bin:
        if os.path.sep in explicit_bin:
            path = Path(explicit_bin)
            if path.exists():
                return str(path)
            return ""
        found = shutil.which(explicit_bin)
        return found or ""
    found = shutil.which("modelscope")
    return found or ""


def get_directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def get_available_disk_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def validate_modelscope_disk_budget(
    source: CorpusSource,
    output_dir: Path,
    max_workers: int,
) -> None:
    if source.estimated_download_gib is None:
        return

    estimated_total_bytes = int(source.estimated_download_gib * (1024 ** 3))
    existing_bytes = get_directory_size_bytes(output_dir)
    remaining_bytes = max(0, estimated_total_bytes - existing_bytes)
    worker_overhead_bytes = max(1, max_workers) * 512 * 1024 * 1024
    required_free_bytes = remaining_bytes + worker_overhead_bytes
    available_bytes = get_available_disk_bytes(output_dir)

    print(
        f"[mini-olmo] 磁盘检查: 预计总下载约 {format_bytes(estimated_total_bytes)}，"
        f"当前目录已有 {format_bytes(existing_bytes)}，"
        f"剩余可能还需 {format_bytes(remaining_bytes)}，"
        f"当前可用空间 {format_bytes(available_bytes)}。"
    )

    if available_bytes < required_free_bytes:
        raise RuntimeError(
            "磁盘剩余空间不足，已在下载前停止。\n"
            f"当前可用: {format_bytes(available_bytes)}\n"
            f"至少建议预留: {format_bytes(required_free_bytes)}\n"
            "如果你只是想做 20G 左右的小模型实验，建议继续使用默认 `elite_cn` 子集，"
            "并先清理旧的全量下载残留目录。"
        )


def modelscope_subset_already_satisfied(source: CorpusSource, output_dir: Path) -> bool:
    if not source.modelscope_include:
        return False
    return all((output_dir / relative_path).exists() for relative_path in source.modelscope_include)


def download_with_resume(
    url: str,
    dest_path: Path,
    *,
    force: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
) -> None:
    tmp_path = dest_path.with_name(dest_path.name + ".part")

    if force:
        if dest_path.exists():
            dest_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()

    if dest_path.exists():
        print(f"[mini-olmo] 已存在，跳过: {dest_path}")
        return

    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    req = request.Request(url, headers=headers)
    try:
        resp = request.urlopen(req, timeout=timeout_seconds)
    except error.HTTPError as exc:
        if exc.code == 416 and tmp_path.exists():
            os.replace(tmp_path, dest_path)
            print(f"[mini-olmo] 分片文件已完整，直接完成: {dest_path}")
            return
        raise

    status = getattr(resp, "status", 200)
    if resume_from > 0 and status != 206:
        resp.close()
        tmp_path.unlink(missing_ok=True)
        resume_from = 0
        req = request.Request(url)
        resp = request.urlopen(req, timeout=timeout_seconds)
        status = getattr(resp, "status", 200)

    with resp:
        content_length = resp.headers.get("Content-Length")
        total_bytes = None
        if content_length is not None:
            raw_length = int(content_length)
            total_bytes = raw_length + resume_from if status == 206 else raw_length

        mode = "ab" if resume_from > 0 and status == 206 else "wb"
        downloaded = resume_from if mode == "ab" else 0
        written_this_run = 0
        started_at = time.time()
        last_log_at = 0.0

        with tmp_path.open(mode) as f:
            while True:
                chunk = resp.read(chunk_size_bytes)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                written_this_run += len(chunk)

                now = time.time()
                if now - last_log_at >= 1.0:
                    if total_bytes is None:
                        progress = format_bytes(downloaded)
                    else:
                        progress = f"{format_bytes(downloaded)} / {format_bytes(total_bytes)}"
                    print(f"[mini-olmo] 下载中 {dest_path.name}: {progress}")
                    last_log_at = now

        os.replace(tmp_path, dest_path)
        elapsed = max(0.001, time.time() - started_at)
        speed = written_this_run / elapsed
        if total_bytes is None:
            total_display = format_bytes(downloaded)
        else:
            total_display = format_bytes(total_bytes)
        print(
            f"[mini-olmo] 下载完成 {dest_path.name}: {total_display}, "
            f"平均速度 {format_bytes(int(speed))}/s"
        )


def download_http_source(
    source: CorpusSource,
    output_dir: Path,
    *,
    dry_run: bool,
    force: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
) -> None:
    for file in source.files:
        dest_path = output_dir / file.filename
        if dry_run:
            print(f"[mini-olmo] DRY RUN 直链下载: {http_command_hint(file, dest_path)}")
            continue

        last_error: Exception | None = None
        for url in file.urls:
            try:
                print(f"[mini-olmo] 开始下载 {source.key}/{file.filename}")
                download_with_resume(
                    url,
                    dest_path,
                    force=force,
                    timeout_seconds=timeout_seconds,
                    chunk_size_bytes=chunk_size_bytes,
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[mini-olmo] URL 失败，尝试下一个候选地址: {url}\n  -> {exc}")

        if last_error is not None:
            raise RuntimeError(f"{source.key}/{file.filename} 下载失败") from last_error


def download_modelscope_source(
    source: CorpusSource,
    output_dir: Path,
    *,
    dry_run: bool,
    modelscope_bin: str,
    max_workers: int,
) -> None:
    hint = modelscope_command_hint(source, output_dir)
    if dry_run:
        print(f"[mini-olmo] DRY RUN ModelScope 下载: {hint}")
        return

    if not modelscope_bin:
        raise RuntimeError(
            "未找到 modelscope CLI。请先安装后重试，例如：\n"
            "  pip install modelscope\n"
            "如果你更希望走国内加速源，可参考 ModelScope 官方安装说明。"
        )

    if modelscope_subset_already_satisfied(source, output_dir):
        print("[mini-olmo] 目标子集所需的 parquet 已经齐全，跳过 ModelScope 下载。")
        return

    validate_modelscope_disk_budget(source, output_dir, max_workers)

    cmd = [
        modelscope_bin,
        "download",
        "--dataset",
        source.dataset_id,
        "--local_dir",
        str(output_dir),
        "--max-workers",
        str(max(1, max_workers)),
    ]
    if source.modelscope_include:
        cmd.append("--include")
        cmd.extend(source.modelscope_include)
    print(f"[mini-olmo] 执行命令: {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, check=True)


def download_source(
    source: CorpusSource,
    output_root: Path,
    *,
    dry_run: bool,
    force: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    modelscope_bin: str,
    modelscope_max_workers: int,
) -> None:
    output_dir = output_root / source.key
    output_dir.mkdir(parents=True, exist_ok=True)
    write_source_info(source, output_dir)

    print(f"\n[mini-olmo] ===== {source.key} / {source.title} =====")
    print(f"[mini-olmo] 类型: {summarize_kind(source.kind)}")
    print(f"[mini-olmo] 说明: {source.description}")
    if source.access_note:
        print(f"[mini-olmo] 提示: {source.access_note}")

    if source.kind == "http_file":
        download_http_source(
            source,
            output_dir,
            dry_run=dry_run,
            force=force,
            timeout_seconds=timeout_seconds,
            chunk_size_bytes=chunk_size_bytes,
        )
        return

    if source.kind == "modelscope_dataset":
        download_modelscope_source(
            source,
            output_dir,
            dry_run=dry_run,
            modelscope_bin=modelscope_bin,
            max_workers=modelscope_max_workers,
        )
        return

    raise ValueError(f"未知语料类型: {source.kind}")


def print_postprocess_summary(selected_sources: Sequence[CorpusSource], output_root: Path) -> None:
    print("\n[mini-olmo] 下载后处理建议：")
    for source in selected_sources:
        print(f"- {source.key}: {source.postprocess_note}")
        print(f"  本地目录: {output_root / source.key}")


def main() -> int:
    args = parse_args()
    if args.list:
        print_catalog()
        return 0

    source_keys = resolve_source_keys(args)
    if not source_keys:
        print("[mini-olmo] 未选择任何语料源。可用 `--list` 查看选项。", file=sys.stderr)
        return 1

    selected_sources = [CATALOG[key] for key in source_keys]
    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if any(source.kind == "http_file" for source in selected_sources):
        print(
            "[mini-olmo] 提示：你当前选择里包含海外 HTTP 直链源；"
            "如果更在意国内下载速度，建议改用默认 `elite_cn` 或 `elite_cn_plus_web`。"
        )

    manifest_payload = build_manifest_payload(output_root, selected_sources, args)
    manifest_path = output_root / args.manifest_name
    write_json(manifest_path, manifest_payload)
    print(f"[mini-olmo] 已写入语料清单: {manifest_path}")

    modelscope_bin = resolve_modelscope_bin(args.modelscope_bin)
    chunk_size_bytes = max(1, args.chunk_size_mib) * 1024 * 1024

    failures: list[tuple[str, str]] = []
    for source in selected_sources:
        try:
            download_source(
                source,
                output_root,
                dry_run=args.dry_run,
                force=args.force,
                timeout_seconds=args.timeout_seconds,
                chunk_size_bytes=chunk_size_bytes,
                modelscope_bin=modelscope_bin,
                modelscope_max_workers=args.modelscope_max_workers,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((source.key, str(exc)))
            print(f"[mini-olmo] 失败: {source.key} -> {exc}", file=sys.stderr)

    print_postprocess_summary(selected_sources, output_root)

    if failures:
        print("\n[mini-olmo] 以下语料源未成功完成：", file=sys.stderr)
        for key, message in failures:
            print(f"- {key}: {message}", file=sys.stderr)
        return 1

    print("\n[mini-olmo] 所有选中语料源处理完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
