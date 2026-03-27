"""准备第二阶段中文聊天 SFT 数据。

输出目录：
- data/sft/<dataset_name>/train.jsonl
- data/sft/<dataset_name>/validation.jsonl
- data/sft/<dataset_name>/test.jsonl

支持：
- 内置中文聊天数据配方，直接从 Hugging Face 下载并抽样
- 从本地 .json / .jsonl 文件读取常见中文指令/对话格式
- 统一清洗为 {"messages": [...]} 结构，便于直接喂给 sft.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from datasets import load_dataset


BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"作为一个\s*ai语言模型",
        r"作为一名\s*ai助手",
        r"我是一个\s*ai语言模型",
        r"作为人工智能",
        r"作为大语言模型",
    )
]


@dataclass(frozen=True)
class HFDatasetSpec:
    dataset_name: str
    source_name: str
    split: str = "train"
    config_name: str | None = None
    max_records: int = 0
    streaming: bool = True


@dataclass(frozen=True)
class LabeledRecord:
    source: str
    record: Dict[str, Any]


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the second-stage Chinese chat SFT dataset")
    parser.add_argument("--dataset-name", type=str, default="zh_sft_stage2_v1")
    parser.add_argument(
        "--recipe",
        type=str,
        default="chat_v1",
        choices=("none", "chat_v1"),
        help="Built-in dataset recipe for stage-2 Chinese chat tuning.",
    )
    parser.add_argument("--input-path", action="append", default=[])
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.005)
    parser.add_argument("--min-message-chars", type=int, default=2)
    parser.add_argument("--min-assistant-chars", type=int, default=8)
    parser.add_argument("--max-message-chars", type=int, default=1200)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.55)
    parser.add_argument("--max-belle-records", type=int, default=180_000)
    parser.add_argument("--max-alpaca-records", type=int, default=80_000)
    parser.add_argument("--streaming-buffer-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--no-seed-set", action="store_true")
    parser.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help="Keep samples containing classic 'as an AI model' boilerplate.",
    )
    return parser.parse_args()


def recipe_specs(args: argparse.Namespace) -> List[HFDatasetSpec]:
    if args.recipe == "none":
        return []
    return [
        HFDatasetSpec(
            dataset_name="BelleGroup/train_1M_CN",
            source_name="belle_1m_cn",
            split="train",
            max_records=args.max_belle_records,
            streaming=True,
        ),
        HFDatasetSpec(
            dataset_name="FreedomIntelligence/alpaca-gpt4-chinese",
            source_name="alpaca_gpt4_zh",
            split="train",
            max_records=args.max_alpaca_records,
            streaming=True,
        ),
    ]


def seed_examples() -> List[Dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": "你好，请介绍一下你自己。"},
                {"role": "assistant", "content": "我是一个中文小模型，目前专注于中文问答、解释、总结和基础聊天。"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "什么是机器学习？请用简单中文解释。"},
                {
                    "role": "assistant",
                    "content": "机器学习是一种让计算机从数据中总结规律的方法。它不需要把所有规则都手写出来，而是通过例子逐渐学会预测、分类或生成。",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "请用三句话介绍一下深度学习。"},
                {
                    "role": "assistant",
                    "content": "深度学习是机器学习中的一类方法，通常使用多层神经网络处理复杂任务。它擅长从大量数据中自动学习特征，因此在图像、语音和文本任务中表现突出。代价是训练成本更高，对数据量和算力的要求也更强。",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "如果你不确定答案，应该怎么回答？"},
                {
                    "role": "assistant",
                    "content": "如果我不确定，我应该直接说明不确定，并给出有限判断或建议用户进一步核实，而不是假装自己知道。",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "请把下面这句话改写得更正式一些：这个方法挺好用的。"},
                {"role": "assistant", "content": "该方法具有较好的实用性。"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "为什么 8GB 显存训练大模型会困难？"},
                {
                    "role": "assistant",
                    "content": "因为模型参数、梯度、优化器状态和中间激活都会占用显存，显存越小，可容纳的模型规模、batch size 和序列长度就越受限制。",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "请给出一个礼貌拒绝示例：用户要求你编造不存在的论文结论。"},
                {
                    "role": "assistant",
                    "content": "我不能编造不存在的论文结论。如果你愿意，我可以帮你整理真实论文的结论，或者先一起确认论文来源。",
                },
            ]
        },
    ]


def normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


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
        elif "a" <= ch.lower() <= "z":
            latin_count += 1
    denom = cjk_count + latin_count
    if denom == 0:
        return 0.0
    return cjk_count / denom


def choose_split(key: str, validation_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    test_threshold = int(test_ratio * 10_000)
    val_threshold = int((test_ratio + validation_ratio) * 10_000)
    if bucket < test_threshold:
        return "test"
    if bucket < val_threshold:
        return "validation"
    return "train"


def output_dir_for(dataset_name: str) -> str:
    root = get_project_root()
    out_dir = os.path.join(root, "data", "sft", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def normalize_role(role: Any) -> str | None:
    role_value = str(role).strip().lower()
    role_map = {
        "system": "system",
        "user": "user",
        "human": "user",
        "question": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "answer": "assistant",
    }
    return role_map.get(role_value)


def extract_message_text(item: Dict[str, Any]) -> str:
    for key in ("content", "value", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def merge_consecutive_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] = f"{merged[-1]['content']}\n{message['content']}"
        else:
            merged.append(dict(message))
    return merged


def contains_boilerplate(text: str) -> bool:
    lowered = normalize_text(text)
    return any(pattern.search(lowered) for pattern in BOILERPLATE_PATTERNS)


def parse_history(history: Any) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if not isinstance(history, list):
        return messages

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text = normalize_text(item[0])
            assistant_text = normalize_text(item[1])
            if user_text and assistant_text:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
            continue
        if isinstance(item, dict):
            user_text = ""
            assistant_text = ""
            for key in ("user", "human", "question", "instruction", "prompt"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    user_text = normalize_text(value)
                    break
            for key in ("assistant", "gpt", "answer", "output", "response"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    assistant_text = normalize_text(value)
                    break
            if user_text and assistant_text:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
    return messages


def sanitize_messages(
    messages: Iterable[Dict[str, Any]],
    min_message_chars: int,
    min_assistant_chars: int,
    max_message_chars: int,
    max_turns: int,
    min_cjk_ratio: float,
    keep_boilerplate: bool,
) -> List[Dict[str, str]]:
    sanitized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role"))
        if role is None:
            role = normalize_role(item.get("from"))
        if role is None:
            continue
        content = extract_message_text(item)
        if not content:
            continue
        if len(content) < min_message_chars or len(content) > max_message_chars:
            return []
        sanitized.append({"role": role, "content": content})

    sanitized = merge_consecutive_messages(sanitized)
    if not sanitized:
        return []

    if sanitized[0]["role"] == "system":
        head = [sanitized[0]]
        body = sanitized[1:]
    else:
        head = []
        body = sanitized

    if not body or body[0]["role"] != "user":
        return []

    expected_role = "user"
    for message in body:
        if message["role"] != expected_role:
            return []
        expected_role = "assistant" if expected_role == "user" else "user"

    trimmed_body = body[-max_turns:]
    if trimmed_body and trimmed_body[0]["role"] == "assistant":
        trimmed_body = trimmed_body[1:]
    if not trimmed_body or trimmed_body[0]["role"] != "user" or trimmed_body[-1]["role"] != "assistant":
        return []

    if head and len(trimmed_body) + 1 <= max_turns:
        full_messages = head + trimmed_body
    else:
        full_messages = trimmed_body
    if full_messages[-1]["role"] != "assistant":
        return []

    assistant_messages = [item["content"] for item in full_messages if item["role"] == "assistant"]
    if not assistant_messages:
        return []
    if min(len(text) for text in assistant_messages) < min_assistant_chars:
        return []
    if not keep_boilerplate and any(contains_boilerplate(text) for text in assistant_messages):
        return []

    combined_text = "\n".join(item["content"] for item in full_messages)
    if cjk_ratio(combined_text) < min_cjk_ratio:
        return []
    return full_messages


def first_text(record: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def record_from_messages(record: Dict[str, Any]) -> Dict[str, Any] | None:
    messages = record.get("messages")
    if isinstance(messages, list):
        return {"messages": messages}
    return None


def record_from_instruction(record: Dict[str, Any]) -> Dict[str, Any] | None:
    instruction = first_text(record, ("instruction", "instruction_zh"))
    optional_input = first_text(record, ("input", "context"))
    output = first_text(record, ("output", "response", "answer", "target"))
    system_text = first_text(record, ("system", "system_prompt"))
    if not instruction or not output:
        return None

    messages = parse_history(record.get("history"))
    if system_text:
        messages.insert(0, {"role": "system", "content": system_text})
    user_text = instruction if not optional_input else f"{instruction}\n{optional_input}"
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": output})
    return {"messages": messages}


def record_from_question_answer(record: Dict[str, Any]) -> Dict[str, Any] | None:
    question = first_text(record, ("question", "query"))
    answer = first_text(record, ("answer", "response"))
    if not question or not answer:
        return None
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def record_from_prompt_response(record: Dict[str, Any]) -> Dict[str, Any] | None:
    prompt = first_text(record, ("prompt",))
    response = first_text(record, ("response", "output"))
    if not prompt or not response:
        return None
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def record_from_conversations(record: Dict[str, Any]) -> Dict[str, Any] | None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    messages: List[Dict[str, str]] = []
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role"))
        if role is None:
            role = normalize_role(item.get("from"))
        content = extract_message_text(item)
        if role and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None
    return {"messages": messages}


def normalize_record(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    candidate = (
        record_from_messages(record)
        or record_from_instruction(record)
        or record_from_question_answer(record)
        or record_from_prompt_response(record)
        or record_from_conversations(record)
    )
    if candidate is None:
        return None

    messages = sanitize_messages(
        candidate["messages"],
        min_message_chars=args.min_message_chars,
        min_assistant_chars=args.min_assistant_chars,
        max_message_chars=args.max_message_chars,
        max_turns=args.max_turns,
        min_cjk_ratio=args.min_cjk_ratio,
        keep_boilerplate=args.keep_boilerplate,
    )
    if not messages:
        return None
    return {"messages": messages}


def iter_records_from_file(path: str) -> Iterator[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif ext == ".json":
            payload = json.load(f)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(payload, dict):
                yield payload


def iter_local_records(paths: List[str]) -> Iterator[LabeledRecord]:
    for item in paths:
        abs_path = os.path.abspath(item)
        if os.path.isfile(abs_path):
            source = f"local::{os.path.basename(abs_path)}"
            for record in iter_records_from_file(abs_path):
                yield LabeledRecord(source=source, record=record)
            continue
        if os.path.isdir(abs_path):
            for root, _, files in os.walk(abs_path):
                for name in sorted(files):
                    if not name.endswith((".jsonl", ".json")):
                        continue
                    path = os.path.join(root, name)
                    source = f"local::{name}"
                    for record in iter_records_from_file(path):
                        yield LabeledRecord(source=source, record=record)
            continue
        raise FileNotFoundError(f"未找到输入路径: {abs_path}")


def iter_hf_records(spec: HFDatasetSpec, seed: int, buffer_size: int) -> Iterator[LabeledRecord]:
    print(f"[mini-olmo] 加载阶段二数据源: {spec.dataset_name}")
    load_kwargs = {
        "path": spec.dataset_name,
        "split": spec.split,
        "streaming": spec.streaming,
    }
    if spec.config_name is not None:
        load_kwargs["name"] = spec.config_name

    try:
        dataset = load_dataset(**load_kwargs)
    except Exception:
        if spec.streaming:
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["streaming"] = False
            dataset = load_dataset(**fallback_kwargs)
        else:
            raise

    if hasattr(dataset, "shuffle"):
        if spec.streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            dataset = dataset.shuffle(seed=seed)

    count = 0
    for row in dataset:
        if not isinstance(row, dict):
            continue
        yield LabeledRecord(source=spec.source_name, record=row)
        count += 1
        if spec.max_records > 0 and count >= spec.max_records:
            break


def iter_all_records(args: argparse.Namespace) -> Iterator[LabeledRecord]:
    if not args.no_seed_set:
        for record in seed_examples():
            yield LabeledRecord(source="seed_examples", record=record)

    for spec in recipe_specs(args):
        yield from iter_hf_records(spec, seed=args.seed, buffer_size=args.streaming_buffer_size)

    if args.input_path:
        yield from iter_local_records(args.input_path)


def main() -> None:
    args = parse_args()
    if args.validation_ratio + args.test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio 必须小于 1.0")

    out_dir = output_dir_for(args.dataset_name)
    output_paths = {
        split: os.path.join(out_dir, f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    writers = {
        split: open(path, "w", encoding="utf-8")
        for split, path in output_paths.items()
    }

    raw_counts = Counter()
    kept_by_source = Counter()
    split_counts = Counter()
    dropped_by_reason = Counter()
    seen = set()

    try:
        for index, labeled in enumerate(iter_all_records(args), start=1):
            raw_counts[labeled.source] += 1
            normalized = normalize_record(labeled.record, args)
            if normalized is None:
                dropped_by_reason["filtered_or_unrecognized"] += 1
                continue

            dedupe_key = json.dumps(normalized["messages"], ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen:
                dropped_by_reason["duplicate"] += 1
                continue
            seen.add(dedupe_key)

            split = choose_split(
                key=dedupe_key,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
            )
            output_record = {
                "messages": normalized["messages"],
                "source": labeled.source,
            }
            writers[split].write(json.dumps(output_record, ensure_ascii=False) + "\n")
            kept_by_source[labeled.source] += 1
            split_counts[split] += 1

            if index % args.log_interval == 0:
                kept_total = sum(split_counts.values())
                print(
                    f"[mini-olmo] 已扫描 {index} 条原始样本，"
                    f"保留 {kept_total} 条，当前去重后来源数 {len(kept_by_source)}。"
                )
    finally:
        for writer in writers.values():
            writer.close()

    total_kept = sum(split_counts.values())
    if total_kept == 0:
        raise ValueError("没有得到任何可用的阶段二 SFT 样本。请检查数据源、过滤参数或网络状态。")

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "recipe": args.recipe,
                "splits": dict(split_counts),
                "raw_counts": dict(raw_counts),
                "kept_by_source": dict(kept_by_source),
                "dropped_by_reason": dict(dropped_by_reason),
                "input_paths": args.input_path,
                "used_seed_set": not args.no_seed_set,
                "min_cjk_ratio": args.min_cjk_ratio,
                "max_turns": args.max_turns,
                "max_message_chars": args.max_message_chars,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[mini-olmo] 第二阶段中文 SFT 数据准备完成：")
    for split, path in output_paths.items():
        print(f"  - {split}: {path} ({split_counts[split]} 条)")
    print("  - 来源统计：")
    for source, count in kept_by_source.most_common():
        print(f"    * {source}: {count}")
    print(f"  - stats: {stats_path}")


if __name__ == "__main__":
    main()
