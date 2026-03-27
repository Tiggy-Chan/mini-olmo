"""在中文 base checkpoint 上执行第一版中文 SFT。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mini_olmo.models.config import MiniOlmoConfig
from mini_olmo.models.transformer import MiniOlmoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for mini-OLMo Chinese V1")
    parser.add_argument("--base-ckpt-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer_zh_v1.json")
    parser.add_argument("--dataset-name", type=str, default="zh_sft_v1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="checkpoints_sft")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    abs_path = resolve_project_path(tokenizer_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"未找到 tokenizer 文件: {abs_path}")
    return Tokenizer.from_file(abs_path)


def load_base_checkpoint(ckpt_path: str, device: torch.device) -> tuple[MiniOlmoModel, MiniOlmoConfig]:
    ckpt = torch.load(resolve_project_path(ckpt_path), map_location=device, weights_only=False)
    config = MiniOlmoConfig(**ckpt["config"])
    model = MiniOlmoModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config


def update_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        lr = base_lr * float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr = 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def render_message(role: str, content: str) -> str:
    prefix_map = {
        "system": "系统：",
        "user": "用户：",
        "assistant": "助手：",
    }
    return f"{prefix_map[role]}{content}\n"


def build_sft_example(
    tokenizer: Tokenizer,
    messages: List[Dict[str, str]],
    max_seq_len: int,
) -> Dict[str, List[int]] | None:
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    input_ids: List[int] = []
    labels: List[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    for message in messages:
        role = message["role"]
        content = message["content"]
        segment_ids = tokenizer.encode(render_message(role, content)).ids
        if not segment_ids:
            continue
        input_ids.extend(segment_ids)
        if role == "assistant":
            labels.extend(segment_ids)
        else:
            labels.extend([-100] * len(segment_ids))

    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id if messages and messages[-1]["role"] == "assistant" else -100)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]

    if not any(label != -100 for label in labels):
        return None

    if pad_id is None:
        raise ValueError("tokenizer 缺少 <pad> token，无法执行 SFT。")

    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer, max_seq_len: int) -> None:
        self.examples: List[Dict[str, List[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages")
                if not isinstance(messages, list):
                    continue
                example = build_sft_example(tokenizer, messages, max_seq_len=max_seq_len)
                if example is not None:
                    self.examples.append(example)

        if not self.examples:
            raise ValueError(f"未从 {path} 构造出任何有效 SFT 样本。")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:  # type: ignore[override]
        return self.examples[idx]


class SFTCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        for item in batch:
            ids = item["input_ids"]
            lbs = item["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            labels.append(lbs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_data_path(dataset_name: str, split: str) -> str:
    return os.path.join(PROJECT_ROOT, "data", "sft", dataset_name, f"{split}.jsonl")


def build_dataloaders(
    dataset_name: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("tokenizer 缺少 <pad> token。")
    collator = SFTCollator(pad_id=pad_id)
    train_dataset = SFTDataset(sft_data_path(dataset_name, "train"), tokenizer, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_path = sft_data_path(dataset_name, "validation")
    val_loader = None
    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        val_dataset = SFTDataset(val_path, tokenizer, max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader


def evaluate(model: MiniOlmoModel, dataloader: DataLoader | None, device: torch.device) -> Dict[str, float]:
    if dataloader is None:
        return {"loss": float("nan"), "ppl": float("nan")}
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            if loss is None:
                continue
            active_tokens = int((labels != -100).sum().item())
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens
            if total_tokens >= 4096:
                break
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "ppl": math.exp(avg_loss)}


def save_checkpoint(
    output_dir: str,
    step: int,
    model: MiniOlmoModel,
    optimizer: torch.optim.Optimizer,
    config: MiniOlmoConfig,
    base_ckpt_path: str,
    dataset_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"sft_step_{step}.pt")
    torch.save(
        {
            "step": step,
            "base_ckpt_path": base_ckpt_path,
            "dataset_name": dataset_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
        },
        ckpt_path,
    )
    print(f"[mini-olmo] 已保存 SFT checkpoint 到 {ckpt_path}")


def optimizer_step(
    model: MiniOlmoModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    global_step: int,
    total_steps: int,
    running_loss: float,
    start_time: float,
    val_loader: DataLoader | None,
    config: MiniOlmoConfig,
) -> tuple[int, float, float]:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    global_step += 1
    lr = update_lr(optimizer, args.lr, global_step, total_steps, args.warmup_steps)

    if global_step % args.log_interval == 0:
        elapsed = time.time() - start_time
        avg_loss = running_loss / args.log_interval
        print(
            f"[sft step {global_step}/{total_steps}] "
            f"loss={avg_loss:.4f}, lr={lr:.6f}, time/batch={elapsed / args.log_interval:.3f}s"
        )
        running_loss = 0.0
        start_time = time.time()

    if global_step % args.eval_interval == 0 and val_loader is not None:
        metrics = evaluate(model, val_loader, next(model.parameters()).device)
        print(
            f"[mini-olmo] SFT eval at step {global_step}: "
            f"loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.2f}"
        )

    if global_step % args.save_interval == 0:
        save_checkpoint(
            output_dir=args.output_dir,
            step=global_step,
            model=model,
            optimizer=optimizer,
            config=config,
            base_ckpt_path=args.base_ckpt_path,
            dataset_name=args.dataset_name,
        )

    return global_step, running_loss, start_time


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mini-olmo] 使用设备: {device}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    model, config = load_base_checkpoint(args.base_ckpt_path, device)
    print(f"[mini-olmo] base checkpoint: {resolve_project_path(args.base_ckpt_path)}")
    print(f"[mini-olmo] tokenizer: {resolve_project_path(args.tokenizer_path)}")

    train_loader, val_loader = build_dataloaders(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    print(f"[mini-olmo] 预计优化步数: {total_steps}")

    global_step = 0
    accum_in_window = 0
    running_loss = 0.0
    start_time = time.time()
    model.train()

    for epoch in range(args.epochs):
        print(f"[mini-olmo] 开始第 {epoch + 1}/{args.epochs} 轮 SFT")
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=device.type == "cuda"):
                out = model(input_ids, labels=labels)
                loss = out["loss"]
                if loss is None:
                    continue
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()
            accum_in_window += 1

            if accum_in_window == args.grad_accum_steps:
                global_step, running_loss, start_time = optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    global_step=global_step,
                    total_steps=total_steps,
                    running_loss=running_loss,
                    start_time=start_time,
                    val_loader=val_loader,
                    config=config,
                )
                accum_in_window = 0

        if accum_in_window > 0:
            global_step, running_loss, start_time = optimizer_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                global_step=global_step,
                total_steps=total_steps,
                running_loss=running_loss,
                start_time=start_time,
                val_loader=val_loader,
                config=config,
            )
            accum_in_window = 0

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            print(
                f"[mini-olmo] 完成 epoch {epoch + 1}: "
                f"val_loss={metrics['loss']:.4f}, val_ppl={metrics['ppl']:.2f}"
            )
        else:
            print(f"[mini-olmo] 完成 epoch {epoch + 1}: 未提供 validation split，跳过评估。")

    save_checkpoint(
        output_dir=args.output_dir,
        step=global_step,
        model=model,
        optimizer=optimizer,
        config=config,
        base_ckpt_path=args.base_ckpt_path,
        dataset_name=args.dataset_name,
    )
    print("[mini-olmo] SFT 完成。")


if __name__ == "__main__":
    main()
