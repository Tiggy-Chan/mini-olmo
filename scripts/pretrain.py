import math
import os
import time
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from mini_olmo.models.config import MiniOlmoConfig
from mini_olmo.models.transformer import MiniOlmoModel
from mini_olmo.data.dataset import create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain mini-OLMo on Wikitext")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-name", type=str, default="wikitext")
    parser.add_argument(
        "--model-size",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        lr = base_lr * float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine decay to 10% of base lr
        lr = 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_checkpoint(
    output_dir: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: MiniOlmoConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
        },
        ckpt_path,
    )
    print(f"[mini-olmo] 已保存 checkpoint 到 {ckpt_path}")


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
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
            # loss 已经是平均在 batch*seq_len 上的 cross-entropy
            num_tokens = input_ids.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            # 为了加快评估，只跑少量 batch
            if total_tokens >= 128 * 512:
                break
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "ppl": ppl}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mini-olmo] 使用设备: {device}")

    if args.model_size == "v1":
        config = MiniOlmoConfig()
    elif args.model_size == "v2":
        config = MiniOlmoConfig(
            vocab_size=32_000,
            max_seq_len=512,
            n_layer=12,
            d_model=512,
            n_head=8,
            d_ff=2_048,
            dropout=0.1,
            attention_dropout=0.1,
        )
    elif args.model_size == "v3":
        config = MiniOlmoConfig(
            vocab_size=32_000,
            max_seq_len=512,
            n_layer=16,
            d_model=640,
            n_head=10,
            d_ff=2_560,
            dropout=0.1,
            attention_dropout=0.1,
        )
    else:
        raise ValueError(f"未知的 model_size: {args.model_size}")

    model = MiniOlmoModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[mini-olmo] 模型参数量: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=device.type == "cuda")

    train_loader = create_dataloader(
        "train",
        config,
        batch_size=args.batch_size,
        shuffle=True,
        corpus_name=args.corpus_name,
    )
    val_loader = create_dataloader(
        "validation",
        config,
        batch_size=args.batch_size,
        shuffle=False,
        corpus_name=args.corpus_name,
    )

    model.train()
    global_step = 0
    step_in_epoch = 0
    running_loss = 0.0
    start_time = time.time()

    while global_step < args.total_steps:
        for batch in train_loader:
            step_in_epoch += 1

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

            if step_in_epoch % args.grad_accum_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                lr = update_lr(optimizer, args.lr, global_step, args.total_steps, args.warmup_steps)

                if global_step % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / args.log_interval
                    print(
                        f"[step {global_step}/{args.total_steps}] "
                        f"loss={avg_loss:.4f}, lr={lr:.6f}, "
                        f"time/batch={elapsed / args.log_interval:.3f}s",
                    )
                    running_loss = 0.0
                    start_time = time.time()

                if global_step % args.eval_interval == 0:
                    metrics = evaluate(model, val_loader, device)
                    print(
                        f"[mini-olmo] Eval at step {global_step}: "
                        f"loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.2f}",
                    )

                if global_step % args.save_interval == 0:
                    save_checkpoint(args.output_dir, global_step, model, optimizer, config)

                if global_step >= args.total_steps:
                    break

        # 走完一个 epoch 后如果还没到 total_steps，就继续 while 循环再跑一轮

    # 训练结束后保存最终模型
    save_checkpoint(args.output_dir, global_step, model, optimizer, config)
    print("[mini-olmo] 训练完成。")


if __name__ == "__main__":
    main()
