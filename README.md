# mini-OLMo

从零实现一个简化版 OLMo，在单卡 8GB GPU 上完整跑通：数据处理 → tokenizer → 模型定义 → 预训练 → 推理 的全流程。

## 模型架构

Decoder-only Transformer（GPT 风格），特点：
- Pre-norm（LayerNorm 在 attention/MLP 之前）
- 可学习位置编码
- 输出头与 token embedding 权重共享
- 支持 v1（26M）、v2（55M）、v3（100M）三种规模

## 目录结构

```
├── src/mini_olmo/       # 核心库（模型、数据管道）
├── scripts/             # 训练、生成、数据准备脚本
├── tokenizer/           # BPE 分词器
├── data/                # 原始和处理后的数据
├── checkpoints/         # 模型检查点
└── docs/                # 项目文档和路线图
```

## 快速开始

### 1. 环境安装

```bash
# Python 3.10+
pip install -r requirements.txt

# PyTorch GPU 版本建议根据 CUDA 版本单独安装
# https://pytorch.org/get-started/locally/
```

### 2. 准备数据

```bash
# Wikitext
python scripts/prepare_wikitext.py

# 扩展语料（Wikitext + Wikipedia）
python scripts/prepare_corpus_v2.py
```

### 3. 训练 tokenizer

```bash
python tokenizer/train_tokenizer.py
```

### 4. 预训练

```bash
# v1 小模型（26M，快速验证）
python scripts/pretrain.py --model-size v1 --total-steps 1000

# v3 完整训练（100M，约 1 天）
python scripts/pretrain.py \
    --model-size v3 \
    --corpus-name corpus_v2 \
    --batch-size 2 \
    --grad-accum-steps 12 \
    --total-steps 80000
```

### 5. 生成文本

```bash
python scripts/generate.py \
    --ckpt-path checkpoints/step_80000.pt \
    --prompt "The history of artificial intelligence" \
    --max-new-tokens 100 \
    --temperature 0.8
```

## 模型配置

| 版本 | 参数量 | 层数 | 维度 | 注意力头 | FFN 维度 |
|------|--------|------|------|----------|----------|
| v1   | ~26M   | 8    | 384  | 6        | 1536     |
| v2   | ~55M   | 12   | 512  | 8        | 2048     |
| v3   | ~100M  | 16   | 640  | 10       | 2560     |

## 硬件要求

- GPU：8GB 显存（如 RTX 4060）
- v3 训练配置：batch_size=2, grad_accum_steps=12, seq_len=512

## 预训练权重

模型权重托管在 Hugging Face Hub：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载权重
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="YOUR_USERNAME/mini-olmo", filename="checkpoints/v3_step_200000.pt")
```

或手动下载：https://huggingface.co/YOUR_USERNAME/mini-olmo

## 项目进展

详见 [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)
