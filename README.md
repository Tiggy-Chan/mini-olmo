# mini-OLMo V1

中文优先的小参数语言模型实验仓库，目标是在单张 `RTX 4060 Laptop 8GB` 上跑通：

`中文语料准备 -> 中文 tokenizer -> 中文预训练 -> 中文 SFT -> 中文 chat`

当前仓库已经不再维护旧的英文 `Wikitext / corpus_v2` 路线，默认入口全部对齐到“中文优先 V1”。

## V1 范围

- 以中文为主，不追求多语言平衡
- 先做中文 base model，再考虑后续中文 SFT
- 优先保证数据、tokenizer、训练入口和推理入口一致
- 在 8GB 显存约束下，优先高频迭代和结果可解释性

当前已实现：

- 中文语料准备脚本
- 中文 tokenizer 训练脚本
- 预训练脚本支持自定义 tokenizer 与词表大小
- 中文 SFT 数据准备脚本
- 中文 SFT 训练脚本
- 推理脚本支持指定中文 tokenizer
- 中文 chat 入口
- 中文 V1 开发方案文档

当前未实现：

- 更系统的中文评测基准
- 更系统的多轮 chat 评测与拒答对齐
- 更强的阶段二数据质量打分

## 目录结构

```text
├── docs/
│   ├── CHINESE_V1_PLAN.md
│   └── PROJECT_ROADMAP.md
├── scripts/
│   ├── prepare_corpus_zh_v1.py
│   ├── train_tokenizer_zh.py
│   ├── pretrain.py
│   ├── prepare_sft_zh.py
│   ├── sft.py
│   ├── generate.py
│   └── chat.py
├── src/mini_olmo/
│   ├── data/
│   └── models/
├── tokenizer/
└── data/
```

## 环境安装

```bash
pip install -r requirements.txt
```

建议：

- Python `3.10+`
- GPU 训练使用与你 CUDA 版本匹配的 PyTorch
- 仓库根目录直接运行脚本即可，不需要手工配置 `PYTHONPATH`

## 快速开始

### 1. 准备中文语料

```bash
conda run -n mini-olmo python scripts/prepare_corpus_zh_v1.py \
  --corpus-name zh_corpus_v1 \
  --include-fineweb \
  --fineweb-data-dir 4_5 \
  --max-wikipedia-docs 100000 \
  --max-fineweb-docs 80000 \
  --extra-text-dir /path/to/your/chinese_texts
```

说明：

- 默认会抽样中文 Wikipedia
- `--include-fineweb` 会加入中文 FineWeb Edu 高分桶抽样
- `--fineweb-data-dir 4_5` 表示优先使用高质量 `4_5` 分桶
- `--extra-text-dir` 可以多次传入本地中文文本目录
- 输出落到 `data/raw/zh_corpus_v1/{train,validation,test}.txt`

主力扩展版 `20GB` 纯顶级语料构建：

```bash
conda run -n mini-olmo python scripts/prepare_corpus_zh_v1.py \
  --corpus-name zh_corpus_v3_elite_20gb \
  --target-size-gb 20 \
  --dedupe-backend sqlite \
  --include-fineweb \
  --fineweb-data-dir 4_5 \
  --max-wikipedia-docs 3000000 \
  --max-fineweb-docs 6000000 \
  --min-chars 100 \
  --max-chars 3000 \
  --min-cjk-ratio 0.75 \
  --log-interval 50000
```

说明：

- 这条命令面向 `5GB-20GB+` 的中文预训练扩展语料
- 会在 `data/raw/zh_corpus_v3_elite_20gb/` 下落盘，并自动在达到 `20GiB` 后停止
- `sqlite` 去重适合大语料，避免把所有摘要哈希放进内存
- 这里默认只使用 `Wikipedia + FineWeb Edu Chinese 4_5`
- 不再默认使用 `Cosmopedia`
- 不再默认使用 `FineWeb 3_4 / 2_3`
- 如果这套仍然达不到你要的最终质量标准，下一步应该补充你自己筛过的中文精选文本，而不是放宽桶位
- 如果看到 `HF Hub` 未认证提示，需要先登录或设置 `HF_TOKEN`
- PowerShell 临时设置方式：`$env:HF_TOKEN="你的_token"`
- 断点续跑时加 `--resume`，脚本会追加写入并重建已有语料的去重状态
- 远程流式数据源遇到瞬时网络错误时，脚本会自动重试；可用 `--source-retries` 和 `--source-retry-delay-seconds` 调整
- 如果下载链路偏慢，可以先试 `--streaming-buffer-size 1000` 或 `2000`
- 如果更在意吞吐而不是随机混样，可以加 `--disable-shuffle`
- 如果磁盘去重成为瓶颈，可以把 `--sqlite-commit-interval` 提高到 `20000` 或更大

### 2. 训练中文 tokenizer

快速验证版：

```bash
conda run -n mini-olmo python scripts/train_tokenizer_zh.py \
  --corpus-name zh_corpus_v1 \
  --vocab-size 16000 \
  --output-path tokenizer/tokenizer_zh_v1.json
```

说明：

- `zh_corpus_v1` 仍然是默认的小规模 tokenizer / smoke 语料
- 默认词表大小是 `16k`
- 输出路径默认是 `tokenizer/tokenizer_zh_v1.json`
- 脚本会打印几条中文 prompt 的编码预览
- 当前不建议直接拿 `20GiB` 级主语料全量训练 tokenizer，应该先从主语料中抽一个代表性子集再训

### 3. 预训练中文 base model

快速验证：

```bash
conda run -n mini-olmo python scripts/pretrain.py \
  --model-size v2-cn \
  --corpus-name zh_corpus_v1 \
  --tokenizer-path tokenizer/tokenizer_zh_v1.json \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --total-steps 2000
```

主力训练：

```bash
conda run -n mini-olmo python scripts/pretrain.py \
  --model-size v3-cn \
  --corpus-name zh_corpus_v3_elite_20gb \
  --tokenizer-path tokenizer/tokenizer_zh_v1.json \
  --batch-size 2 \
  --grad-accum-steps 12 \
  --total-steps 80000 \
  --lr 3e-4 \
  --warmup-steps 2000
```

关键点：

- `pretrain.py` 会根据 tokenizer 的真实词表大小自动构建模型
- 缓存 token ids 时会带上 tokenizer 名称，避免不同 tokenizer 共享旧缓存
- `v1-cn / v2-cn / v3-cn` 目前对应同一套模型档位，只是对外统一成中文 V1 命名
- 当前大语料准备脚本支持 `--resume` 断点续跑；继续时请保持相同的数据筛选参数

### 4. 中文生成

```bash
conda run -n mini-olmo python scripts/generate.py \
  --ckpt-path checkpoints/step_80000.pt \
  --tokenizer-path tokenizer/tokenizer_zh_v1.json \
  --prompt "你好，请介绍一下你自己。" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 40
```

### 5. 准备第二阶段中文聊天数据

```bash
conda run -n mini-olmo python scripts/prepare_sft_zh.py \
  --dataset-name zh_sft_stage2_v1 \
  --recipe chat_v1 \
  --input-path /path/to/your/high_quality_chat_data
```

说明：

- 默认会加入一小批内置中文种子样本，便于流程稳定启动
- `--recipe chat_v1` 会直接抽样公开中文指令数据，当前默认配方是：
  - `BelleGroup/train_1M_CN`
  - `FreedomIntelligence/alpaca-gpt4-chinese`
- 支持 `messages`、`instruction/output`、`question/answer`、`prompt/response`、`conversations`、带 `history` 的指令格式
- 会做中文占比、长度、轮数、重复样本和常见 AI boilerplate 过滤
- 输出落到 `data/sft/zh_sft_stage2_v1/{train,validation,test}.jsonl`

### 6. 在中文 base checkpoint 上做 SFT

```bash
conda run -n mini-olmo python scripts/sft.py \
  --base-ckpt-path checkpoints/step_80000.pt \
  --tokenizer-path tokenizer/tokenizer_zh_v1.json \
  --dataset-name zh_sft_stage2_v1 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --epochs 1
```

### 7. 使用 chat 模板做中文对话

```bash
conda run -n mini-olmo python scripts/chat.py \
  --ckpt-path checkpoints_sft/sft_step_100.pt \
  --tokenizer-path tokenizer/tokenizer_zh_v1.json \
  --user-prompt "你好，请简单介绍一下你自己。"
```

## 模型档位

| 档位 | 命令参数 | 典型规模 | 用途 |
|------|----------|----------|------|
| 小型 | `v1-cn` | ~20M-30M | 冒烟测试、快速排错 |
| 中型 | `v2-cn` | ~50M-60M | 中文 V1 快速验证 |
| 主力 | `v3-cn` | ~90M-110M | 中文 V1 主训练 |

说明：

- 实际参数量会随 tokenizer 词表大小变化
- 当前默认序列长度是 `512`
- 默认建议中文语料占比 `95%+`
- 阶段二默认目标是先做出“回答清楚、口吻像助手”的中文 SFT 版，再继续补更复杂的多轮能力

## 当前设计原则

- 默认目标不是“智能涌现”，而是“基本清晰的中文生成与问答”
- 先把中文 tokenizer 和中文 base model 训顺，再进入 SFT
- 优先少量高质量中文数据，而不是一开始追求超大规模脏数据
- 先保证命令入口、默认参数、文档叙事完全一致

## 文档

- 开发方案见 [docs/CHINESE_V1_PLAN.md](docs/CHINESE_V1_PLAN.md)
- 简版路线图见 [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)

## 当前状态

- 仓库基线已经对齐到中文 V1
- 旧英文实验路线和参考仓库内容已移出主流程
- 中文 base model、SFT 和 chat 三段入口已经具备

## 许可证

本项目代码采用 MIT 许可证。
