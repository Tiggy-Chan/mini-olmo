"""详细计算和展示 mini-OLMo 中文 V1 模型的参数量。

这个脚本会逐层分析模型结构，展示每个组件的参数数量。
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_olmo.models.config import MiniOlmoConfig
from mini_olmo.models.transformer import MiniOlmoModel

DEFAULT_VOCAB_SIZE = 16_000


def count_parameters_detailed(config: MiniOlmoConfig) -> None:
    """详细计算模型各部分的参数量。"""
    
    print(f"\n{'='*70}")
    print(f"模型配置: {config.n_layer} 层, {config.d_model} 维, {config.n_head} 头")
    print(f"{'='*70}\n")
    
    # 1. Embedding 层
    print("【1. Embedding 层】")
    token_emb_params = config.vocab_size * config.d_model
    pos_emb_params = config.max_seq_len * config.d_model
    print(f"  Token Embedding: {config.vocab_size:,} × {config.d_model} = {token_emb_params:,}")
    print(f"  Position Embedding: {config.max_seq_len:,} × {config.d_model} = {pos_emb_params:,}")
    embedding_total = token_emb_params + pos_emb_params
    print(f"  小计: {embedding_total:,}\n")
    
    # 2. 单个 Transformer Block
    print("【2. 单个 Transformer Block】")
    
    # LayerNorm 1
    ln1_params = config.d_model * 2  # gamma 和 beta
    print(f"  LayerNorm 1: {config.d_model} × 2 = {ln1_params:,}")
    
    # Multi-head Attention
    # Q, K, V 投影: 3 个 (d_model × d_model) 的权重矩阵 + 偏置
    qkv_params = 3 * (config.d_model * config.d_model + config.d_model)
    # 输出投影: (d_model × d_model) + 偏置
    out_proj_params = config.d_model * config.d_model + config.d_model
    attn_params = qkv_params + out_proj_params
    print(f"  Multi-head Attention:")
    print(f"    - Q, K, V 投影: 3 × ({config.d_model} × {config.d_model} + {config.d_model}) = {qkv_params:,}")
    print(f"    - 输出投影: {config.d_model} × {config.d_model} + {config.d_model} = {out_proj_params:,}")
    print(f"    - 小计: {attn_params:,}")
    
    # LayerNorm 2
    ln2_params = config.d_model * 2
    print(f"  LayerNorm 2: {config.d_model} × 2 = {ln2_params:,}")
    
    # MLP (Feed-Forward Network)
    # 第一层: d_model → d_ff
    mlp1_params = config.d_model * config.d_ff + config.d_ff
    # 第二层: d_ff → d_model
    mlp2_params = config.d_ff * config.d_model + config.d_model
    mlp_params = mlp1_params + mlp2_params
    print(f"  MLP (Feed-Forward):")
    print(f"    - 第一层: {config.d_model} × {config.d_ff} + {config.d_ff} = {mlp1_params:,}")
    print(f"    - 第二层: {config.d_ff} × {config.d_model} + {config.d_model} = {mlp2_params:,}")
    print(f"    - 小计: {mlp_params:,}")
    
    block_params = ln1_params + attn_params + ln2_params + mlp_params
    print(f"  单个 Block 总计: {block_params:,}\n")
    
    # 3. 所有 Transformer Blocks
    all_blocks_params = block_params * config.n_layer
    print(f"【3. 所有 Transformer Blocks】")
    print(f"  {config.n_layer} 层 × {block_params:,} = {all_blocks_params:,}\n")
    
    # 4. 最终 LayerNorm
    ln_f_params = config.d_model * 2
    print(f"【4. 最终 LayerNorm】")
    print(f"  {config.d_model} × 2 = {ln_f_params:,}\n")
    
    # 5. 输出头（与 token embedding 权重共享，不计入额外参数）
    print(f"【5. 输出头 (LM Head)】")
    print(f"  与 Token Embedding 权重共享，不计入额外参数\n")
    
    # 总计
    total_params = embedding_total + all_blocks_params + ln_f_params
    print(f"{'='*70}")
    print(f"【总参数量】")
    print(f"  Embeddings:        {embedding_total:>15,}")
    print(f"  Transformer Blocks: {all_blocks_params:>15,}")
    print(f"  Final LayerNorm:   {ln_f_params:>15,}")
    print(f"  {'-'*70}")
    print(f"  总计:              {total_params:>15,}")
    print(f"  约:                {total_params/1e6:>15.2f}M")
    print(f"{'='*70}\n")
    
    # 实际验证
    model = MiniOlmoModel(config)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"【实际模型参数量（PyTorch 统计）】")
    print(f"  {actual_params:,} ({actual_params/1e6:.2f}M)")
    
    if actual_params == total_params:
        print(f"  ✅ 计算正确！")
    else:
        print(f"  ⚠️  与手动计算有差异: {abs(actual_params - total_params):,}")
    print()


def main():
    print("\n" + "="*70)
    print("mini-OLMo 模型参数量详细分析")
    print("="*70)
    
    # v1 配置
    print("\n\n" + "🔹 " * 20)
    print("v1-cn 模型 (16k 词表示例)")
    print("🔹 " * 20)
    config_v1 = MiniOlmoConfig(vocab_size=DEFAULT_VOCAB_SIZE)
    count_parameters_detailed(config_v1)
    
    # v2 配置
    print("\n\n" + "🔹 " * 20)
    print("v2-cn 模型 (16k 词表示例)")
    print("🔹 " * 20)
    config_v2 = MiniOlmoConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        max_seq_len=512,
        n_layer=12,
        d_model=512,
        n_head=8,
        d_ff=2_048,
        dropout=0.1,
        attention_dropout=0.1,
    )
    count_parameters_detailed(config_v2)
    
    # v3 配置
    print("\n\n" + "🔹 " * 20)
    print("v3-cn 模型 (16k 词表示例)")
    print("🔹 " * 20)
    config_v3 = MiniOlmoConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        max_seq_len=512,
        n_layer=16,
        d_model=640,
        n_head=10,
        d_ff=2_560,
        dropout=0.1,
        attention_dropout=0.1,
    )
    count_parameters_detailed(config_v3)
    
    # 参数量对比
    print("\n" + "="*70)
    print("【三个版本对比】")
    print("="*70)
    print(f"{'版本':<8} {'层数':<6} {'维度':<6} {'头数':<6} {'FFN维度':<8} {'参数量':<15}")
    print("-"*70)
    
    for name, cfg in [("v1-cn", config_v1), ("v2-cn", config_v2), ("v3-cn", config_v3)]:
        model = MiniOlmoModel(cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<8} {cfg.n_layer:<6} {cfg.d_model:<6} {cfg.n_head:<6} {cfg.d_ff:<8} {params:>12,} ({params/1e6:.1f}M)")
    
    print("="*70)
    
    print("\n💡 参数量说明:")
    print("  - 1M = 1 Million = 100万")
    print("  - 26M = 2600万个参数（需要学习的数字）")
    print("  - 每个参数通常用 float32 存储（4 字节）")
    print("  - 所以 100M 参数的模型大约占用 400MB 内存")
    print("  - 训练时还需要存储梯度、优化器状态等，实际显存占用约 4-6 倍\n")


if __name__ == "__main__":
    main()
