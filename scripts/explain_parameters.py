"""详细解释 mini-OLMo 中文 V1 模型的参数量计算。

不需要安装 PyTorch，纯数学计算。
"""

DEFAULT_VOCAB_SIZE = 16_000


def explain_parameters():
    """用通俗的语言解释什么是模型参数。"""
    
    print("\n" + "="*70)
    print("什么是模型参数？")
    print("="*70)
    print("""
神经网络就像一个巨大的数学公式，里面有很多需要"学习"的数字。
这些数字就是【参数（Parameters）】，也叫【权重（Weights）】。

举个简单例子：
  y = w1 * x1 + w2 * x2 + b
  
这里的 w1, w2, b 就是参数（3个参数）。
训练就是不断调整这些数字，让模型的预测越来越准确。

在大语言模型中：
  - 26M = 26 Million = 2600万个参数
  - 100M = 100 Million = 1亿个参数  
  - 200M = 200 Million = 2亿个参数
  - GPT-3 = 175B = 1750亿个参数
""")


def count_v1_parameters():
    """计算 v1 模型的参数量。"""
    
    print("\n" + "="*70)
    print("v1 模型参数计算（~26M）")
    print("="*70)
    
    # 配置（中文 V1 默认示例）
    vocab_size = DEFAULT_VOCAB_SIZE
    max_seq_len = 512
    n_layer = 8
    d_model = 384
    n_head = 6
    d_ff = 1_536
    
    print(f"\n配置:")
    print(f"  - 词表大小: {vocab_size:,}")
    print(f"  - 最大序列长度: {max_seq_len}")
    print(f"  - 层数: {n_layer}")
    print(f"  - 模型维度: {d_model}")
    print(f"  - 注意力头数: {n_head}")
    print(f"  - FFN 维度: {d_ff}")
    
    print("\n" + "-"*70)
    print("【1. Embedding 层】")
    print("-"*70)
    
    # Token Embedding: 每个词需要一个 d_model 维的向量
    token_emb = vocab_size * d_model
    print(f"Token Embedding: {vocab_size:,} 个词 × {d_model} 维 = {token_emb:,} 个参数")
    print(f"  含义: 每个词都有一个 {d_model} 维的向量表示")
    
    # Position Embedding: 每个位置需要一个 d_model 维的向量
    pos_emb = max_seq_len * d_model
    print(f"Position Embedding: {max_seq_len} 个位置 × {d_model} 维 = {pos_emb:,} 个参数")
    print(f"  含义: 每个位置都有一个 {d_model} 维的向量表示")
    
    embedding_total = token_emb + pos_emb
    print(f"\nEmbedding 层总计: {embedding_total:,} 个参数")
    
    print("\n" + "-"*70)
    print("【2. 单个 Transformer Block】")
    print("-"*70)
    
    # LayerNorm: 每个维度有 2 个参数（gamma 和 beta）
    ln1 = d_model * 2
    print(f"\nLayerNorm 1: {d_model} × 2 = {ln1:,} 个参数")
    
    # Multi-head Attention
    print(f"\nMulti-head Attention:")
    # Q, K, V 三个投影矩阵
    qkv = 3 * (d_model * d_model + d_model)
    print(f"  - Q, K, V 投影: 3 × ({d_model} × {d_model} + {d_model}) = {qkv:,}")
    print(f"    含义: 把输入投影成 Query、Key、Value 三个矩阵")
    
    # 输出投影
    out_proj = d_model * d_model + d_model
    print(f"  - 输出投影: {d_model} × {d_model} + {d_model} = {out_proj:,}")
    
    attn_total = qkv + out_proj
    print(f"  - Attention 小计: {attn_total:,}")
    
    # LayerNorm 2
    ln2 = d_model * 2
    print(f"\nLayerNorm 2: {d_model} × 2 = {ln2:,} 个参数")
    
    # MLP (Feed-Forward Network)
    print(f"\nMLP (前馈网络):")
    mlp1 = d_model * d_ff + d_ff
    print(f"  - 第一层: {d_model} × {d_ff} + {d_ff} = {mlp1:,}")
    print(f"    含义: 把 {d_model} 维扩展到 {d_ff} 维")
    
    mlp2 = d_ff * d_model + d_model
    print(f"  - 第二层: {d_ff} × {d_model} + {d_model} = {mlp2:,}")
    print(f"    含义: 把 {d_ff} 维压缩回 {d_model} 维")
    
    mlp_total = mlp1 + mlp2
    print(f"  - MLP 小计: {mlp_total:,}")
    
    block_total = ln1 + attn_total + ln2 + mlp_total
    print(f"\n单个 Block 总计: {block_total:,} 个参数")
    
    print("\n" + "-"*70)
    print("【3. 所有 Transformer Blocks】")
    print("-"*70)
    all_blocks = block_total * n_layer
    print(f"{n_layer} 层 × {block_total:,} = {all_blocks:,} 个参数")
    
    print("\n" + "-"*70)
    print("【4. 最终 LayerNorm】")
    print("-"*70)
    ln_f = d_model * 2
    print(f"{d_model} × 2 = {ln_f:,} 个参数")
    
    print("\n" + "-"*70)
    print("【5. 输出头】")
    print("-"*70)
    print("与 Token Embedding 权重共享，不计入额外参数")
    print("含义: 复用 Token Embedding 的权重，节省参数")
    
    print("\n" + "="*70)
    print("【总计】")
    print("="*70)
    total = embedding_total + all_blocks + ln_f
    print(f"Embeddings:          {embedding_total:>15,}")
    print(f"Transformer Blocks:  {all_blocks:>15,}")
    print(f"Final LayerNorm:     {ln_f:>15,}")
    print("-"*70)
    print(f"总参数量:            {total:>15,}")
    print(f"约:                  {total/1e6:>15.2f}M")
    print("="*70)
    
    return total


def count_all_versions():
    """计算所有版本的参数量。"""
    
    print("\n" + "="*70)
    print("三个版本对比（按中文 V1 默认 16k 词表估算）")
    print("="*70)
    
    configs = {
        "v1": {
            "vocab_size": DEFAULT_VOCAB_SIZE,
            "max_seq_len": 512,
            "n_layer": 8,
            "d_model": 384,
            "n_head": 6,
            "d_ff": 1_536,
        },
        "v2": {
            "vocab_size": DEFAULT_VOCAB_SIZE,
            "max_seq_len": 512,
            "n_layer": 12,
            "d_model": 512,
            "n_head": 8,
            "d_ff": 2_048,
        },
        "v3": {
            "vocab_size": DEFAULT_VOCAB_SIZE,
            "max_seq_len": 512,
            "n_layer": 16,
            "d_model": 640,
            "n_head": 10,
            "d_ff": 2_560,
        },
    }
    
    print(f"\n{'版本':<6} {'层数':<6} {'维度':<6} {'头数':<6} {'FFN':<8} {'参数量':<20}")
    print("-"*70)
    
    for version, cfg in configs.items():
        # 计算参数量
        vocab_size = cfg["vocab_size"]
        max_seq_len = cfg["max_seq_len"]
        n_layer = cfg["n_layer"]
        d_model = cfg["d_model"]
        d_ff = cfg["d_ff"]
        
        # Embeddings
        embedding = vocab_size * d_model + max_seq_len * d_model
        
        # Single block
        ln1 = d_model * 2
        attn = 3 * (d_model * d_model + d_model) + (d_model * d_model + d_model)
        ln2 = d_model * 2
        mlp = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
        block = ln1 + attn + ln2 + mlp
        
        # All blocks
        all_blocks = block * n_layer
        
        # Final LN
        ln_f = d_model * 2
        
        # Total
        total = embedding + all_blocks + ln_f
        
        print(f"{version:<6} {n_layer:<6} {d_model:<6} {cfg['n_head']:<6} {d_ff:<8} {total:>12,} ({total/1e6:>6.1f}M)")
    
    print("="*70)


def explain_memory():
    """解释参数量与内存占用的关系。"""
    
    print("\n" + "="*70)
    print("参数量与内存占用")
    print("="*70)
    print("""
1. 模型权重存储:
   - 每个参数通常用 float32 存储（4 字节）
   - 100M 参数 = 100,000,000 × 4 字节 = 400MB
   
2. 训练时的显存占用（约 4-6 倍）:
   - 模型权重: 400MB
   - 梯度: 400MB（每个参数都有对应的梯度）
   - 优化器状态: 800MB（AdamW 需要存储动量和方差）
   - 激活值: 几百 MB（前向传播的中间结果）
   - 总计: 约 2-3GB
   
3. 推理时的显存占用（约 1-2 倍）:
   - 只需要模型权重和激活值
   - 100M 参数约需 500MB-1GB 显存

4. 为什么 v3（100M）在 8GB 显存上能跑？
   - 使用混合精度训练（AMP）: float16 只需 2 字节
   - 梯度累积: 减小 batch size，分多次累积梯度
   - 实际占用约 6-7GB，留有余量
""")


def main():
    explain_parameters()
    count_v1_parameters()
    count_all_versions()
    explain_memory()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
📊 参数量的意义:
  - 参数越多，模型容量越大，理论上能力越强
  - 但也需要更多数据、更长训练时间、更多显存
  
🎯 本项目的选择:
  - v1-cn: 冒烟测试和快速验证
  - v2-cn: 中文 V1 快速验证
  - v3 (100M): 在 8GB 显存限制下的最优选择
  
🚀 对比其他模型:
  - mini-OLMo v3: 100M (0.1B)
  - GPT-2: 117M / 345M / 762M / 1.5B
  - GPT-3: 175B
  - LLaMA: 7B / 13B / 33B / 65B
  - GPT-4: 估计 1.7T (1700B)
  
💡 200M 参数的含义:
  - 就是 2 亿个需要学习的数字
  - 约占用 800MB 存储空间（float32）
  - 训练时需要约 3-4GB 显存
  - 在 8GB 显存上可以训练（需要优化）
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
