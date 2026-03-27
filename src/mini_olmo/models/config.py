from dataclasses import dataclass


@dataclass
class MiniOlmoConfig:
    """mini-OLMo 模型的超参数配置。

    当前仓库基线是中文优先 V1。
    默认配置用于单张 8GB GPU 上的小模型冒烟测试，
    更大的实验档位由训练脚本在运行时覆盖。
    """

    # vocab & 位置
    vocab_size: int = 32_000
    max_seq_len: int = 512

    # Transformer 结构
    n_layer: int = 8
    d_model: int = 384
    n_head: int = 6
    d_ff: int = 1_536  # 通常约为 4 * d_model

    # dropout
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # 数值精度/初始化等可以后续按需补充
