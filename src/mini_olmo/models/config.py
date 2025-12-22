from dataclasses import dataclass


@dataclass
class MiniOlmoConfig:
    """mini-OLMo 模型的超参数配置。

    这一版是针对单张 8GB GPU 设计的约 20M 参数模型，
    以 Wikitext 预训练为主要目标。
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
