from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniOlmoConfig


class MiniOlmoBlock(nn.Module):
    """mini-OLMo 的单层 Transformer block（pre-norm decoder-only）。"""

    def __init__(self, config: MiniOlmoConfig) -> None:
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Self-attention
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)

        # MLP
        residual = x
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + self.dropout(mlp_out)

        return x


class MiniOlmoModel(nn.Module):
    """简化版 OLMo 语言模型（decoder-only Transformer LM）。"""

    def __init__(self, config: MiniOlmoConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([MiniOlmoBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)

        # 输出头与 token embedding 权重共享
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 生成 (seq_len, seq_len) 的下三角 mask，供 MultiheadAttention 使用
        # 注意：nn.MultiheadAttention 期望的 attn_mask 是 (batch*num_heads, tgt_len, src_len)
        # 或 (tgt_len, src_len)。我们这里先构造 (seq_len, seq_len)。
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """前向传播。

        参数：
            input_ids: (batch, seq_len) 的 token id。
            labels: (batch, seq_len) 的标签，如果提供则返回 cross-entropy loss。
        返回：
            dict，包含：
                - logits: (batch, seq_len, vocab_size)
                - loss: 如果提供 labels，则为标量 loss，否则为 None
        """

        bsz, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} 超过了 max_seq_len={self.config.max_seq_len}")

        device = input_ids.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_embedding(input_ids) + self.pos_embedding(pos)

        # 构造 causal mask：形状 (seq_len, seq_len)
        attn_mask = self._build_causal_mask(seq_len, device)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # 将 logits/labels 展平到 (batch * seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}
