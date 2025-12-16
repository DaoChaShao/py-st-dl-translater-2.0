#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 14:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   attention_multi.py
# @Desc     :   

from math import sqrt
from torch import nn, Tensor, einsum


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention mechanism (Transformer style) """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """ Initialize the multi-head attention mechanism
        :param hidden_size: hidden state dimension
        :param num_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

        # Linear projections for Q, K, V
        self._query_proj = nn.Linear(hidden_size, hidden_size)
        self._key_proj = nn.Linear(hidden_size, hidden_size)
        self._value_proj = nn.Linear(hidden_size, hidden_size)

        self._output_proj = nn.Linear(hidden_size, hidden_size)
        self._dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """ Forward pass for multi-head attention
        :param query: query tensor [batch_size, seq_len, hidden_size]
        :param key: key tensor [batch_size, seq_len, hidden_size]
        :param value: value tensor [batch_size, seq_len, hidden_size]
        :param mask: attention mask [batch_size, seq_len, seq_len]
        :return: attention output, attention weights
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self._query_proj(query)
        K = self._key_proj(key)
        V = self._value_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self._num_heads, self._head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = einsum("bhid,bhjd->bhij", Q, K) / sqrt(self._head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self._dropout(attn_weights)

        # Apply attention to values
        context = einsum("bhij,bhjd->bhid", attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self._hidden_size)

        # Final linear projection
        output = self._output_proj(context)

        return output, attn_weights


if __name__ == "__main__":
    pass
