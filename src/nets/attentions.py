#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/20 13:46
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   attentions.py
# @Desc     :   

from math import sqrt
from torch import (Tensor, bmm,
                   nn, tanh,
                   rand, tensor, bool as torch_bool)
from typing import override

from src.nets.base_attn import BaseAttn


class AdditiveAttention(BaseAttn):
    """ Bahdanau (Additive) Attention """

    def __init__(self, enc_hn_dims: int, dec_hn_dims: int, attn_hidden: int | None = None) -> None:
        """ Initialize the additive attention mechanism
        :param enc_hn_dims: encoder hidden size
        :param dec_hn_dims: decoder hidden size
        :param attn_hidden: intermediate attention hidden size, default min(enc, dec)
        """
        super().__init__(enc_hn_dims, dec_hn_dims)
        self._attn_hidden = attn_hidden or min(enc_hn_dims, dec_hn_dims)

        # Linear layers for additive attention
        self._enc_key = nn.Linear(enc_hn_dims, self._attn_hidden, bias=False)
        self._dec_query = nn.Linear(dec_hn_dims, self._attn_hidden, bias=False)
        self._value = nn.Linear(self._attn_hidden, 1, bias=False)

    def score(self, query: Tensor, memory: Tensor, mask: Tensor | None = None) -> Tensor:
        """ Compute additive attention scores
        :param query: decoder hidden at current time step [B, D_dec]
        :param memory: encoder outputs (key/value) [B, S, D_enc]
        :param mask: attention mask [B, S, S]
        :return scores: alignment scores [B, S]
        """
        # memory: [B, S, D_enc] -> [B, S, attn_hidden]
        M: Tensor = self._enc_key(memory)

        # query: [B, D_dec] -> [B, 1, attn_hidden]
        Q: Tensor = self._dec_query(query).unsqueeze(1)

        # Additive combination + tanh -> [B, S, attn_hidden]
        combined = tanh(M + Q)
        # Project to scalar score -> [B, S, 1] -> squeeze -> [B, S]
        scores = self._value(combined).squeeze(2)

        return scores


class DotProductAttention(BaseAttn):
    """ Dot-Product Attention Mechanism """

    def __init__(self, enc_hn_dims: int, dec_hn_dims: int) -> None:
        """ Initialize the dot-product attention mechanism
        :param enc_hn_dims: hidden state dimension of the encoder
        :param dec_hn_dims: hidden state dimension of the decoder
        """
        super().__init__(
            enc_hn_dims=enc_hn_dims,
            dec_hn_dims=dec_hn_dims
        )
        assert enc_hn_dims == dec_hn_dims, "DotAttention requires encoder_dims == decoder_dims"

    @override
    def score(self, query: Tensor, memory: Tensor) -> Tensor:
        """ Compute alignment scores using dot product
        :param query:   [B, D_dec]
        :param memory:  [B, S, D_enc]
        :return: scores [B, S]
        """
        # Dot product between query and memory
        # query:  [B, D_dec] -> [B, 1, D_dec]
        # memory: [B, S, D_enc] -> [B, S, D_dec]
        # scores: bmm -> [B, 1, S] -> squeeze -> [B, S]
        scores: Tensor = bmm(query.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)

        return scores


class ScaledDotProductAttention(BaseAttn):
    """ Scaled Dot-Product Attention """

    def __init__(self, enc_hn_dims: int, dec_hn_dims: int) -> None:
        super().__init__(enc_hn_dims, dec_hn_dims)
        assert enc_hn_dims == dec_hn_dims, "ScaledDotProductAttention requires enc_hn_dims == dec_hn_dims"
        self._scale = sqrt(enc_hn_dims)

    def score(self, query: Tensor, memory: Tensor) -> Tensor:
        """ Compute scaled dot-product attention scores
        :param query: [B, D_dec]
        :param memory: [B, S, D_enc]
        :return: scores: [B, S]
        """
        # query: [B, D] -> [B, 1, D]
        # memory: [B, S, D]
        # bmm -> [B, 1, S] -> squeeze -> [B, S]
        scores = bmm(query.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)
        # Scaled by sqrt(d)
        scores = scores / self._scale
        return scores


if __name__ == "__main__":
    batch_size = 2
    src_len = 5
    hidden_dim = 16

    curr_dec_hn: Tensor = rand(batch_size, hidden_dim)  # decoder hidden t
    enc_outs: Tensor = rand(batch_size, src_len, hidden_dim)  # encoder outputs
    mask: Tensor = tensor([[0, 0, 1, 0, 1], [0, 0, 0, 1, 1]], dtype=torch_bool)

    # Additive (Bahdanau) Attention
    attn_bahdanau = AdditiveAttention(enc_hn_dims=hidden_dim, dec_hn_dims=hidden_dim)
    context, weights = attn_bahdanau(curr_dec_hn, enc_outs, mask)
    print("bahdanau context:", context.shape)  # [B, D]
    print("bahdanau weights:", weights.shape)  # [B, S]

    # Dot-Product Attention
    attn_dot = DotProductAttention(enc_hn_dims=hidden_dim, dec_hn_dims=hidden_dim)
    context, weights = attn_dot(curr_dec_hn, enc_outs, mask)
    print("dot context:", context.shape)  # [B, D]
    print("dot weights:", weights.shape)  # [B, S]

    # Scaled Dot-Product Attention
    attn_scaled = ScaledDotProductAttention(enc_hn_dims=hidden_dim, dec_hn_dims=hidden_dim)
    context, weights = attn_scaled(curr_dec_hn, enc_outs, mask)
    print("scaled context:", context.shape)  # [B, D]
    print("scaled weights:", weights.shape)  # [B, S]
