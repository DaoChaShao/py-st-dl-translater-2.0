#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/20 13:46
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   attentions.py
# @Desc     :   

from torch import nn, Tensor, bmm, einsum, matmul
from typing import override

from src.nets.base_attn import BaseAttn


class AdditiveAttention(BaseAttn):
    """ Bahdanau Additive Attention Mechanism """

    def __init__(self, enc_hn_size: int, dec_hn_size: int, *, attn_size: int) -> None:
        """ Initialize the additive attention mechanism
        :param enc_hn_size: hidden state dimension of the encoder
        :param dec_hn_size: hidden state dimension of the decoder
        :param attn_size: size of the attention layer
        """
        super().__init__(
            enc_hn_size=enc_hn_size,
            dec_hn_size=dec_hn_size
        )
        self._attn_size: int = attn_size

        # Support key and value from encoder
        self._key: nn.Linear = nn.Linear(self._enc_size, self._attn_size, bias=False)
        self._value: nn.Linear = nn.Linear(self._attn_size, 1, bias=False)
        # Support query from decoder
        self._query: nn.Linear = nn.Linear(self._dec_size, self._attn_size, bias=False)

    @override
    def forward(self, dec_hn: Tensor, enc_outs: Tensor, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """ Forward pass for additive attention
        :param dec_hn: decoder hidden state [1, batch_size, hidden_size]
        :param enc_outs: encoder outputs [src_len, batch_size, hidden_size]
        :param pad_mask: padding mask [src_len, batch_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        if dec_hn.dim() == 3:
            dec_hn = dec_hn.squeeze(0)  # [batch_size, dec_hidden]

        # Project encoder outputs and decoder hidden state
        key = self._key(enc_outs)  # [src_len, batch, attn_size]
        query = self._query(dec_hn).unsqueeze(0)  # [1, batch, attn_size]

        # Calculate energies (additive)
        score = self._value(nn.functional.tanh(key + query)).squeeze(-1)

        # Transpose to [batch_size, src_len]
        score = score.transpose(0, 1)  # [batch, src_len]

        # Apply mask (if provided)
        if pad_mask is not None:
            # batch_size=Ture, otherwise transpose it
            score = score.masked_fill(pad_mask == 1, float("-inf"))

        # Compute attention weights - [batch_size, src_len]
        attn_weights = nn.functional.softmax(score, dim=1)

        # Compute context vector - # [batch_size, hidden_size]
        context_vector = bmm(attn_weights.unsqueeze(1), enc_outs.transpose(0, 1)).squeeze(1)

        return attn_weights, context_vector

    @property
    def attn_size(self) -> int:
        return self._attn_size

    @property
    def key(self) -> nn.Linear:
        return self._key

    @property
    def value(self) -> nn.Linear:
        return self._value

    @property
    def query(self) -> nn.Linear:
        return self._query


class DotProductAttention(BaseAttn):
    """ Dot-Product Attention Mechanism """

    def __init__(self, enc_hn_size: int, dec_hn_size: int) -> None:
        """ Initialize the dot-product attention mechanism
        :param enc_hn_size: hidden state dimension of the encoder
        :param dec_hn_size: hidden state dimension of the decoder
        """
        super().__init__(
            enc_hn_size=enc_hn_size,
            dec_hn_size=dec_hn_size
        )

    @override
    def forward(self, dec_hn: Tensor, enc_outs: Tensor, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """ Forward pass for dot-product attention
        :param dec_hn: decoder hidden state [1, batch_size, hidden_size]
        :param enc_outs: encoder outputs [src_len, batch_size, hidden_size]
        :param pad_mask: padding mask [src_len, batch_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        if dec_hn.dim() == 3:
            dec_hn = dec_hn.squeeze(0)  # [batch_size, dec_hidden]

        # Calculate energies (dot product)
        energies: Tensor = einsum("bh,sbh->bs", dec_hn, enc_outs)  # [batch, src_len]

        # Apply mask (if provided) - [batch, src_len]
        if pad_mask is not None:
            # batch_size=Ture, otherwise transpose it
            energies: Tensor = energies.masked_fill(pad_mask == 1, float("-inf"))

        # Compute attention weights - [batch_size, src_len]
        attn_weights: Tensor = nn.functional.softmax(energies, dim=1)

        # Compute context vector - # [batch_size, hidden_size]
        context: Tensor = bmm(attn_weights.unsqueeze(1), enc_outs.transpose(0, 1)).squeeze(1)

        return attn_weights, context


class ScaledDotProductAttention(BaseAttn):

    def __init__(self, enc_hn_size: int, dec_hn_size: int) -> None:
        """ Initialize the scaled dot-product attention mechanism
        :param enc_hn_size: hidden state dimension of the encoder
        :param dec_hn_size: hidden state dimension of the decoder
        """
        super().__init__(
            enc_hn_size=enc_hn_size,
            dec_hn_size=dec_hn_size
        )

        self._query: nn.Linear = nn.Linear(self._enc_size, self._enc_size, bias=False)
        self._key: nn.Linear = nn.Linear(self._enc_size, self._enc_size, bias=False)
        self._value: nn.Linear = nn.Linear(self._enc_size, self._enc_size, bias=False)

    @override
    def forward(self, dec_hn: Tensor, enc_outs: Tensor, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """ Forward pass for scaled dot-product attention
        :param dec_hn: decoder hidden state [1, batch_size, hidden_size]
        :param enc_outs: encoder outputs [src_len, batch_size, hidden_size]
        :param pad_mask: padding mask [src_len, batch_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        Q: Tensor = self._query(dec_hn.squeeze(0).unsqueeze(1))  # [batch, 1, hidden]
        keys: Tensor = enc_outs.transpose(0, 1)
        K: Tensor = self._key(keys)  # [batch, src_len, hidden]
        V: Tensor = self._value(keys)  # [batch, src_len, hidden]

        energies: Tensor = matmul(Q, K.transpose(-2, -1)) / (self.enc_size ** 0.5)  # [batch, 1, src_len]
        if pad_mask is not None and pad_mask.dim() == 2:
            # batch_size=Ture, otherwise transpose it
            pad_mask = pad_mask.unsqueeze(1)  # [src_len,batch] -> batch,1,src_len
            energies = energies.masked_fill(pad_mask == 1, float("-inf"))

        attn_weights = nn.functional.softmax(energies, dim=-1).squeeze(1)
        context = matmul(attn_weights.unsqueeze(1), V).squeeze(1)

        return attn_weights, context

    @property
    def key(self) -> nn.Linear:
        return self._key

    @property
    def value(self) -> nn.Linear:
        return self._value

    @property
    def query(self) -> nn.Linear:
        return self._query


if __name__ == "__main__":
    pass
