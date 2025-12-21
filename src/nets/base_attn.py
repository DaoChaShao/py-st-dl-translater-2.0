#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/20 17:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_attn.py
# @Desc     :   

from abc import ABC, abstractmethod
from torch import nn, Tensor


class BaseAttn(ABC, nn.Module):
    """ Base attention mechanism for sequence models """

    def __init__(self, enc_hn_size: int, dec_hn_size: int) -> None:
        """ Initialize the base attention mechanism
        :param enc_hn_size: hidden state dimension of the encoder
        :param dec_hn_size: hidden state dimension of the decoder
        """
        super().__init__()
        self._enc_size: int = enc_hn_size
        self._dec_size: int = dec_hn_size

        assert self._enc_size == self._dec_size, "Encoder and decoder hidden sizes must match in BaseAttn."

    @abstractmethod
    def forward(self, dec_hn: Tensor, enc_outs: Tensor, pad_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """ Forward pass for attention
        :param dec_hn: decoder hidden state [1, batch_size, hidden_size]
        :param enc_outs: encoder outputs [src_len, batch_size, hidden_size]
        :param pad_mask: padding mask [src_len, batch_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        pass

    @property
    def enc_size(self) -> int:
        return self._enc_size

    @property
    def dec_size(self) -> int:
        return self._dec_size


if __name__ == "__main__":
    pass
