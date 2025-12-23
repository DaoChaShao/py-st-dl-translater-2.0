#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/20 17:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_attn.py
# @Desc     :   

from abc import ABC, abstractmethod
from torch import nn, Tensor, bmm


class BaseAttn(ABC, nn.Module):
    """ Base attention mechanism for sequence models """

    def __init__(self, enc_hn_dims: int, dec_hn_dims: int) -> None:
        """ Initialize the base attention mechanism
        :param enc_hn_dims: hidden dimension of encoder outputs (keys / values)
        :param dec_hn_dims: hidden dimension of decoder hidden state (query)
        """
        super().__init__()
        self._enc_hn_dims: int = enc_hn_dims
        self._dec_hn_dims: int = dec_hn_dims

    @abstractmethod
    def score(self, query: Tensor, memory: Tensor) -> Tensor:
        """ Compute alignment scores
        :param query: decoder hidden at current time step [B, D_dec]
        :param memory: encoder outputs (key/value) [B, S, D_enc]
        :return scores: alignment scores [B, S]
        """
        pass

    def forward(self,
                query: Tensor,
                memory: Tensor,
                mask: Tensor | None = None,
                ) -> tuple[Tensor, Tensor]:
        """ Forward pass for attention mechanism
        :param query: decoder hidden at current time step [B, D_dec]
        :param memory: encoder outputs (key/value) [B, S, D_enc]
        :param mask: padding mask, True means "ignore" [B, S]
        :return:
        - context: [B, D_enc]
        - weights: [B, S]
        """
        # Sanity checks (lightweight, non-restrictive)
        assert query.dim() == 2, "query must be [B, D_dec]"
        assert memory.dim() == 3, "memory must be [B, S, D_enc]"
        assert memory.size(0) == query.size(0), "batch size mismatch"

        # Alignment scores
        scores: Tensor = self.score(query, memory)  # [B, S]

        # Masking (before softmax)
        if mask is not None:
            # mask: True = ignore
            scores = scores.masked_fill(mask, float("-inf"))

        # Attention weights
        weights: Tensor = nn.functional.softmax(scores, dim=1)  # [B, S]

        # Set context vector
        # weights: [B, S] -> [B, 1, S]
        # memory:  [B, S, D_enc]
        # context: [B, D_enc]
        context: Tensor = bmm(weights.unsqueeze(1), memory).squeeze(1)

        return context, weights

    @property
    def enc_hn_dims(self) -> int:
        return self._enc_hn_dims

    @property
    def dec_hn_dims(self) -> int:
        return self._dec_hn_dims


if __name__ == "__main__":
    pass
