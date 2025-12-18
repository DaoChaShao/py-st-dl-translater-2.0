#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 14:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   attention_single.py
# @Desc     :

from torch import (Tensor, nn, bmm, cat)

from src.configs.cfg_types import AttnScorer
from typing import Literal


class SingleHeadAttention(nn.Module):
    """ Base attention mechanism for sequence models """

    def __init__(self,
                 encoder_hidden_size: int, decoder_hidden_size: int,
                 method: str | AttnScorer | Literal["dot", "general", "concat"] = "dot"
                 ) -> None:
        """ Initialize the attention mechanism
        :param encoder_hidden_size: hidden state dimension
        :param method: attention method ('dot', 'general', 'concat')
        """
        super().__init__()
        self._encoder_hidden_size = encoder_hidden_size
        self._decoder_hidden_size = decoder_hidden_size
        self._method = method

        # Projection layer if dimensions differ
        if encoder_hidden_size != decoder_hidden_size:
            self._projection = nn.Linear(decoder_hidden_size, encoder_hidden_size, bias=False)
        else:
            self._projection = None

        if self._method == "general":
            self._attn = nn.Linear(encoder_hidden_size, encoder_hidden_size, bias=False)
        elif self._method == "concat":
            self._attn = nn.Linear(encoder_hidden_size * 2, encoder_hidden_size, bias=False)
            self._value = nn.Linear(encoder_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """ Forward pass for attention
        :param decoder_hidden: decoder hidden state [1, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden.squeeze(0)

        if self._method == "dot":
            attn_energies = self._dot_score(encoder_outputs, decoder_hidden)
        elif self._method == "general":
            attn_energies = self._general_score(encoder_outputs, decoder_hidden)
        elif self._method == "concat":
            attn_energies = self._concat_score(encoder_outputs, decoder_hidden)
        else:
            raise ValueError(f"Unknown attention method: {self._method}")

        # Normalize energies to weights
        attn_weights = nn.functional.softmax(attn_energies, dim=1)

        # Calculate context vector
        context = bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return attn_weights, context

    def _dot_score(self, encoder_outputs: Tensor, decoder_hidden: Tensor) -> Tensor:
        """ Dot product attention score
        :param encoder_outputs: encoder outputs [batch, src_len, hidden]
        :param decoder_hidden: decoder hidden state [1, batch_size, hidden_size]
        :return: attention energies [batch_size, src_len]
        """
        # print("Dot product attention score called.")
        # encoder_outputs: [batch, src_len, hidden]
        # print(f"dot encoder outpus: {encoder_outputs.shape}")
        # hidden: [1, batch_size, hidden_size]
        # print(f"dot decoder hidden: {decoder_hidden.shape}")

        hidden = decoder_hidden.squeeze(0)  # [batch_size, hidden_size]

        if self._projection is not None:
            hidden = self._projection(hidden)  # [batch_size, encoder_hidden_size]
            # print(f"After projection: {hidden.shape}")

        return bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)

    def _general_score(self, encoder_outputs: Tensor, decoder_hidden: Tensor) -> Tensor:
        """ General attention score with learned linear transformation
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :param decoder_hidden: decoder hidden state [1, batch_size, hidden_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        # print("General attention score called.")
        # hidden: [1, batch_size, hidden_size]
        hidden = decoder_hidden.squeeze(0)  # [batch_size, hidden_size]

        if self._projection is not None:
            hidden = self._projection(hidden)  # [batch_size, encoder_hidden_size]
            # print(f"After projection: {hidden.shape}")

        # Transform encoder outputs
        transformed = self._attn(encoder_outputs)  # [batch_size, src_len, hidden_size]

        # Calculate scores
        return bmm(transformed, hidden.unsqueeze(2)).squeeze(2)  # [batch_size, src_len]

    def _concat_score(self, encoder_outputs: Tensor, decoder_hidden: Tensor) -> Tensor:
        """ Concat attention score with learned transformations
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :param decoder_hidden: decoder hidden state [1, batch_size, hidden_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        # print("Concat attention score called.")

        # hidden: [batch_size, hidden_size]
        hidden = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)

        if self._projection is not None:
            hidden = self._projection(hidden)  # [batch_size, encoder_hidden_size]
            # print(f"After projection: {hidden.shape}")

        # concat on last dim
        energy = nn.functional.tanh(self._attn(cat([hidden, encoder_outputs], dim=2)))  # [B, S, H]

        # linear → score
        return self._value(energy).squeeze(-1)  # [B, S]


if __name__ == "__main__":
    pass
