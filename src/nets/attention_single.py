#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 14:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   attention_single.py
# @Desc     :

from torch import Tensor, nn, tanh, softmax, bmm, cat

from src.configs.cfg_types import Attentions


class Attention(nn.Module):
    """ Base attention mechanism for sequence models """

    def __init__(self, hidden_size: int, method: Attentions = "dot") -> None:
        """ Initialize the attention mechanism
        :param hidden_size: hidden state dimension
        :param method: attention method ('dot', 'general', 'concat')
        """
        super().__init__()
        self._hidden_size = hidden_size
        self._method = method

        if self._method == "general":
            self._attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self._method == "concat":
            self._attn = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self._value = nn.Parameter(nn.init.xavier_uniform_(nn.Linear(hidden_size, 1, bias=False).weight))

    def forward(self, hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """ Forward pass for attention
        :param hidden: decoder hidden state [1, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        if self._method == "dot":
            attn_energies = self._dot_score(hidden, encoder_outputs)
        elif self._method == "general":
            attn_energies = self._general_score(hidden, encoder_outputs)
        elif self._method == "concat":
            attn_energies = self._concat_score(hidden, encoder_outputs)
        else:
            raise ValueError(f"Unknown attention method: {self._method}")

        # Normalize energies to get attention weights
        attn_weights = softmax(attn_energies, dim=1)

        # Calculate context vector
        context = bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)

        return attn_weights, context

    @staticmethod
    def _dot_score(hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """ Dot product attention score
        :param hidden: decoder hidden state [1, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :return: attention energies [batch_size, src_len]
        """
        # hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [src_len, batch_size, hidden_size]
        hidden = hidden.squeeze(0)  # [batch_size, hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, src_len, hidden_size]

        # Dot product
        attn_energies = bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)  # [batch_size, src_len]

        return attn_energies

    def _general_score(self, hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """ General attention score with learned linear transformation
        :param hidden: decoder hidden state [1, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        # hidden: [1, batch_size, hidden_size]
        hidden = hidden.squeeze(0)  # [batch_size, hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, src_len, hidden_size]

        # Transform encoder outputs
        transformed = self._attn(encoder_outputs)  # [batch_size, src_len, hidden_size]

        # Calculate scores
        attn_energies = bmm(transformed, hidden.unsqueeze(2)).squeeze(2)  # [batch_size, src_len]

        return attn_energies

    def _concat_score(self, hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """ Concatenation attention score (Bahdanau style)
        :param hidden: decoder hidden state [1, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs [src_len, batch_size, hidden_size]
        :return: attention weights [batch_size, src_len], context vector [batch_size, hidden_size]
        """
        # hidden: [1, batch_size, hidden_size]
        src_len = encoder_outputs.size(0)
        hidden = hidden.repeat(src_len, 1, 1)  # [src_len, batch_size, hidden_size]
        hidden = hidden.transpose(0, 1)  # [batch_size, src_len, hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, src_len, hidden_size]

        # Concatenate and transform
        combined = tanh(self._attn(cat([hidden, encoder_outputs], dim=2)))  # [batch_size, src_len, hidden_size]

        # Calculate energies
        attn_energies = self._value.matmul(combined.transpose(1, 2)).squeeze(1)  # [batch_size, src_len]

        return attn_energies


if __name__ == "__main__":
    pass
