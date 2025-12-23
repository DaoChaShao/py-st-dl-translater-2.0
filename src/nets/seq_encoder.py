#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/15 18:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_encoder.py
# @Desc     :   

from random import choice
from torch import (Tensor, nn, zeros_like, device,
                   randint)
from typing import override, Literal

from src.configs.cfg_types import SeqNets
from src.nets.base_rnn import BaseRNN


class SeqEncoder(BaseRNN):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0,
                 *,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = SeqNets.GRU,
                 ) -> None:
        kwargs = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "bidirectional": bidirectional,
            "accelerator": accelerator,
            "PAD": PAD
        }
        super().__init__(**kwargs)
        """ Initialise the Encoder class
        :param vocab_size: size of the source vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD: padding index for the embedding layer
        """
        self._type: str = net_category.lower()

        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=self._bid,
            dropout=self._dropout
        )

    @staticmethod
    def _select_net(net_category: str) -> type:
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        return nets[net_category]

    @override
    def forward(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        embeddings = self._embed(src)
        lengths: Tensor = (src != self._PAD).sum(dim=1)

        # Keep consistent return types
        if self._type == "lstm":
            outputs, (hidden, cell) = self._net(embeddings)
        else:
            # RNN or GRU
            outputs, hidden = self._net(embeddings)
            cell = zeros_like(hidden, device=device(self._accelerator))

        return outputs, (hidden, cell), lengths


if __name__ == "__main__":
    vocab_size = 10
    embedding_dim = 8
    hidden_size = 16
    num_layers = 2
    bid = choice([True, False])
    seq_len = 5
    batch_size = 3

    # Initialise encoder
    gru = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="gru")
    lstm = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="lstm")
    rnn = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="rnn")

    # Input random sequence (batch_size, seq_len)
    src = randint(0, vocab_size, (batch_size, seq_len))

    # Encoder forward
    outputs_gru, (hidden_gru, cell_gru), lengths_gru = gru(src)
    outputs_lstm, (hidden_lstm, cell_lstm), lengths_lstm = lstm(src)
    outputs_rnn, (hidden_rnn, cell_rnn), lengths_rnn = rnn(src)

    print("*" * 64)
    print(f"Encoder Test Results (bidirectional={bid})")
    print("*" * 64)
    print("GRU Encoder outputs shape:", outputs_gru.shape)
    print("GRU Hidden shape:", hidden_gru.shape)
    print("GRU Cell shape:", cell_gru.shape)
    print("GRU Lengths:", lengths_gru)
    print("-" * 64)
    print("LSTM Encoder outputs shape:", outputs_lstm.shape)
    print("LSTM Hidden shape:", hidden_lstm.shape)
    print("LSTM Cell shape:", cell_lstm.shape)
    print("LSTM Lengths:", lengths_lstm)
    print("-" * 64)
    print("RNN Encoder outputs shape:", outputs_rnn.shape)
    print("RNN Hidden shape:", hidden_rnn.shape)
    print("RNN Cell shape:", cell_rnn.shape)
    print("RNN Lengths:", lengths_rnn)
    print("*" * 64)
