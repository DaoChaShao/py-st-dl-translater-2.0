#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/15 18:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_encoder.py
# @Desc     :   

from torch import Tensor, nn
from typing import override

from src.configs.cfg_types import Seq2SeqNets
from src.nets.base_rnn import BaseRNN


class SeqEncoder(BaseRNN):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str = "cpu",
                 PAD: int = 0,
                 *,
                 net_category: Seq2SeqNets | str = Seq2SeqNets.GRU,
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
        self._type: str = net_category

        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=self._bid,
            dropout=self._dropout if num_layers > 1 else 0.0
        )

    @staticmethod
    def _select_net(net_category: str) -> type:
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        return nets[net_category]

    @override
    def forward(self, src: Tensor) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        embeddings = self._embed(src)
        lengths: Tensor = (src != self._PAD).sum(dim=1)

        result = self._net(embeddings)

        if self._type == "lstm":
            outputs, (hidden, cell) = result
            return outputs, (hidden, cell), lengths
        else:
            outputs, hidden = result
            return outputs, hidden, lengths


if __name__ == "__main__":
    pass
