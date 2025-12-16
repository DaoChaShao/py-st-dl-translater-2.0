#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 00:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_decoder.py
# @Desc     :   

from torch import Tensor, nn, device
from typing import final

from src.configs.cfg_types import Seq2SeqNets


class SeqDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 *,
                 dropout_rate: float = 0.3, bidirectional: bool = False,
                 accelerator: str = "cpu", PAD: int = 0,
                 net_category: Seq2SeqNets | str = Seq2SeqNets.GRU,
                 ) -> None:
        super().__init__()
        """ Initialise the Decoder class
        :param vocab_size: size of the target vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD: padding index for the embedding layer
        :param net_category: network category (e.g., 'gru')
        """
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        self._L: int = vocab_size  # Lexicon/Vocabulary size
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._dropout: float = dropout_rate if num_layers > 1 else 0.0
        self._bid: bool = bidirectional
        self._accelerator: device = device(accelerator)
        self._PAD: int = PAD
        self._type = net_category  # Network category

        self._embed = nn.Embedding(self._L, self._H, padding_idx=self._PAD)
        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self._drop = nn.Dropout(p=self._dropout)
        self._linear = nn.Linear(self._M, self._L)

    @staticmethod
    def _select_net(net_category: str) -> type:
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if net_category not in nets:
            raise ValueError(f"Unsupported net_category: {net_category}")
        return nets[net_category]

    @final
    def forward(self, tgt: Tensor, hidden: Tensor | tuple[Tensor, Tensor]) -> tuple:
        embeddings = self._embed(tgt)

        if self._type == "lstm":
            outputs, (hn, cn) = self._net(embeddings, hidden)
            logits = self._linear(self._drop(outputs))
            return logits, (hn, cn)
        else:
            h = hidden[0] if isinstance(hidden, tuple) else hidden
            outputs, hn = self._net(embeddings, h)
            logits = self._linear(self._drop(outputs))
            return logits, (hn,)


if __name__ == "__main__":
    pass
