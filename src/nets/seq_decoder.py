#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 00:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_decoder.py
# @Desc     :   

from random import choice
from torch import (Tensor, nn, device, zeros_like,
                   randint)
from typing import final, Literal, Final

from src.configs.cfg_types import SeqNets
from src.nets.seq_encoder import SeqEncoder


class SeqDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu", PAD: int = 0,
                 *,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = SeqNets.GRU,
                 ) -> None:
        """ Initialise the Decoder class
        :param vocab_size: size of the target vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_size: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD: padding index for the embedding layer
        :param net_category: network category (e.g., 'gru')
        """
        super().__init__()
        self._L: int = vocab_size  # Lexicon/Vocabulary size
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._dropout: float = dropout_rate if num_layers > 1 else 0.0
        self._num_directions: int = self._set_num_directions(bidirectional)
        self._accelerator: device = device(accelerator.lower())
        self._PAD: Final[int] = PAD
        self._type: str = net_category.lower()  # Network category

        self._embed = nn.Embedding(self._L, self._H, padding_idx=self._PAD)
        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=False,
            dropout=self._dropout
        )
        self._drop = nn.Dropout(p=self._dropout)
        self._linear = nn.Linear(self._M, self._L)

    @staticmethod
    def _set_num_directions(bidirectional: bool) -> int:
        """ Set the number of directions based on bidirectionality
        :param bidirectional: whether the RNN is bidirectional
        :return: number of directions (1 or 2)
        """
        return 2 if bidirectional else 1

    @staticmethod
    def _select_net(net_category: str) -> type:
        """ Select the RNN network type based on the category
        :param net_category: network category ('rnn', 'lstm', 'gru')
        :return: corresponding PyTorch RNN class
        """
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        if net_category not in nets:
            raise ValueError(f"Unsupported net_category: {net_category}")

        return nets[net_category]

    @final
    def forward(self, tgt: Tensor, enc_hn: Tensor | tuple[Tensor, Tensor]) -> tuple:
        """ Forward pass for the decoder
        :param tgt: target input sequence [batch_size, tgt_len]
        :param enc_hn: initial hidden state (and cell state for LSTM)
        :return: logits [batch_size, tgt_len, vocab_size], final hidden state (and cell state for LSTM)
        """
        embeddings = self._embed(tgt)

        # Keep consistent return types
        if self._type == "lstm":
            outputs, (hn, cn) = self._net(embeddings, enc_hn)
        else:
            # RNN & GRU
            outputs, hn = self._net(embeddings, enc_hn)
            cn = zeros_like(hn, device=self._accelerator)

        logits = self._linear(self._drop(outputs))

        return logits, (hn, cn)

    # Decoder Preparation
    def init_decoder_entries(self,
                             enc_hn: Tensor,
                             merge_method: str | Literal["concat", "max", "mean", "sum"] = "mean"
                             ) -> Tensor:
        """ Initialize the decoder input from the encoder hidden state
        :param enc_hn: encoder hidden state [num_layers * num_directions, batch_size, hidden_size]
        :param merge_method: method to combine bidirectional hidden states ('mean', 'max', 'sum', 'concat')
        :return: decoder initial hidden state [num_layers, batch_size, hidden_size] or
        """
        num_layers_times_num_directions, batches, enc_hn_size = enc_hn.shape
        num_layers: int = num_layers_times_num_directions // self._num_directions

        # reshape encoder hidden to [num_layers, enc_num_directions, batch, hidden_size]
        hn = enc_hn.view(num_layers, self._num_directions, batches, enc_hn_size)
        # Reconstruct hidden state
        match merge_method.lower():
            case "mean":
                return enc_hn.view(num_layers, self._num_directions, batches, enc_hn_size).mean(dim=1)
            case "max":
                return enc_hn.view(num_layers, self._num_directions, batches, enc_hn_size).max(dim=1).values
            case "sum":
                return enc_hn.view(num_layers, self._num_directions, batches, enc_hn_size).sum(dim=1)
            case "concat":
                # if merge_method="concat"，decoder hidden_size must double that of encoder
                return hn.transpose(1, 2).reshape(num_layers, batches, self._num_directions * enc_hn_size)
            case _:
                raise ValueError(f"Unsupported method: {merge_method}")


if __name__ == "__main__":
    vocab_size = 10
    embedding_dim = 8
    hidden_size = 16
    num_layers = 2
    seq_len = 5
    batch_size = 3

    # Initialise encoder
    bid: bool = choice([True, False])
    encoder_gru = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="gru")
    encoder_lstm = SeqEncoder(
        vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="lstm"
    )
    encoder_rnn = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="rnn")

    # Initialise encoder
    decoder_gru = SeqDecoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="gru")
    decoder_lstm = SeqDecoder(
        vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="lstm"
    )
    decoder_rnn = SeqDecoder(vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=bid, net_category="rnn")

    # Input random sequence (batch_size, seq_len)
    src = randint(0, vocab_size, (batch_size, seq_len))
    tgt = randint(0, vocab_size, (batch_size, seq_len))

    # Encoder forward
    outputs_gru, (hidden_gru, cell_gru), lengths_gru = encoder_gru(src)
    outputs_lstm, (hidden_lstm, cell_lstm), lengths_lstm = encoder_lstm(src)
    outputs_rnn, (hidden_rnn, cell_rnn), lengths_rnn = encoder_rnn(src)

    print("*" * 64)
    print(f"Encoder Test Results (bidirectional={bid})")
    print("*" * 64)
    print("GRU Encoder outputs shape:", outputs_gru.shape)
    print("GRU Encoder Hidden shape:", hidden_gru.shape)
    print("GRU Encoder Cell shape:", cell_gru.shape)
    print("GRU Encoder Lengths:", lengths_gru)
    print("-" * 64)
    print("LSTM Encoder outputs shape:", outputs_lstm.shape)
    print("LSTM Encoder Hidden shape:", hidden_lstm.shape)
    print("LSTM Encoder Cell shape:", cell_lstm.shape)
    print("LSTM Encoder Lengths:", lengths_lstm)
    print("-" * 64)
    print("RNN Encoder outputs shape:", outputs_rnn.shape)
    print("RNN Encoder Hidden shape:", hidden_rnn.shape)
    print("RNN Encoder Cell shape:", cell_rnn.shape)
    print("RNN Encoder Lengths:", lengths_rnn)
    print("*" * 64)
    print()

    # Decoder forward
    logits_gru, (hn_gru, cn_gru) = decoder_gru(tgt, decoder_gru.init_decoder_entries(hidden_gru))
    logits_lstm, (hn_lstm, cn_lstm) = decoder_lstm(
        tgt, (decoder_gru.init_decoder_entries(hidden_lstm), decoder_gru.init_decoder_entries(cell_lstm))
    )
    logits_rnn, (hn_rnn, cn_rnn) = decoder_rnn(tgt, decoder_gru.init_decoder_entries(hidden_rnn))

    print("*" * 64)
    print(f"Decoder Test Outputs (bidirectional={bid})")
    print("*" * 64)
    print("GRU Decoder outputs shape:", logits_gru.shape)
    print("GRU Decoder Hidden shape:", hn_gru.shape)
    print("GRU Decoder Cell shape:", cn_gru.shape)
    print("-" * 64)
    print("LSTM Decoder outputs shape:", logits_lstm.shape)
    print("LSTM Decoder Hidden shape:", hn_lstm.shape)
    print("LSTM Decoder Cell shape:", cn_lstm.shape)
    print("-" * 64)
    print("RNN Decoder outputs shape:", logits_rnn.shape)
    print("RNN Decoder Hidden shape:", hn_rnn.shape)
    print("RNN Decoder Cell shape:", cn_rnn.shape)
    print("*" * 64)
