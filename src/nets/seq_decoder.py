#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 00:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_decoder.py
# @Desc     :   

from torch import (Tensor, nn, device, zeros_like,
                   randint)
from typing import final, Literal, Final

from src.configs.cfg_types import SeqNets
from src.nets.seq_encoder import SeqEncoder


class SeqDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = False,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu", PAD: int = 0,
                 *,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = SeqNets.GRU,
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
        self._L: int = vocab_size  # Lexicon/Vocabulary size
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._dropout: float = dropout_rate if num_layers > 1 else 0.0
        self._bid: bool = bidirectional
        self._accelerator: device = device(accelerator.lower())
        self._PAD: Final[int] = PAD
        self._type: str = net_category.lower()  # Network category

        self._embed = nn.Embedding(self._L, self._H, padding_idx=self._PAD)
        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers, batch_first=True,
            dropout=dropout_rate
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

        # Keep consistent return types
        if self._type == "lstm":
            outputs, (hn, cn) = self._net(embeddings, hidden)
        else:
            # RNN & GRU
            outputs, hn = self._net(embeddings, hidden)
            cn = zeros_like(hn, device=device(self._accelerator))

        logits = self._linear(self._drop(outputs))

        return logits, (hn, cn)

    # Decoder Preparation
    def init_decoder_entries(self,
                             hidden: Tensor,
                             merge_method: str | Literal["concat", "max", "mean", "sum"] = "mean"
                             ) -> Tensor:
        """ Initialize the decoder input from the encoder hidden state
        :param hidden: encoder hidden state [num_layers * num_directions, batch_size, hidden_size]
        :param encoder_bid: whether the encoder is bidirectional
        :param decoder_bid: whether the decoder is bidirectional
        :param merge_method: method to combine bidirectional hidden states ('mean', 'max', 'sum', 'concat')
        :return: decoder initial hidden state [num_layers, batch_size, hidden_size] or
        """
        if self._bid:
            raise ValueError("Currently only supports decoder_bid=False")

        num_layers_times_num_directions, batches, hidden_size = hidden.shape
        num_directions: int = 2
        num_layers: int = num_layers_times_num_directions // num_directions

        # Reconstruct hidden state
        match merge_method.lower():
            case "mean":
                return hidden.view(num_layers, num_directions, batches, hidden_size).mean(dim=1)
            case "max":
                return hidden.view(num_layers, num_directions, batches, hidden_size).max(dim=1).values
            case "sum":
                return hidden.view(num_layers, num_directions, batches, hidden_size).sum(dim=1)
            case "concat":
                # if merge_method="concat"，decoder hidden_size must double that of encoder
                hn = hidden.view(num_layers, num_directions, batches, hidden_size)
                # Forward + Backward concatenation = 2 * hidden_size
                return hn.transpose(1, 2).reshape(num_layers, batches, num_directions * hidden_size)
            case _:
                raise ValueError(f"Unsupported method: {self._method}")


if __name__ == "__main__":
    vocab_size = 10
    embedding_dim = 8
    hidden_size = 16
    num_layers = 2
    seq_len = 5
    batch_size = 3

    # Initialise encoder
    encoder_gru = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, net_category="gru")
    encoder_lstm = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, net_category="lstm")
    encoder_rnn = SeqEncoder(vocab_size, embedding_dim, hidden_size, num_layers, net_category="rnn")

    # Initialise encoder
    decoder_gru = SeqDecoder(vocab_size, embedding_dim, hidden_size, num_layers, net_category="gru")
    decoder_lstm = SeqDecoder(vocab_size, embedding_dim, hidden_size, num_layers, net_category="lstm")
    decoder_rnn = SeqDecoder(vocab_size, embedding_dim, hidden_size, num_layers, net_category="rnn")

    # Input random sequence (batch_size, seq_len)
    src = randint(0, vocab_size, (batch_size, seq_len))
    tgt = randint(0, vocab_size, (batch_size, seq_len))

    # Encoder forward
    outputs_gru, (hidden_gru, cell_gru), lengths_gru = encoder_gru(src)
    outputs_lstm, (hidden_lstm, cell_lstm), lengths_lstm = encoder_lstm(src)
    outputs_rnn, (hidden_rnn, cell_rnn), lengths_rnn = encoder_rnn(src)

    print("*" * 64)
    print("Encoder Test Results")
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
    print("Decoder Test Outputs")
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
