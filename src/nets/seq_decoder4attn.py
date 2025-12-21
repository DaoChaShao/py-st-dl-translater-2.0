#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/20 14:43
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_decoder4attn.py
# @Desc     :   

from torch import (Tensor, nn, device, zeros_like,
                   cat)
from typing import final, Literal, Final

from src.configs.cfg_types import SeqNets
from src.nets.attentions import (AdditiveAttention,
                                 DotProductAttention,
                                 ScaledDotProductAttention)


class SeqAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu", PAD: int = 0,
                 *,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = SeqNets.GRU,
                 use_attention: bool = False,
                 attn_category: str | Literal["bahdanau", "dot", "sdot"] = "bahdanau"
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
        # Initialise additive attention
        self._use_attn: bool = use_attention
        self._attn_type: str = attn_category.lower()
        if self._use_attn:
            self._init_attn()
            # When using attention, the input size to RNN increases
            entry_size: int = self._H + self._M
        else:
            entry_size: int = self._H

        self._embed = nn.Embedding(self._L, self._H, padding_idx=self._PAD)
        self._net = self._select_net(self._type)(
            entry_size, self._M, num_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout_rate
        )
        self._drop = nn.Dropout(p=self._dropout)
        self._linear = nn.Linear(self._M, self._L)

    def _init_attn(self):
        match self._attn_type:
            case "bahdanau":
                self._attn = self._select_attn(self._attn_type)(
                    enc_hn_size=self._M, dec_hn_size=self._M, attn_size=self._M
                )
            case _:
                self._attn = self._select_attn(self._attn_type)(
                    enc_hn_size=self._M, dec_hn_size=self._M
                )

    @staticmethod
    def _select_attn(attn_category: str) -> type:
        attentions: dict[str, type] = {
            "bahdanau": AdditiveAttention,
            "dot": DotProductAttention,
            "sdot": ScaledDotProductAttention,
        }

        if attn_category not in attentions:
            raise ValueError(f"Unsupported attn_category: {attn_category}")

        return attentions[attn_category]

    @staticmethod
    def _set_num_directions(bidirectional: bool) -> int:
        return 2 if bidirectional else 1

    @staticmethod
    def _select_net(net_category: str) -> type:
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        if net_category not in nets:
            raise ValueError(f"Unsupported net_category: {net_category}")

        return nets[net_category]

    @final
    def forward(self,
                tgt: Tensor, hidden: Tensor | tuple[Tensor, Tensor],
                enc_outs: Tensor, src: Tensor | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """ Forward pass for the decoder with attention
        :param tgt: target sequence input [batch_size, tgt_len]
        :param hidden: previous hidden state (and cell state for LSTM)
                       [num_layers, batch_size, hidden_size] or
                       tuple of ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        :param enc_outs: encoder outputs for attention [src_len, batch_size, hidden_size]
        :param src: source sequence input for padding mask [batch_size, src_len]
        :return: logits [batch_size, tgt_len, vocab_size], new hidden state (and cell state for LSTM)
        """
        # Embedding the target sequence
        embeddings = self._embed(tgt)

        # Apply attention mechanism if enabled
        if self._use_attn:
            # Apply attention mechanism, get the last enc hn as query in decoder
            dec_hn = hidden[-1]
            # Set padding mask if source is provided
            pad_mask = (src == self._PAD) if src is not None else None
            # Add attention context to embeddings, need [batch, src_len]
            attn_weights, context = self._attn(dec_hn, enc_outs, pad_mask=pad_mask)
            # Concatenate context with embeddings
            embeddings = cat((embeddings, context.unsqueeze(1).expand_as(embeddings)), dim=2)

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
        num_layers_times_num_directions, batches, hidden_size = hidden.shape
        num_layers: int = num_layers_times_num_directions // self._num_directions

        # Reconstruct hidden state
        match merge_method.lower():
            case "mean":
                return hidden.view(num_layers, self._num_directions, batches, hidden_size).mean(dim=1)
            case "max":
                return hidden.view(num_layers, self._num_directions, batches, hidden_size).max(dim=1).values
            case "sum":
                return hidden.view(num_layers, self._num_directions, batches, hidden_size).sum(dim=1)
            case "concat":
                # if merge_method="concat"，decoder hidden_size must double that of encoder
                hn = hidden.view(num_layers, self._num_directions, batches, hidden_size)
                # Forward + Backward concatenation = 2 * hidden_size
                return hn.transpose(1, 2).reshape(num_layers, batches, self._num_directions * hidden_size)
            case _:
                raise ValueError(f"Unsupported method: {self._method}")


if __name__ == "__main__":
    pass
