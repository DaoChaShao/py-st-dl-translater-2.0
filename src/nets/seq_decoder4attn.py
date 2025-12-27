#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/20 14:43
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq_decoder4attn.py
# @Desc     :   

from random import choice
from torch import (Tensor, nn, device, zeros_like,
                   cat,
                   randint)
from typing import final, Literal, Final

from src.configs.cfg_types import SeqNets
from src.nets.attentions import AdditiveAttention, DotProductAttention, ScaledDotProductAttention
from src.nets.seq_encoder import SeqEncoder


class SeqDecoderWithAttn(nn.Module):
    """ Sequence Decoder with Attention Mechanism """

    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = SeqNets.GRU,
                 *,
                 use_attention: bool = False,
                 attn_category: str | Literal["bahdanau", "dot", "sdot"] = "dot"
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
        :param use_attention: whether to use attention mechanism
        :param attn_category: attention category (e.g., 'dot')
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

        # Initialise attention
        self._use_attn: bool = use_attention
        self._attn_type: str = attn_category.lower()
        if self._use_attn:
            self._init_attn()

            # Align encoder output dimension with decoder hidden dimension
            if self._num_directions > 1:
                self._aligner = nn.Sequential(nn.Linear(self._M * self._num_directions, self._M), nn.ReLU())
            else:
                self._aligner = nn.Identity()

        self._embed = nn.Embedding(self._L, self._H, padding_idx=self._PAD)
        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=False,
            dropout=self._dropout
        )
        self._drop = nn.Dropout(p=self._dropout)
        # Output linear layer
        self._linear = nn.Linear(self._M, self._L)

    def _init_attn(self):
        self._attn = self._select_attn(self._attn_type)(
            enc_hn_dims=self._M, dec_hn_dims=self._M
        )

    @staticmethod
    def _select_attn(attn_category: str) -> type:
        attentions: dict[str, type] = {
            "bahdanau": AdditiveAttention,
            "dot": DotProductAttention,
            "sdot": ScaledDotProductAttention
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
    def forward(self, tgt: Tensor, hidden: Tensor | tuple[Tensor, Tensor], enc_outs: Tensor):
        """ Forward pass for the decoder with attention
        :param tgt: target sequences [batch_size, tgt_len]
        :param hidden: initial hidden state for the decoder [num_layers, batch_size, hidden_size]
        :param enc_outs: encoder outputs for attention [batch_size, src_len, hidden_size]
        :return:
        - logits: output logits [batch_size, tgt_len, vocab_size]
        - (hn, cn): final hidden and cell states
        """
        B, tgt_len = tgt.size()

        # Embedding
        embeddings = self._embed(tgt)

        if self._type == "lstm":
            outputs, (hn, cn) = self._net(embeddings, hidden)
        else:
            outputs, hn = self._net(embeddings, hidden)
            cn = zeros_like(hn, device=self._accelerator)

        if self._use_attn:
            # Align encoder outputs with decoder hidden size
            aligned_enc_outs: Tensor = self._aligner(enc_outs)

            context: list[Tensor] = []
            for ts in range(tgt_len):
                # Get hidden state at current time step - [B, H]
                curr_dec_hn: Tensor = outputs[:, ts, :]

                # Get the context score and vector - [B, M]
                curr_context, _ = self._attn(curr_dec_hn, aligned_enc_outs)

                # Extend context vector dimension - [B, 1, M]
                context.append(curr_context.unsqueeze(1))

            # concat all context
            contexts = cat(context, dim=1)  # [B, tgt_len, H]

            # concat hidden output and context vector
            attn_outs: list[Tensor] = []
            for ts in range(tgt_len):
                attn_outs.append((outputs[:, ts, :] + contexts[:, ts, :]).unsqueeze(1))
            attn_outs: Tensor = cat(attn_outs, dim=1)  # [B, tgt_len, H]
            logits = self._linear(self._drop(attn_outs))

            return logits, (hn, cn)
        else:
            logits = self._linear(self._drop(outputs))

            return logits, (hn, cn)

    # Decoder Preparation
    def init_decoder_entries(self,
                             hidden: Tensor | tuple[Tensor, Tensor],
                             merge_method: str | Literal["concat", "max", "mean", "sum"] = "mean"
                             ) -> Tensor:
        """ Initialize the decoder input from the encoder hidden state
        :param hidden: encoder hidden state [num_layers * num_directions, batch_size, hidden_size]
        :param merge_method: method to combine bidirectional hidden states ('mean', 'max', 'sum', 'concat')
        :return: decoder initial hidden state [num_layers, batch_size, hidden_size] or
        """
        enc_hn = hidden if isinstance(hidden, Tensor) else hidden[0]

        num_layers_times_num_directions, batches, enc_hn_size = hidden.shape
        num_layers: int = num_layers_times_num_directions // self._num_directions

        # reshape encoder hidden to [num_layers, enc_num_directions, batch, hidden_size]
        hn = enc_hn.view(num_layers, self._num_directions, batches, enc_hn_size)
        # Reconstruct hidden state
        match merge_method.lower():
            case "mean":
                return hn.mean(dim=1)
            case "max":
                return hn.max(dim=1).values
            case "sum":
                return hn.sum(dim=1)
            case "concat":
                # Forward + Backward concatenation = 2 * hidden_size
                h = hn.transpose(1, 2).reshape(num_layers, batches, self._num_directions * enc_hn_size)
                return self._aligner(h)
            case _:
                raise ValueError(f"Unsupported method: {merge_method}")


if __name__ == "__main__":
    batch_size = 2
    src_len = 5
    tgt_len = 4
    vocab_size_src = 10
    vocab_size_tgt = 12
    embedding_dim = 16
    hidden_size = 16
    num_layers = 1
    bidirectional = choice([True, False])
    attn_usage = choice([True, False])
    PAD = 0

    # Initialise Encoder and Decoder
    encoder = SeqEncoder(
        vocab_size=vocab_size_src,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        PAD=PAD
    )
    decoder = SeqDecoderWithAttn(
        vocab_size=vocab_size_tgt,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        use_attention=attn_usage,
        attn_category="dot",
        PAD=PAD
    )

    src = randint(1, vocab_size_src, (batch_size, src_len))
    tgt = randint(1, vocab_size_tgt, (batch_size, tgt_len))

    # Encoder forward
    enc_outs, (enc_hn, enc_cn), lengths = encoder(src)
    print("*" * 64)
    print(f"Encoder Results [bidirectional={bidirectional}, attention usage={attn_usage}]")
    print("-" * 64)
    print("Encoder Outputs Shape:", enc_outs.shape)  # [B, src_len, H*directions]
    print("Encoder Hidden Shape:", enc_hn.shape)  # [num_layers*directions, B, H]
    print("Encoder Cell Shape:", enc_cn.shape)  # [num_layers*directions, B, H]
    print("*" * 64)
    print()

    # Initialise decoder hidden entries
    dec_hn = decoder.init_decoder_entries(enc_hn, merge_method="mean")
    print("*" * 64)
    print("Decoder Initial Entries")
    print("-" * 64)
    print(f"Decoder Entries Shape: {dec_hn.shape}")  # [num_layers, B, H]
    print("*" * 64)
    print()

    # Decoder forward
    logits, (dec_hn_out, dec_cn_out) = decoder(tgt, dec_hn, enc_outs)
    print("*" * 64)
    print(f"Decoder Results [bidirectional={bidirectional}, attention usage={attn_usage}]")
    print("-" * 64)
    print("Decoder logits Shape:", logits.shape)  # [B, tgt_len, vocab_size_tgt]
    print("Decoder Hidden Shape:", dec_hn_out.shape)  # [num_layers, B, H]
    print("Decoder Cell Shape:", dec_cn_out.shape)  # [num_layers, B, H]
    print("*" * 64)
