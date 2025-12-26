#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 16:44
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_attn_lstm.py
# @Desc     :   

from random import random
from torch import (Tensor, nn,
                   device, full, long, cat, stack, arange,
                   randint)
from typing import override, Literal

from src.configs.cfg_types import SeqMergeMethods, SeqStrategies, SeqNets, AttnScorer
from src.nets.base_seq2seq import BaseSeqNet
from src.nets.seq_decoder import SeqDecoder
from src.nets.seq_decoder4attn import SeqDecoderWithAttn
from src.nets.seq_encoder import SeqEncoder
from src.utils.PT import TorchRandomSeed


class SeqToSeqLSTMWithAttn(BaseSeqNet):
    """ Sequence-to-Sequence LSTM Network for Sequence Tasks with and without Attention Mechanism """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD_SRC: int = 0, PAD_TGT: int = 0, SOS: int = 2, EOS: int = 3,
                 *,
                 merge_method: str | SeqMergeMethods | Literal["concat", "max", "mean", "sum"] = "concat",
                 teacher_forcing_ratio: float = 0.5,
                 use_attention: bool = True,
                 attn_scorer: str | AttnScorer | Literal["bahdanau", "dot", "sdot"] = "dot"
                 ) -> None:
        """ Initialize the SeqToSeqTaskGRU class
        :param vocab_size_src: size of the source vocabulary
        :param vocab_size_tgt: size of the target vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_size: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD_SRC: padding index for the source embedding layer
        :param PAD_TGT: padding index for the target embedding layer
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        :param merge_method: method to merge bidirectional hidden states
        :param use_attention: whether to use attention mechanism
        :param attn_scorer: attention scorer type
        """
        super().__init__(
            vocab_size_src=vocab_size_src,
            vocab_size_tgt=vocab_size_tgt,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            accelerator=accelerator,
            PAD_SRC=PAD_SRC,
            PAD_TGT=PAD_TGT,
            SOS=SOS,
            EOS=EOS
        )
        self._method: str = merge_method.lower()
        self._teacher_ratio: float = teacher_forcing_ratio

        self._use_attn: bool = use_attention
        self._scorer: str = attn_scorer

        # Initialise encoder and decoder
        self._encoder: nn.Module = self.init_encoder()
        self._decoder: nn.Module = self.init_decoder()

        self.init_weights()

    @override
    def init_encoder(self) -> nn.Module:
        """ Initialize the encoder module
        :return: encoder module
        """
        return SeqEncoder(
            self._vocab_src, self._H, self._M, self._C,
            dropout_rate=self._dropout, bidirectional=self._bid,
            accelerator=self._accelerator,
            PAD=self._PAD_SRC,
            net_category=SeqNets.LSTM,
        )

    @override
    def init_decoder(self) -> nn.Module:
        """ Initialize the decoder module
        :return: decoder module
        """
        if self._use_attn:
            return SeqDecoderWithAttn(
                self._vocab_tgt, self._H, self._M, self._C,
                dropout_rate=self._dropout, bidirectional=self._bid,
                accelerator=self._accelerator,
                PAD=self._PAD_TGT,
                net_category=SeqNets.LSTM,
                use_attention=self._use_attn,
                attn_category=self._scorer
            )
        else:
            hidden_size = self._M * 2 if self._bid and self._method == "concat" else self._M

            return SeqDecoder(
                self._vocab_tgt, self._H, hidden_size, self._C,
                dropout_rate=self._dropout, bidirectional=self._bid,
                accelerator=self._accelerator,
                PAD=self._PAD_TGT,
                net_category=SeqNets.LSTM,
            )

    @override
    def _merge_bidirectional_hidden(self, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """ Merge bidirectional hidden states for decoder initialization
        :param hidden: hidden states from the encoder
        :return: merged hidden states
        """
        h = self._decoder.init_decoder_entries(hidden[0], merge_method=self._method)
        c = self._decoder.init_decoder_entries(hidden[1], merge_method=self._method)

        return h, c

    @override
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass through the Seq2Seq model
        :param src: source/input tensor
        :param tgt: target/output tensor
        :return: output logits tensor
        """
        _, tgt_len = tgt.shape

        # Encoder forward
        enc_outs, (enc_hn, enc_cn), _ = self._encoder(src)
        # Merge bidirectional hidden for decoder initialization
        dec_hn_entries, dec_cn_entries = self._merge_bidirectional_hidden((enc_hn, enc_cn))

        # Initialize decoder input with <SOS> tokens, shape: (B,1)
        tokens_entries: Tensor = tgt[:, 0].unsqueeze(1)

        L: list[Tensor] = []
        # start from 1 because 0 is <SOS>
        for ts in range(1, tgt_len):
            # Decoder step
            # decide decoder call based on attention usage
            if self._use_attn:
                logit, (dec_hn_entries, dec_cn_entries) = self._decoder(
                    tokens_entries, (dec_hn_entries, dec_cn_entries), enc_outs
                )
            else:
                logit, (dec_hn_entries, dec_cn_entries) = self._decoder(
                    tokens_entries, (dec_hn_entries, dec_cn_entries)
                )
            # shape: (B,1,V)
            L.append(logit)

            # Teacher forcing decision
            if random() < self._teacher_ratio:
                # teacher forcing
                tokens_entries = tgt[:, ts].unsqueeze(1)
            else:
                # model prediction
                tokens_entries = logit.argmax(dim=2)

        # Concatenate all logits along time dimension, shape: (B, tgt_len-1, V)
        return cat(L, dim=1)

    @override
    def generate(self,
                 src: Tensor, max_len: int = 50,
                 strategy: str | SeqStrategies | Literal["greedy", "beam"] = "greedy", beam_width: int = 5,
                 dec_hn: Tensor | None = None
                 ) -> Tensor:
        """ Generate sequences
        :param src: source/input tensor
        :param max_len: maximum length of generated sequences
        :param strategy: generation strategy ('greedy' or 'beam')
        :param beam_width: beam width for beam search
        :param dec_hn: decoder hidden state
        :return: generated sequences tensor
        """
        batches = src.size(0)

        # Encoder
        outputs, (hn, cn), _ = self._encoder(src)

        # Combine bidirectional hidden states
        hn_entries, cn_entries = self._merge_bidirectional_hidden((hn, cn))

        match strategy:
            case "greedy":
                return self._greedy_decode(
                    (hn_entries, cn_entries), batches, max_len, src.device, enc_outs=outputs
                )
            case "beam":
                return self._beam_search_decode(
                    (hn_entries, cn_entries), batches, max_len, beam_width, src.device, enc_outs=outputs
                )
            case _:
                raise ValueError(f"Unknown generation strategy: {strategy}")

    @override
    def _greedy_decode(self,
                       hidden: tuple[Tensor, Tensor],
                       batch_size: int, max_len: int, accelerator: device,
                       *,
                       enc_outs: Tensor | None = None,
                       ) -> Tensor:
        """ Greedy decoding implementation
        :param hidden: initial hidden state for the decoder
        :param batch_size: size of the batch
        :param max_len: maximum length of generated sequences
        :param accelerator: device for computation
        :return: generated sequences tensor
        """
        dec_hn, dec_cn = hidden

        tgt = full((batch_size, 1), self._SOS, dtype=long, device=accelerator)

        generation: list[Tensor] = []
        for _ in range(max_len):
            # Decoder step
            if self._use_attn:
                logits, (dec_hn, dec_cn) = self._decoder(tgt, (dec_hn, dec_cn), enc_outs)
            else:
                logits, (dec_hn, dec_cn) = self._decoder(tgt, (dec_hn, dec_cn))
            # Take the most probable token
            next_tokens = logits.argmax(dim=2)  # [B, 1]
            generation.append(next_tokens)
            # Update target tokens
            tgt = next_tokens

        # Concatenate along time dimension - [B, max_len]
        return cat(generation, dim=1)

    @override
    def _beam_search_decode(self,
                            hidden: tuple[Tensor, Tensor],
                            batch_size: int, max_len: int, beam_width: int, accelerator: device,
                            *,
                            enc_outs: Tensor | None = None
                            ) -> Tensor:
        """ Beam search decoding implementation
        :param hidden: initial hidden state for the decoder
        :param batch_size: size of the batch
        :param max_len: maximum length of generated sequences
        :param beam_width: beam width for beam search - K
        :param accelerator: device for computation
        :return: generated sequences tensor
        """
        dec_hn, dec_cn = hidden

        # beams: [B, K, 1]
        beams = full((batch_size, beam_width, 1), self._SOS, dtype=long, device=accelerator)

        # scores: [B, K]
        scores = full((batch_size, beam_width), float("-inf"), device=accelerator)
        scores[:, 0] = 0.0

        # hidden: [L, B, H] → [L, B, K, H]
        dec_h = dec_hn.unsqueeze(2).expand(dec_hn.size(0), dec_hn.size(1), beam_width, -1)
        dec_c = dec_cn.unsqueeze(2).expand(dec_cn.size(0), dec_cn.size(1), beam_width, -1)

        # enc_outs: [B, S, H] → [B, K, S, H]
        if enc_outs is not None:
            enc_outs = enc_outs.unsqueeze(1).expand(-1, beam_width, -1, -1)

        for _ in range(max_len):
            # tgt: [B*K, 1]
            tgt = beams[:, :, -1].reshape(batch_size * beam_width, 1)

            # hidden: [L, B*K, H]
            flat_hn = dec_h.reshape(dec_h.size(0), batch_size * beam_width, -1)
            flat_cn = dec_c.reshape(dec_c.size(0), batch_size * beam_width, -1)

            if self._use_attn and enc_outs is not None:
                flat_enc = enc_outs.reshape(batch_size * beam_width, enc_outs.size(2), enc_outs.size(3))
                logits, (new_hn, new_cn) = self._decoder(tgt, (flat_hn, flat_cn), flat_enc)
            else:
                logits, (new_hn, new_cn) = self._decoder(tgt, (flat_hn, flat_cn))

            # log_probs: [B, K, V]
            log_probs = logits.squeeze(1).log_softmax(dim=-1)
            log_probs = log_probs.view(batch_size, beam_width, -1)

            # total_scores: [B, K, V]
            total_scores = scores.unsqueeze(2) + log_probs

            # flatten to [B, K*V]
            total_scores = total_scores.view(batch_size, -1)

            # top-K
            top_scores, top_indices = total_scores.topk(beam_width, dim=1)

            vocab_size = log_probs.size(-1)
            beam_indices = top_indices // vocab_size  # [B, K]
            token_indices = top_indices % vocab_size  # [B, K]

            # update beams
            new_beams = []
            for b in range(batch_size):
                new_beams.append(cat([beams[b, beam_indices[b]], token_indices[b].unsqueeze(1)], dim=1))

            beams = stack(new_beams, dim=0)  # [B, K, T+1]
            scores = top_scores

            # update hidden
            new_hn = new_hn.view(new_hn.size(0), batch_size, beam_width, dec_hn.size(2))
            new_cn = new_cn.view(new_cn.size(0), batch_size, beam_width, dec_cn.size(2))
            dec_h = new_hn.gather(2, beam_indices.unsqueeze(0).unsqueeze(-1).expand_as(new_hn))
            dec_c = new_cn.gather(2, beam_indices.unsqueeze(0).unsqueeze(-1).expand_as(new_cn))

        # select best beam
        best = scores.argmax(dim=1)  # [B]
        batch_idx = arange(batch_size, device=accelerator)
        outputs = beams[batch_idx, best]  # [B, T]

        return outputs[:, 1:]


if __name__ == "__main__":
    with TorchRandomSeed("Test"):
        model = SeqToSeqLSTMWithAttn(
            vocab_size_src=80,
            vocab_size_tgt=100,
            embedding_dim=65,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            merge_method="concat",
            use_attention=True,
            attn_scorer="dot"
        )

        batch_size = 4
        src_len = 11
        tgt_len = 17

        src = randint(4, 70, (batch_size, src_len))
        tgt = cat([
            full((batch_size, 1), model._SOS, dtype=long),
            randint(5, 80, (batch_size, tgt_len - 2)),
            full((batch_size, 1), model._EOS, dtype=long)
        ], dim=1)

        # Forward
        logits = model(src, tgt)
        print("Forward pass output shape:", logits.shape)  # [B, tgt_len-1, vocab_tgt]
        print()

        # Greedy
        generated = model.generate(src, max_len=8, strategy="greedy")
        print(f"Generated sequences (greedy):\n{generated}")
        print()

        # Beam
        generated = model.generate(src, max_len=8, strategy="beam")
        print("Generated sequences (beam):\n", generated)
        print()
