#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 13:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_task_lstm.py
# @Desc     :   

from math import log
from random import choice
from torch import (Tensor, nn,
                   cat,
                   device, full, long, ones, bool as torch_bool, where, full_like, empty,
                   tensor, topk,
                   randint)
from typing import override, Literal

from src.configs.cfg_types import SeqMergeMethods, SeqStrategies, SeqNets
from src.nets.base_seq2seq import BaseSeqNet
from src.nets.seq_encoder import SeqEncoder
from src.nets.seq_decoder import SeqDecoder


class SeqToSeqLSTM(BaseSeqNet):
    """ Sequence-to-Sequence LSTM Network for Sequence Tasks """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD_SRC: int = 0, PAD_TGT: int = 0, SOS: int = 2, EOS: int = 3,
                 *,
                 merge_method: str | SeqMergeMethods | Literal["concat", "max", "mean", "sum"] = "mean",
                 ) -> None:
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
        """ Initialize the SeqToSeqTaskLSTM class
        :param vocab_size_src: size of the source vocabulary
        :param vocab_size_tgt: size of the target vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD_SRC: padding index for the source embedding layer
        :param PAD_TGT: padding index for the target embedding layer
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        :param net_category: network category (e.g., 'gru')
        """
        self._method: str = merge_method.lower()

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
            PAD=self._PAD_SRC, net_category=SeqNets.LSTM,
        )

    @override
    def init_decoder(self) -> nn.Module:
        """ Initialize the decoder module
        :return: decoder module
        """
        hidden_size = self._M * 2 if (self._bid and self._method == "concat") else self._M

        return SeqDecoder(
            self._vocab_tgt, self._H, hidden_size, self._C,
            dropout_rate=self._dropout, bidirectional=self._bid,
            accelerator=self._accelerator,
            PAD=self._PAD_TGT, net_category=SeqNets.LSTM,
        )

    @override
    def _merge_bidirectional_hidden(self, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """ Merge bidirectional hidden states for decoder initialization
        :param hidden: hidden states from the encoder
        :return: merged hidden states
        """
        hn, cn = hidden
        h = self._decoder.init_decoder_entries(hn, self._method)
        c = self._decoder.init_decoder_entries(cn, self._method)

        return (h, c)

    @override
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass through the Seq2Seq model
        :param src: source/input tensor
        :param tgt: target/output tensor
        :return: output logits tensor
        """
        # Encode
        _, (hidden, cell), lengths_src = self._encoder(src)

        # Combine bidirectional hidden states
        hidden_ety = self._merge_bidirectional_hidden((hidden, cell))
        # Decode input excludes the EOS token
        input_ety = tgt[:, :-1]
        logits, (_, _) = self._decoder(input_ety, hidden_ety)

        return logits

    @override
    def generate(self,
                 src: Tensor, max_len: int = 100,
                 strategy: str | SeqStrategies | Literal["greedy", "beam"] = "greedy", beam_width: int = 5) -> Tensor:
        """ Generate sequences
        :param src: source/input tensor
        :param max_len: maximum length of generated sequences
        :param strategy: generation strategy ('greedy' or 'beam')
        :param beam_width: beam width for beam search
        :return: generated sequences tensor
        """
        batches = src.size(0)

        # Encoder
        _, (hidden, cell), lengths_src = self._encoder(src)

        # Combine bidirectional hidden states
        decoder_hidden = self._merge_bidirectional_hidden((hidden, cell))

        match strategy:
            case "greedy":
                return self._greedy_decode(decoder_hidden, batches, max_len, src.device)
            case "beam":
                return self._beam_search_decode(decoder_hidden, batches, max_len, beam_width, src.device)
            case _:
                raise ValueError(f"Unknown generation strategy: {strategy}")

    @override
    def _greedy_decode(self,
                       decoder_hidden: tuple[Tensor, Tensor],
                       batch_size: int, max_len: int, accelerator: device,
                       encoder_hidden: Tensor | None = None
                       ) -> Tensor:
        """ Greedy decoding implementation
        :param decoder_hidden: initial hidden state for the decoder
        :param batch_size: size of the batch
        :param max_len: maximum length of generated sequences
        :param accelerator: device for computation
        :return: generated sequences tensor
        """

        # Start from SOS token
        decoder_input = full((batch_size, 1), self._SOS, dtype=long, device=accelerator)
        generated = []

        # Track which sequences are still active
        active = ones(batch_size, dtype=torch_bool, device=accelerator)

        for step in range(max_len):
            if not active.any():
                break

            logits, (hn, cn) = self._decoder(decoder_input, decoder_hidden)

            next_token = logits.argmax(dim=2)
            next_token = where(active.unsqueeze(1), next_token, full_like(next_token, self._EOS))

            generated.append(next_token)
            active = active & (next_token.squeeze(1) != self._EOS)
            decoder_input = next_token

            decoder_hidden = (hn, cn)

        return cat(generated, dim=1) if generated else empty((batch_size, 0), dtype=long, device=accelerator)

    @override
    def _beam_search_decode(self,
                            decoder_hidden: tuple[Tensor, Tensor],
                            batch_size: int, max_len: int, beam_width: int, accelerator: device
                            ) -> Tensor:
        """ Beam search decoding implementation
        :param decoder_hidden: initial hidden state for the decoder
        :param batch_size: size of the batch
        :param max_len: maximum length of generated sequences
        :param beam_width: beam width for beam search
        :param accelerator: device for computation
        :return: generated sequences tensor
        """
        hn, cn = decoder_hidden

        results = []

        for idx in range(batch_size):
            # Get hidden state of a single example
            batch_hidden = (hn[:, idx:idx + 1], cn[:, idx:idx + 1])

            # Initialize beams
            beams = [{
                "tokens": [self._SOS],
                "score": 0.0,
                "hidden": batch_hidden,
                "finished": False
            }]

            for step in range(max_len):
                new_beams = []

                for beam in beams:
                    if beam["finished"]:
                        new_beams.append(beam)
                        continue

                    last_token = beam["tokens"][-1]
                    input_token = tensor([[last_token]], device=accelerator)

                    logits, (hn_new, cn_new) = self._decoder(input_token, beam["hidden"])
                    probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

                    top_k_probs, top_k_indices = topk(probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        token = top_k_indices[0, i].item()
                        token_prob = max(top_k_probs[0, i].item(), 1e-10)

                        new_beam = {
                            "tokens": beam["tokens"] + [token],
                            "score": beam["score"] + log(token_prob + 1e-10),
                            "hidden": (hn_new, cn_new),
                            "finished": (token == self._EOS)
                        }
                        new_beams.append(new_beam)

                beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]

                if all(beam["finished"] for beam in beams):
                    break

            best_beam = beams[0]
            result_tokens = best_beam["tokens"][1:]
            result_tensor = tensor(result_tokens, device=accelerator)
            results.append(result_tensor)

        return nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=self._EOS)


if __name__ == "__main__":
    options: list[SeqMergeMethods] = [
        SeqMergeMethods.CONCAT,
        SeqMergeMethods.MAX,
        SeqMergeMethods.MEAN,
        SeqMergeMethods.SUM
    ]

    model = SeqToSeqLSTM(
        vocab_size_src=5000,
        vocab_size_tgt=6000,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        bidirectional=True,
        merge_method=choice(options)
    )

    model.summary()

    batch_size = 4
    src_len = 10
    tgt_len = 8

    src = randint(3, 5000, (batch_size, src_len))
    tgt = cat([
        full((batch_size, 1), model._SOS),
        randint(3, 6000, (batch_size, tgt_len - 2)),
        full((batch_size, 1), model._EOS)
    ], dim=1)

    logits = model(src, tgt)
    print(f"Forward test passed! Logits shape: {logits.shape}")
