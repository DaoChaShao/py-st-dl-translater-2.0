#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 16:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_attn_gru.py
# @Desc     :   

from math import log
from torch import (Tensor, nn,
                   cat,
                   device, full, long, ones, bool as torch_bool, where, full_like, empty,
                   tensor, topk,
                   randint)
from typing import override, final, Literal

# Attention imports (if needed in future extensions)
from src.nets.attention_multi import MultiHeadAttention
from src.nets.attention_single import SingleHeadAttention
# ---

from src.configs.cfg_types import AttnScorer, SeqMergeMethods
from src.nets.base_seq import BaseSeqNet
from src.nets.seq_encoder import SeqEncoder
from src.nets.seq_decoder import SeqDecoder
from src.utils.PT import TorchRandomSeed
from src.utils.highlighter import starts, lines


class AttnGRUForSeqToSeq(BaseSeqNet):
    """ Sequence-to-Sequence GRU Network for Sequence Tasks """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD_SRC: int = 0, PAD_TGT: int = 0, SOS: int = 2, EOS: int = 3,
                 *,
                 merge_method: str | SeqMergeMethods | Literal["concat", "max", "mean", "sum"] = "mean",
                 # Attentional parameters (not used in GRU but reserved for future use)
                 use_attention: bool = False,
                 attention_method: str | AttnScorer | Literal["dot", "general", "concat"] = "dot",
                 attention_type: str | Literal["single", "multi"] = "single",
                 head_num: int = 8,
                 # ---
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
        """ Initialize the SeqToSeqTaskGRU class
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
        :param merge_method: method to merge bidirectional hidden states ('average' or 'concat')
        ---
        :param use_attention: whether to use attention mechanism
        :param attention_method: attention method ('dot', 'general', 'concat')
        :param attention_type: type of attention ('single' or 'multi')
        :param head_num: number of heads for multi-head attention
        """
        self._method: str = merge_method

        # Attention parameters (not used in GRU but reserved for future use)
        self._use_attention: bool = use_attention
        self._attention_method: str | AttnScorer = attention_method.lower()
        # self._attention_method: str = "concat"
        self._attention_type: str = attention_type.lower()
        # ---

        # Initialize encoder and decoder
        self._encoder: nn.Module = self.init_encoder()
        self._decoder: nn.Module = self.init_decoder()

        # Initialize attention module if needed
        if self._use_attention:
            self._init_attention(head_num)
            # Adjust output projection layer to account for attention context
            self._projection = nn.Linear(
                self._vocab_tgt + (self._M * self._num_directions if self._bid else self._M),
                self._vocab_tgt
            )
        # ---

        # Initialize weights
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
            PAD=self._PAD_SRC, net_category="gru",
        )

    @override
    def init_decoder(self) -> nn.Module:
        """ Initialize the decoder module
        :return: decoder module
        """
        hidden_size = self._M * self._num_directions if self._bid and self._method == "concat" else self._M

        return SeqDecoder(
            self._vocab_tgt, self._H, hidden_size, self._C,
            dropout_rate=self._dropout, bidirectional=False,
            accelerator=self._accelerator,
            PAD=self._PAD_TGT, net_category="gru",
        )

    @final
    def _init_attention(self, head_num: int) -> None:
        """ Initialize the attention module
        :param head_num: number of heads for multi-head attention
        """
        encoder_hidden_size = self._M * self._num_directions if self._bid else self._M
        decoder_hidden_size = self._M

        starts()
        print(f"Attention Module Initialization:")
        lines()
        print(f"- bidirectional:       {self._bid}")
        print(f"- M (hidden_size):     {self._M}")
        print(f"- num_directions:      {self._num_directions}")
        print(f"- encoder_hidden_size: {encoder_hidden_size}")
        print(f"- decoder_hidden_size: {decoder_hidden_size}")
        print(f"- attention_type:      {self._attention_type}")
        print(f"- attention_method:    {self._attention_method}")
        starts()
        print()

        match self._attention_type:
            case "single":
                self._attention = SingleHeadAttention(
                    encoder_hidden_size=encoder_hidden_size,
                    decoder_hidden_size=decoder_hidden_size,
                    method=self._attention_method
                )
            case "multi":
                self._attention = MultiHeadAttention(
                    hidden_size=encoder_hidden_size,
                    num_heads=head_num,
                    dropout=self._dropout
                )
            case _:
                raise ValueError(f"Unknown attention type: {self._attention_type}")

    @override
    def _merge_bidirectional_hidden(self, hidden_src: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """ Merge bidirectional hidden states for decoder initialization
        :param hidden: hidden states from the encoder
        :return: merged hidden states
        """
        return self._decoder.init_decoder_entries(hidden_src, merge_method=self._method)

    @override
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass through the Seq2Seq model
        :param src: source/input tensor
        :param tgt: target/output tensor
        :return: output logits tensor
        """
        # Encode
        outputs_src, (hidden_src, _), lengths_src = self._encoder(src)

        # Combine bidirectional hidden states for decoder init
        hidden_ety = self._merge_bidirectional_hidden(hidden_src)

        # Decoder input excludes EOS token
        tgt_ety = tgt[:, :-1]

        if self._use_attention:
            return self._forward_with_attention(tgt_ety, hidden_ety, outputs_src)
        else:
            logits, _ = self._decoder(tgt_ety, hidden_ety)
            return logits

    @final
    def _forward_with_attention(self, decoder_input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """ Forward pass with attention mechanism """
        tgt_len: int = decoder_input.size(1)
        outputs: list[Tensor] = []

        for i in range(tgt_len):
            input_step = decoder_input[:, i].unsqueeze(1)  # [B, 1]

            # Decoder step
            output_tgt, (hn_tgt, _) = self._decoder(input_step, decoder_hidden)

            # Take last layer hidden for attention
            hidden_attn = hn_tgt[-1:].contiguous()  # [1, B, H]

            if self._attention_type == "single":
                attn_weights, context = self._attention(hidden_attn, encoder_outputs)
                # combine decoder output and context
                combined = cat([output_tgt, context.unsqueeze(1)], dim=-1)
                output = self._projection(combined)  # [B, 1, vocab_tgt]
            elif self._attention_type == "multi":
                query = hidden_attn.transpose(0, 1)  # [B, 1, H]
                key = value = encoder_outputs.transpose(0, 1)  # [B, S, H]
                context, _ = self._attention(query, key, value)
                combined = cat([output_tgt, context], dim=-1)
                output = self._projection(combined)
            else:
                raise ValueError(f"Unknown attention type: {self._attention_type}")

            outputs.append(output)

        return cat(outputs, dim=1)  # [B, tgt_len, H + context_size]

    @override
    def generate(self, src: Tensor, max_len: int = 100, strategy: str = "greedy", beam_width: int = 5) -> Tensor:
        """ Generate sequences with optional attention """
        batches: int = src.size(0)

        # Encode
        outputs_src, (hidden_src, _), lengths_src = self._encoder(src)

        # Combine bidirectional hidden states
        hidden_ety = self._merge_bidirectional_hidden(hidden_src)

        match strategy:
            case "greedy":
                return self._greedy_decode(
                    hidden_ety, batches, max_len, src.device, outputs_src if self._use_attention else None
                )
            case "beam":
                return self._beam_search_decode(
                    hidden_ety, batches, max_len, beam_width, src.device,
                    outputs_src if self._use_attention else None
                )
            case _:
                raise ValueError(f"Unknown generation strategy: {strategy}")

    @override
    def _greedy_decode(self,
                       decoder_hidden: Tensor,
                       batch_size: int, max_len: int, accelerator: device,
                       encoder_outputs: Tensor | None = None
                       ) -> Tensor:
        """ Greedy decoding with optional attention """
        decoder_input = full((batch_size, 1), self._SOS, dtype=long, device=accelerator)
        generated: list[Tensor] = []

        active = ones(batch_size, dtype=torch_bool, device=accelerator)

        logits: Tensor | None = None
        for _ in range(max_len):
            if not active.any():
                break

            output_tgt, (hn_tgt, _) = self._decoder(decoder_input, decoder_hidden)

            if self._use_attention and encoder_outputs is not None:
                hidden_attn = hn_tgt[-1:].contiguous()
                if self._attention_type == "single":
                    attn_weights, context = self._attention(hidden_attn, encoder_outputs)
                    combined = cat([output_tgt, context.unsqueeze(1)], dim=-1)
                    logits: Tensor = self._projection(combined)
                elif self._attention_type == "multi":
                    query = hidden_attn.transpose(0, 1)
                    key = value = encoder_outputs.transpose(0, 1)
                    context, _ = self._attention(query, key, value)
                    combined = cat([output_tgt, context], dim=-1)
                    logits: Tensor = self._projection(combined)
            else:
                logits: Tensor = output_tgt

            # Select the token with the highest probability
            next_token = logits.argmax(dim=2)
            next_token = where(active.unsqueeze(1), next_token, full_like(next_token, self._EOS))
            generated.append(next_token)

            # Update active sequences
            active = active & (next_token.squeeze(1) != self._EOS)
            decoder_input = next_token

            # Update decoder hidden
            decoder_hidden = hn_tgt

        return cat(generated, dim=1) if generated else empty((batch_size, 0), dtype=long, device=accelerator)

    @override
    def _beam_search_decode(self,
                            decoder_hidden: Tensor,
                            batch_size: int, max_len: int, beam_width: int, accelerator: device,
                            encoder_outputs: Tensor = None
                            ) -> Tensor:
        """ Beam search decoding with optional attention """
        results: list[Tensor] = []

        # Initialise beams
        for b in range(batch_size):
            batch_hidden = decoder_hidden[:, b:b + 1]
            beams = [{
                "tokens": [self._SOS],
                "score": 0.0,
                "hidden": batch_hidden,
                "finished": False
            }]

            logits: Tensor | None = None
            for _ in range(max_len):
                new_beams = []
                for beam in beams:
                    if beam["finished"]:
                        new_beams.append(beam)
                        continue

                    last_token = beam["tokens"][-1]
                    input_token = tensor([[last_token]], device=accelerator)
                    # Decoder step
                    outputs_tgt, (hn_tgt, _) = self._decoder(input_token, beam["hidden"])

                    if self._use_attention and encoder_outputs is not None:
                        # Get last layer hidden state for the current batch
                        hidden_base = hn_tgt[-1].contiguous()  # [1, decoder_hidden_size]
                        # Expand hidden state for attention avoiding generation 1d tensor error
                        batch_expand = 2  # 可以是任意大于1的数
                        hidden_attn = hidden_base.expand(batch_expand, hidden_base.size(-1))

                        # Get encoder outputs for the current batch [1, src_len, encoder_hidden_size]
                        encoder_slice = encoder_outputs[:, b:b + 1].transpose(0, 1)
                        # Expand encoder outputs for matching attention hidden batch size
                        encoder_slice = encoder_slice.expand(
                            batch_expand,
                            encoder_slice.size(1),  # src_len
                            encoder_slice.size(2)  # encoder_hidden_size
                        )

                        if self._attention_type == "single":
                            attn_weights, context = self._attention(
                                hidden_attn,  # [batch_expand, decoder_hidden_size]
                                encoder_slice  # [batch_expand, src_len, encoder_hidden_size]
                            )
                            combined = cat([outputs_tgt, context[0:1].unsqueeze(1)], dim=-1)
                            logits: Tensor = self._projection(combined)
                        elif self._attention_type == "multi":
                            query = hidden_attn.transpose(0, 1)
                            key = value = encoder_outputs[:, b:b + 1].transpose(0, 1)
                            context, _ = self._attention(query, key, value)
                            combined = cat([outputs_tgt, context], dim=-1)
                            logits: Tensor = self._projection(combined)
                    else:
                        logits: Tensor = outputs_tgt

                    probs = nn.functional.softmax(logits[:, -1, :], dim=-1)
                    top_k_probs, top_k_indices = topk(probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        token = top_k_indices[0, i].item()
                        token_prob = max(top_k_probs[0, i].item(), 1e-10)
                        new_beam = {
                            "tokens": beam["tokens"] + [token],
                            "score": beam["score"] + log(token_prob + 1e-10),
                            "hidden": hn_tgt,
                            "finished": (token == self._EOS)
                        }
                        new_beams.append(new_beam)

                # Keep top beam_width beams
                beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]
                # Early stop if all beams finished
                if all(beam["finished"] for beam in beams):
                    break

            # Take best beam
            best_beam = beams[0]
            # Exclude SOS token
            result_tokens = best_beam["tokens"][1:]
            results.append(tensor(result_tokens, device=accelerator))

        # Pad sequences to the same length
        return nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=self._EOS)


if __name__ == "__main__":
    with TorchRandomSeed("Test"):
        model = AttnGRUForSeqToSeq(
            vocab_size_src=80,
            vocab_size_tgt=100,
            embedding_dim=65,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            merge_method="mean",
            use_attention=True,
            attention_method="concat",
            attention_type="single",
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
