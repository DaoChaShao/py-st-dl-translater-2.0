#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/16 16:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq_attn_rnn.py
# @Desc     :   

from torch import (Tensor, nn,
                   cat,
                   device, full, long, ones, bool as torch_bool, where, full_like, empty,
                   tensor, topk, log,
                   randint)
from typing import override, final

# Attention imports (if needed in future extensions)
from src.nets.attention_multi import MultiHeadAttention
from src.nets.attention_single import SingleHeadAttention
# ---

from src.configs.cfg_types import Attentions, SeqMergeMethods
from src.nets.base_seq import BaseSeqNet
from src.nets.seq_encoder import SeqEncoder
from src.nets.seq_decoder import SeqDecoder


class AttentionRNNForSeqToSeq(BaseSeqNet):
    """ Sequence-to-Sequence RNN Network for Sequence Tasks """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str = "cpu",
                 PAD_SRC: int = 0, PAD_TGT: int = 0, SOS: int = 2, EOS: int = 3,
                 *,
                 merge_method: str | SeqMergeMethods = "average",
                 # Attentional parameters (not used in GRU but reserved for future use)
                 use_attention: bool = False,
                 attention_method: Attentions = "dot",
                 attention_type: str = "single",
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
        self._attention_method: Attentions = attention_method
        self._attention_type: str = attention_type
        # ---

        # Initialize encoder and decoder
        self._encoder: nn.Module = self.init_encoder()
        self._decoder: nn.Module = self.init_decoder()

        # Initialize attention module if needed
        if self._use_attention:
            self._init_attention(head_num)
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
            PAD=self._PAD_SRC, net_category="rnn",
        )

    @override
    def init_decoder(self) -> nn.Module:
        """ Initialize the decoder module
        :return: decoder module
        """
        hidden_size = self._M * 2 if (self._bid and self._method == "concat") else self._M
        return SeqDecoder(
            self._vocab_tgt, self._H, hidden_size, self._C,
            dropout_rate=self._dropout, bidirectional=False,
            accelerator=self._accelerator,
            PAD=self._PAD_TGT, net_category="rnn",
        )

    @final
    def _init_attention(self, head_num: int) -> None:
        """ Initialize the attention module
        :param head_num: number of heads for multi-head attention
        """
        encoder_hidden_size = self._M * 2 if self._bid else self._M

        match self._attention_type:
            case "single":
                self._attention = SingleHeadAttention(
                    hidden_size=encoder_hidden_size,
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
    def _merge_bidirectional_hidden(self, hidden: Tensor | tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor]:
        """ Merge bidirectional hidden states for decoder initialization
        :param hidden: hidden states from the encoder
        :return: merged hidden states
        """
        if not self._bid:
            return hidden

        # Returns a single Tensor
        batch_size = hidden.size(1)
        hidden = hidden.view(self._C, self._num_directions, batch_size, self._M)

        if self._method == "average":
            return hidden.mean(dim=1)
        elif self._method == "concat":
            return cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        else:
            raise ValueError(f"Unknown merge method: {self._method}")

    @final
    def _forward_with_attention(self, decoder_input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """ Forward pass with attention mechanism """
        tgt_len: int = decoder_input.size(1)
        outputs = []

        for t in range(tgt_len):
            input_step = decoder_input[:, t].unsqueeze(1)

            # Decoder step
            decoder_output, decoder_hidden = self._decoder(input_step, decoder_hidden)

            # Attention step
            combined = None
            match self._attention_type:
                case "single":
                    attn_weights, context = self._attention(decoder_hidden, encoder_outputs)
                    # Combine decoder output and context
                    combined = cat([decoder_output, context.unsqueeze(1)], dim=-1)
                case "multi":
                    # Transpose for attention computation
                    query = decoder_hidden.transpose(0, 1)  # [batch_size, 1, hidden_size]
                    key = value = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, hidden_size]

                    context, _ = self._attention(query, key, value)
                    combined = cat([decoder_output, context], dim=-1)

            outputs.append(combined)

        return cat(outputs, dim=1)

    @override
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass through the Seq2Seq model
        :param src: source/input tensor
        :param tgt: target/output tensor
        :return: output logits tensor
        """
        # Encode
        encoder_outputs, encoder_hidden, src_lengths = self._encoder(src)
        # Combine bidirectional hidden states
        decoder_hidden = self._merge_bidirectional_hidden(encoder_hidden)
        # Decode input excludes the EOS token
        decoder_input = tgt[:, :-1]  # Remove EOS token for decoder input

        # Add attention mechanism if enabled (not implemented in this GRU version)
        if self._use_attention:
            return self._forward_with_attention(decoder_input, decoder_hidden, encoder_outputs)
        # ---
        else:
            logits, _ = self._decoder(decoder_input, decoder_hidden)

        return logits

    @override
    def generate(self, src: Tensor, max_len: int = 100, strategy: str = "greedy", beam_width: int = 5) -> Tensor:
        """ Generate sequences
        :param src: source/input tensor
        :param max_len: maximum length of generated sequences
        :param strategy: generation strategy ('greedy' or 'beam')
        :param beam_width: beam width for beam search
        :return: generated sequences tensor
        """
        batch_size = src.size(0)

        # Encoder
        encoder_outputs, encoder_hidden, lengths = self._encoder(src)

        # Combine bidirectional hidden states
        decoder_hidden = self._merge_bidirectional_hidden(encoder_hidden)

        match strategy:
            case "greedy":
                return self._greedy_decode(decoder_hidden, batch_size, max_len, src.device)
            case "beam":
                return self._beam_search_decode(decoder_hidden, batch_size, max_len, beam_width, src.device)
            case _:
                raise ValueError(f"Unknown generation strategy: {strategy}")

    @override
    def _greedy_decode(self,
                       decoder_hidden: Tensor | tuple[Tensor, Tensor],
                       batch_size: int, max_len: int, accelerator: device
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

            logits, decoder_hidden = self._decoder(decoder_input, decoder_hidden)

            next_token = logits.argmax(dim=2)
            next_token = where(active.unsqueeze(1), next_token, full_like(next_token, self._EOS))

            generated.append(next_token)
            active = active & (next_token.squeeze(1) != self._EOS)
            decoder_input = next_token

        return cat(generated, dim=1) if generated else empty((batch_size, 0), dtype=long, device=accelerator)

    @override
    def _beam_search_decode(self,
                            decoder_hidden: Tensor | tuple[Tensor, Tensor],
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
        results = []

        for idx in range(batch_size):
            # Get hidden state of a single example
            batch_hidden = decoder_hidden[:, idx:idx + 1]

            # Initialize beams
            beams = [{
                "tokens": [self._SOS],
                "score": tensor(0.0, device=accelerator),
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

                    logits, new_hidden = self._decoder(input_token, beam["hidden"])
                    probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

                    top_k_probs, top_k_indices = topk(probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        token = top_k_indices[0, i].item()
                        token_prob = max(top_k_probs[0, i].item(), 1e-10)

                        new_beam = {
                            "tokens": beam["tokens"] + [token],
                            "score": beam["score"] + log(tensor(token_prob + 1e-10, device=accelerator)),
                            "hidden": new_hidden,
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
    model = AttentionRNNForSeqToSeq(
        vocab_size_src=5000,
        vocab_size_tgt=6000,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        bidirectional=True,
        merge_method="average"
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
