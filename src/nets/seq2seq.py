#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/11 16:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq.py
# @Desc     :   

from math import log
from torch import (nn, Tensor, tensor,
                   full, long, cat,
                   randint, ones, bool as torch_bool, where, full_like, empty,
                   topk)
from torch.xpu import device
from typing import final, Final, Literal

from src.configs.cfg_types import SeqNets, SeqStrategies
from src.utils.highlighter import starts, lines

WIDTH: int = 64


class SeqEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 PAD: int = 0,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = "gru",
                 ):
        super().__init__()
        """ Initialise the Encoder class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param PAD: padding index for the embedding layer
        :param net_category: network category (e.g., 'gru')
        """
        self._L: int = vocab_size  # Lexicon/Vocabulary size for encoder / input
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._type: str | SeqNets = net_category.lower()  # Network category

        self._embed = nn.Embedding(self._L, self._H, padding_idx=PAD)
        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

    @staticmethod
    def _select_net(net_category: str) -> type:
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if net_category not in nets:
            raise ValueError(f"Unknown network category: {net_category}")

        return nets[net_category]

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        embeddings = self._embed(src)
        lengths: Tensor = (src != self._embed.padding_idx).sum(dim=1)

        result = self._net(embeddings)

        if self._type == "lstm":
            outputs, (hidden, cell) = result
            return outputs, (hidden, cell), lengths
        else:
            outputs, hidden = result
            return outputs, hidden, lengths


class SeqDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = False,
                 PAD: int = 0,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = "gru",
                 ):
        super().__init__()
        """ Initialise the Decoder class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param pad_idx: padding index for the embedding layer
        :param net_category: network category (e.g., 'gru')
        """
        self._L: int = vocab_size  # Lexicon/Vocabulary size
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._bid: bool = bidirectional
        self._type: str | SeqNets = net_category.lower()  # Network category

        self._embed = nn.Embedding(self._L, self._H, padding_idx=PAD)
        self._net = self._select_net(self._type)(
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=self._bid,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self._dropout = nn.Dropout(p=dropout_rate if num_layers > 1 else 0.0)
        self._linear = nn.Linear(self._M, self._L)

    @staticmethod
    def _select_net(net_category: str) -> type:
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if net_category not in nets:
            raise ValueError(f"Unknown network category: {net_category}")

        return nets[net_category]

    def forward(self, tgt: Tensor, hidden: Tensor | tuple[Tensor, Tensor]) -> tuple:
        embeddings = self._embed(tgt)

        if self._type == "lstm":
            outputs, (hn, cn) = self._net(embeddings, hidden)
            logits = self._linear(self._dropout(outputs))
            return logits, (hn, cn)
        else:
            h = hidden[0] if isinstance(hidden, tuple) else hidden
            outputs, hn = self._net(embeddings, h)
            logits = self._linear(self._dropout(outputs))
            return logits, (hn,)


class SeqToSeqCoder(nn.Module):
    """ An RNN model for sequence-to-sequence tasks using PyTorch """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int,
                 embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 PAD_SRC: int = 0, PAD_TGT: int = 0,
                 net_category: str | SeqNets | Literal["gru", "lstm", "rnn"] = "gru",
                 SOS: int = 2, EOS: int = 3,
                 ):
        super().__init__()
        """ Initialise the SeqToSeqRNN class
        :param vocab_size_src: size of the input vocabulary
        :param vocab_size_tgt: size of the output vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag for the encoder
        :param PAD_SRC: padding index for the input embedding layer
        :param PAD_TGT: padding index for the output embedding layer
        :param accelerator: computation accelerator (e.g., 'cpu', 'cuda')
        :param net_category: network category (e.g., 'gru')
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        """
        self._size_src: int = vocab_size_src  # Lexicon/Vocabulary size for encoder / input
        self._size_tgt: int = vocab_size_tgt  # Lexicon/Vocabulary size for decoder / output
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._bid: bool = bidirectional  # Bidirectional flag for encoder
        self._type: str | SeqNets = net_category.lower()  # Network category
        self._SOS: Final[int] = SOS  # Start-of-Sequence token index
        self._EOS: Final[int] = EOS  # End-of-Sequence token index

        self._encoder = SeqEncoder(
            self._size_src, self._H, self._M, self._C,
            dropout_rate=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=self._bid, PAD=PAD_SRC, net_category=net_category,
        )
        self._decoder = SeqDecoder(
            self._size_tgt, self._H, self._M, self._C,
            dropout_rate=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=False, PAD=PAD_TGT, net_category=net_category,
        )

        self._init_weights()

    def _init_weights(self):
        """ Initialize model weights """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass through the Seq2Seq model
        :param src: source/input tensor
        :param tgt: target/output tensor
        :return: output logits tensor
        """
        c = None

        # Encoder
        if self._type == "lstm":
            outputs, (h, c), lengths = self._encoder(src)
        else:
            outputs, h, lengths = self._encoder(src)

        # Decoder
        if self._bid:
            B = h.size(1)
            if self._type == "lstm":
                h = h.view(self._C, 2, B, self._M)
                c = c.view(self._C, 2, B, self._M)
                decoder_state = ((h[:, 0] + h[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2)
            else:
                h = h.view(self._C, 2, B, self._M)
                decoder_state = ((h[:, 0] + h[:, 1]) / 2,)
        else:
            decoder_state = (h, c) if self._type == "lstm" else (h,)

        decoder_input = tgt[:, :-1]
        logits, _ = self._decoder(decoder_input, decoder_state)

        return logits

    def summary(self):
        """ Print a summary of the model architecture and parameters """
        print("*" * WIDTH)
        print(f"Model: {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"Encoder Vocab Size: {self._size_src}")
        print(f"Decoder Vocab Size: {self._size_tgt}")
        print(f"Embedding Dim: {self._H}")
        print(f"Hidden Size: {self._M}")
        print(f"Num Layers: {self._C}")
        print(f"Bidirectional Encoder: {self._bid}")
        print(f"RNN Type: {self._type}")
        print("-" * WIDTH)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("*" * WIDTH)

    def generate(self,
                 src: Tensor, max_len: int = 100,
                 strategy: str | SeqStrategies | Literal["greedy", "beam"] = "greedy", beam_width: int = 5
                 ) -> Tensor:
        """ Regression generate automatically
        :param src: source/input tensor
        :param max_len: maximum length of the generated sequence
        :param strategy: generation strategy ("greedy" or beam")
        :param beam_width: beam width for beam search (if applicable)
        :return: generated sequence tensor
        """
        batch_size = src.size(0)

        # Encoder
        encoder_outputs, encoder_hidden, lengths = self._encoder(src)

        match strategy:
            case "greedy":
                return self._greedy_decode(encoder_hidden, batch_size, max_len, src.device)
            case "beam":
                return self._beam_search_decode(encoder_hidden, batch_size, max_len, beam_width, src.device)
            case _:
                raise ValueError(f"Unknown generation strategy: {strategy}")

    def _greedy_decode(self,
                       encoder_hidden: tuple | Tensor, batch_size: int, max_len: int, accelerator: device
                       ) -> Tensor:
        """ Greedy decoding implementation
        :param encoder_hidden: encoder hidden state
        :param batch_size: batch size
        :param max_len: maximum length of the generated sequence
        :param accelerator: device of the source tensor
        :return: generated sequence tensor
        """
        # Decoder state initialization
        if self._type == "lstm":
            # LSTM: encoder_hidden is tuple (h, c)
            h, c = encoder_hidden
            if self._bid:
                B = h.size(1)
                h = h.view(self._C, 2, B, self._M)
                c = c.view(self._C, 2, B, self._M)
                decoder_hidden = ((h[:, 0] + h[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2)
            else:
                decoder_hidden = (h, c)
        else:
            # GRU/RNN: encoder_hidden is Tensor
            h = encoder_hidden
            if self._bid:
                B = h.size(1)
                h = h.view(self._C, 2, B, self._M)
                decoder_hidden = ((h[:, 0] + h[:, 1]) / 2,)  # tuple
            else:
                decoder_hidden = (h,)  # tuple

        # Start from SOS token (assumed to be index 1)
        decoder_input = full((batch_size, 1), self._SOS, dtype=long, device=accelerator)
        generated = []

        # Track which sequences are still active, sequences mask
        active = ones(batch_size, dtype=torch_bool, device=accelerator)

        for step in range(max_len):
            if not active.any():  # All sequences have finished
                break

            logits, decoder_hidden = self._decoder(decoder_input, decoder_hidden)

            next_token = logits.argmax(dim=2)
            next_token = where(active.unsqueeze(1), next_token, full_like(next_token, self._EOS))

            # Only update active sequences
            generated.append(next_token)

            # Update active mask
            active = active & (next_token.squeeze(1) != self._EOS)

            # Prepare next input
            decoder_input = next_token

        return cat(generated, dim=1) if generated else empty((batch_size, 0), dtype=long, device=accelerator)

    def _beam_search_decode(self,
                            encoder_hidden: tuple | Tensor,
                            batch_size: int, max_len: int, beam_width: int, accelerator: device
                            ) -> Tensor:
        """ Beam search decoding implementation
        :param encoder_hidden: encoder hidden state
        :param batch_size: batch size
        :param max_len: maximum length of the generated sequence
        :param beam_width: beam width for beam search
        :param accelerator: device of the source tensor
        :return: generated sequence tensor
        """
        # Decoder state initialization
        if self._type == "lstm":
            h, c = encoder_hidden
            if self._bid:
                B = h.size(1)
                h = h.view(self._C, 2, B, self._M)
                c = c.view(self._C, 2, B, self._M)
                initial_hidden = ((h[:, 0] + h[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2)
            else:
                initial_hidden = (h, c)
        else:
            h = encoder_hidden
            if self._bid:
                B = h.size(1)
                h = h.view(self._C, 2, B, self._M)
                initial_hidden = ((h[:, 0] + h[:, 1]) / 2,)
            else:
                initial_hidden = (h,)

        # Set beam search variables for each sample in the batch
        results = []

        for idx in range(batch_size):
            if isinstance(initial_hidden, tuple):
                if len(initial_hidden) == 2:
                    # LSTM
                    h_batch = initial_hidden[0][:, idx:idx + 1]
                    c_batch = initial_hidden[1][:, idx:idx + 1]
                    batch_hidden = (h_batch, c_batch)
                else:
                    # GRU
                    batch_hidden = (initial_hidden[0][:, idx:idx + 1],)
            else:
                batch_hidden = initial_hidden[:, idx:idx + 1]

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

                    # Get last token and prepare input
                    last_token = beam["tokens"][-1]
                    input_token = tensor([[last_token]], device=accelerator)

                    # Decode step
                    logits, new_hidden = self._decoder(input_token, beam["hidden"])
                    probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

                    # Get top k candidates（k = beam_width）
                    top_k_probs, top_k_indices = topk(probs, beam_width, dim=-1)

                    for i in range(beam_width):
                        token = top_k_indices[0, i].item()
                        token_prob = top_k_probs[0, i].item()

                        new_beam = {
                            "tokens": beam["tokens"] + [token],
                            "score": beam["score"] + log(token_prob + 1e-10),  # Probability to log-probability
                            "hidden": new_hidden,
                            "finished": (token == self._EOS)
                        }
                        new_beams.append(new_beam)

                # Sort and select top beams
                beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]

                # Stop if all beams are finished
                if all(beam["finished"] for beam in beams):
                    break

            # Select the best beam
            best_beam = beams[0]
            # Collect result tokens excluding SOS
            result_tokens = best_beam["tokens"][1:]  # Exclude SOS token
            result_tensor = tensor(result_tokens, device=accelerator)

            results.append(result_tensor)

        return nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=self._EOS)


if __name__ == "__main__":
    test_cases = [
        ("gru", True, "GRU-bid"), ("gru", False, "GRU-one"),
        ("lstm", True, "LSTM-bid"), ("lstm", False, "LSTM-one"),
        ("rnn", True, "RNN-bid"), ("rnn", False, "RNN-one"),
    ]

    for rnn_type, bid, desc in test_cases:
        starts()
        print(f"Test: {desc}")
        lines()

        model = SeqToSeqCoder(
            vocab_size_src=5000,
            vocab_size_tgt=6000,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2,
            bidirectional=bid,
            net_category=rnn_type
        )

        src = randint(3, 5000, (3, 8))
        tgt = cat([
            full((3, 1), 1),
            randint(3, 6000, (3, 7)),
            full((3, 1), 2)
        ], dim=1)

        # Forward pass test
        logits = model(src, tgt)
        assert logits.shape == (3, 8, 6000), f"Logits Error Size: {logits.shape}"

        # Generation test
        # generated = model.generate(src, max_len=10, strategy="greedy")
        generated = model.generate(src, max_len=10, strategy="beam", beam_width=3)
        assert generated.shape[0] == 3, f"Generation Batch Error: {generated.shape}"
        assert generated.shape[1] <= 10, f"Generation Length Error: {generated.shape}"

        print(f"Successfully!")
        print(f"- Logits Shape: {logits.shape}")
        print(f"- Generation Shape: {generated.shape}")
        starts()
        print()
