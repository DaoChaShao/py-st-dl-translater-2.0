#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/15 17:37
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_seq2seq.py
# @Desc     :   

from abc import ABC, abstractmethod
from pathlib import Path
from torch import (Tensor, device, nn,
                   zeros,
                   save, load)
from typing import final, Literal, Final

WIDTH: int = 64


class BaseSeqNet(ABC, nn.Module):
    """ Abstract Base Class for Sequence-based Networks """

    def __init__(self,
                 vocab_size_src: int, vocab_size_tgt: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 *,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD_SRC: int = 0, PAD_TGT: int = 0, SOS: int = 2, EOS: int = 3,
                 ) -> None:
        super().__init__()
        """ Initialize the BaseSeq class
        :param vocab_size_src: size of the input vocabulary
        :param vocab_size_tgt: size of the output vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD_SRC: padding index for the input embedding layer
        :param PAD_TGT: padding index for the output embedding layer
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        """
        self._vocab_src: int = vocab_size_src
        self._vocab_tgt: int = vocab_size_tgt
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count
        self._dropout: float = dropout_rate
        self._bid: bool = bidirectional
        self._accelerator: str = accelerator.lower()
        self._PAD_SRC: Final[int] = PAD_SRC
        self._PAD_TGT: Final[int] = PAD_TGT
        self._SOS: Final[int] = SOS
        self._EOS: Final[int] = EOS

        self._num_directions: int = self._set_num_directions(self._bid)

        self._encoder = None
        self._decoder = None

    @staticmethod
    def _set_num_directions(bid: bool) -> int:
        """ Set the num_directions based on bidirectionality
        :param bid: bidirectional flag
        :return: number of directions (1 for unidirectional, 2 for bidirectional)
        """
        return 2 if bid else 1

    @property
    def source_vocab_size(self) -> int:
        """ Get the source vocabulary size
        :return: source vocabulary size
        """
        return self._vocab_src

    @property
    def target_vocab_size(self) -> int:
        """ Get the target vocabulary size
        :return: target vocabulary size
        """
        return self._vocab_tgt

    @property
    def embedding_dim(self) -> int:
        """ Get the embedding dimension
        :return: embedding dimension
        """
        return self._H

    @property
    def hidden_size(self) -> int:
        """ Get the hidden size
        :return: hidden size
        """
        return self._M

    @property
    def num_layers(self) -> int:
        """ Get the number of layers
        :return: number of layers
        """
        return self._C

    def init_weights(self) -> None:
        """ Initialize model weights with appropriate schemes """
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if param.dim() == 2 and any(key in name for key in ["embed", "embedder", ]):
                nn.init.normal_(param, mean=0.0, std=0.01)

            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

            elif "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @final
    def init_hidden(self, batch_size: int) -> Tensor:
        """ Initialize h0
        :param batch_size: size of the batch
        :return: h0 and c0 tensors
        """
        shape: tuple[int, int, int] = (self._C * self._num_directions, batch_size, self._M)
        h0: Tensor = zeros(shape, device=device(self._accelerator))
        return h0

    @final
    def init_lstm_hidden(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """ Initialize h0 and c0
        :param batch_size: size of the batch
        :param accelerator: device for PyTorch
        :return: h0 and c0 tensors
        """
        shape: tuple[int, int, int] = (self._C * self._num_directions, batch_size, self._M)
        h0: Tensor = zeros(shape, device=device(self._accelerator))
        c0: Tensor = zeros(shape, device=device(self._accelerator))
        return h0, c0

    @abstractmethod
    def init_encoder(self) -> None:
        """ Set the encoder module """
        pass

    @abstractmethod
    def init_decoder(self) -> None:
        """ Set the decoder module """
        pass

    @abstractmethod
    def _merge_bidirectional_hidden(self, enc_hn: Tensor) -> Tensor:
        """ Merge bidirectional hidden states for decoder initialization
        :param enc_hn: hidden state tensor(s)
        :return: merged hidden state tensor(s)
        """
        pass

    @abstractmethod
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass of the model
        :param src: source tensor, shape (batch_size, src_sequence_length)
        :param tgt: target tensor, shape (batch_size, tgt_sequence_length)
        :return: output tensor, shape (batch_size, output_sequence_length)
        """
        pass

    @abstractmethod
    def generate(self, src: Tensor, max_len: int, strategy: str, beam_width: int) -> Tensor:
        """ Regression generate automatically
        :param src: source/input tensor
        :param max_len: maximum length of the generated sequence
        :param strategy: generation strategy ("greedy" or beam")
        :param beam_width: beam width for beam search (if applicable)
        :return: generated sequence tensor
        """
        pass

    @abstractmethod
    def _greedy_decode(self,
                       dec_hn: Tensor,
                       batch_size: int, max_len: int, accelerator: device,
                       ) -> Tensor:
        """ Greedy decoding strategy
        :param dec_hn: initial hidden state for the decoder
        :param batch_size: size of the batch
        :param max_len: maximum length of the generated sequence
        :param accelerator: device for PyTorch
        :return: generated sequence tensor
        """
        pass

    @abstractmethod
    def _beam_search_decode(self,
                            dec_hn: Tensor,
                            batch_size: int, max_len: int, beam_width: int, accelerator: device
                            ) -> Tensor:
        """ Beam search decoding strategy
        :param dec_hn: initial hidden state for the decoder
        :param batch_size: size of the batch
        :param max_len: maximum length of the generated sequence
        :param beam_width: beam width for beam search
        :param accelerator: device for PyTorch
        :return: generated sequence tensor
        """
        pass

    @final
    def save_model(self, path: str | Path) -> None:
        """ Save the model - all networks share the same method
        :param path: path to save the model
        """
        save(self.state_dict(), path)
        print("The model has been saved successfully.")

    @final
    def load_model(self, path: str | Path, strict: bool = False) -> None:
        """ Load the model - all networks share the same method
        :param path: path to load the model from
        :param strict: whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict function
        """
        self.load_state_dict(load(path, map_location=device(self._accelerator)), strict=strict)
        print("The model has been loaded successfully.")

    @final
    def _count_parameters(self) -> tuple[int, int]:
        """ Count total and trainable parameters
        :return: total parameters and trainable parameters
        """
        total_params = 0
        trainable_params = 0

        for param in self.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return total_params, trainable_params

    def summary(self):
        """ Print a summary of the model parameters """
        print("*" * WIDTH)
        print(f"Model Summary for {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"- Source Vocabulary Size: {self._vocab_src}")
        print(f"- Target Vocabulary Size: {self._vocab_tgt}")
        print(f"- Embedding Dimension:    {self._H}")
        print(f"- Hidden Size:            {self._M}")
        print(f"- Number of Layers:       {self._C}")
        print(f"- Dropout Rate:           {self._dropout}")
        print(f"- Bidirectional:          {self._bid}")
        print(f"- Device:                 {self._accelerator}")
        print(f"- PAD Token (Source):     {self._PAD_SRC}")
        print(f"- PAD Token (Target):     {self._PAD_TGT}")
        print(f"- SOS Token:              {self._SOS}")
        print(f"- EOS Token:              {self._EOS}")
        print("-" * WIDTH)
        # Calculate parameters
        total_params, trainable_params = self._count_parameters()
        print(f"Total parameters:         {total_params:,}")
        print(f"Trainable parameters:     {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("*" * WIDTH)


if __name__ == "__main__":
    pass
