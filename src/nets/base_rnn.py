#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/15 11:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   base_rnn.py
# @Desc     :   

from abc import ABC, abstractmethod
from pathlib import Path
from torch import (Tensor, nn, zeros,
                   save, load)
from typing import final, Literal, Final

WIDTH: int = 64


class BaseRNN(ABC, nn.Module):
    """ Abstract Base Class for RNN-based Networks """

    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 *,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0
                 ) -> None:
        super().__init__()
        """ Initialize the ABCNet class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD: padding index for the embedding layer
        """
        self._L: int = vocab_size  # Lexicon/Vocabulary size
        self._H: int = embedding_dim  # Embedding dimension
        self._M: int = hidden_size  # Hidden dimension
        self._C: int = num_layers  # RNN layers count

        self._dropout: float = dropout_rate if num_layers > 1 else 0.0
        self._bid: bool = bidirectional
        self._accelerator: str = accelerator.lower()

        self._PAD: Final[int] = PAD
        self._num_directions: int = self._set_num_directions(self._bid)

        self._embed: nn.Embedding = nn.Embedding(self._L, self._H, padding_idx=self._PAD)

    @staticmethod
    def _set_num_directions(bid: bool) -> int:
        """ Set the num_directions based on bidirectionality
        :param bid: bidirectional flag
        :return: number of directions (1 for unidirectional, 2 for bidirectional)
        """
        return 2 if bid else 1

    @property
    def vocab_size(self) -> int:
        """ Get vocabulary size
        :return: vocabulary size
        """
        return self._L

    @property
    def embedding_dim(self) -> int:
        """ Get embedding dimension
        :return: embedding dimension
        """
        return self._H

    @property
    def hidden_size(self) -> int:
        """ Get hidden size
        :return: hidden size
        """
        return self._M

    @property
    def num_layers(self) -> int:
        """ Get number of layers
        :return: number of layers
        """
        return self._C

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the network
        :param x: input tensor
        :return: output tensor
        """
        pass

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
        h0: Tensor = zeros(shape, device=self._accelerator)
        return h0

    @final
    def init_lstm_hidden(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """ Initialize h0 and c0
        :param batch_size: size of the batch
        :param accelerator: device for PyTorch
        :return: h0 and c0 tensors
        """
        shape: tuple[int, int, int] = (self._C * self._num_directions, batch_size, self._M)
        h0: Tensor = zeros(shape, device=self._accelerator)
        c0: Tensor = zeros(shape, device=self._accelerator)
        return h0, c0

    @final
    def save_model(self, path: str | Path) -> None:
        """ Save the model - all networks share the same method
        :param path: path to save the model
        """
        save(self.state_dict(), str(path))
        print("The model has been saved successfully.")

    @final
    def load_model(self, path: str | Path, strict: bool = False) -> None:
        """ Load the model - all networks share the same method
        :param path: path to load the model from
        :param strict: whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict function
        """
        self.load_state_dict(load(str(path)), strict=strict)
        print("The model has been loaded successfully.")

    @final
    def _count_parameters(self) -> tuple[int, int]:
        """ Count total and trainable parameters
        :return: total and trainable parameters
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
        print(f"- Vocabulary Size:        {self._L}")
        print(f"- Embedding Dimension:    {self._H}")
        print(f"- Hidden Size:            {self._M}")
        print(f"- Number of Layers:       {self._C}")
        print(f"- Dropout Rate:           {self._dropout}")
        print(f"- Bidirectional:          {self._bid}")
        print(f"- Device:                 {self._accelerator}")
        print(f"- PAD Token:              {self._PAD}")
        print("-" * WIDTH)
        # Calculate parameters
        total_params, trainable_params = self._count_parameters()
        print(f"Total parameters:         {total_params:,}")
        print(f"Trainable parameters:     {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("*" * WIDTH)


if __name__ == "__main__":
    pass
