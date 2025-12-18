#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/15 15:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   multi_task_gru.py
# @Desc     :   

from torch import (Tensor, nn,
                   cat, zeros, arange,
                   randint)
from typing import override, Literal

from src.configs.cfg_types import Tasks
from src.nets.base_rnn import BaseRNN


class MultiTaskGRU(BaseRNN):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 accelerator: str | Literal["cuda", "cpu"] = "cpu",
                 PAD: int = 0,
                 *,
                 task: str | Tasks | Literal["classification", "generation"] = "classification", class_num: int = 2,
                 ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            accelerator=accelerator,
            PAD=PAD
        )
        """ Initialise the GRU class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bidirectional: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param PAD: padding index for the embedding layer
        :param task: type of task (classification or generation)
        :param class_num: number of classes for classification tasks
        """
        self._task: str = task
        self._classes: int = class_num

        self._gru = nn.GRU(
            self._H, self._M, self._C,
            dropout=self._dropout if self._C > 1 else 0.0,
            batch_first=True, bidirectional=self._bid
        )
        self._drop: nn.Dropout = nn.Dropout(self._dropout)
        self._linear = nn.Linear(self._M * self._num_directions, self._classes)

        self.init_weights()

    @override
    def forward(self, X: Tensor) -> Tensor:
        """ Forward pass of the model
        :param X: input tensor, shape (batch_size, sequence_length)
        :return: output tensor and new hidden state tensor, shapes (batch_size, sequence_length, vocab_size) and (num_layers, batch_size, hidden_dim)
        """
        embeddings = self._embed(X)

        batches = X.size(0)
        h0 = self.init_hidden(batches)
        output, hidden = self._gru(embeddings, h0)

        last_hidden = None
        match self._task:
            # Sequence Classification
            case "classification":
                # Method I, which is better for classification tasks
                if self._num_directions == 2:
                    forward_hn = hidden[-2]  # [batch_size, hidden_size]
                    backward_hn = hidden[-1]  # [batch_size, hidden_size]
                    last_hidden = cat([forward_hn, backward_hn], dim=1)  # [batch_size, hidden_size*2]
                else:
                    last_hidden = hidden[-1]
            case "generation":
                # Causal Language Modeling
                # Method II, using the last output timestep, which is better for sequence generation tasks
                lengths: Tensor = (X != self._PAD).sum(dim=1)  # shape: (batch,)
                batch_idx = arange(batches, device=X.device)

                forward_last = output[batch_idx, lengths - 1, :self._M]
                if self._num_directions == 2:
                    backward_first = output[batch_idx, 0, self._M:]
                else:
                    backward_first = zeros((batches, self._M), device=X.device)

                last_hidden = cat([forward_last, backward_first], dim=1)
            case _:
                raise ValueError(f"Unsupported task type: {self._task}")

        last_hidden = self._drop(last_hidden)
        # Fully connected layer, shape (batch_size, num_classes)
        result = self._linear(last_hidden)

        return result


if __name__ == "__main__":
    vocab_size: int = 7459
    batch_size: int = 16
    seq_len: int = 111

    model = MultiTaskGRU(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout_rate=0.5,
        bidirectional=True,
        task=Tasks.CLASSIFICATION,
        class_num=2
    )
    model.summary()

    X = randint(0, vocab_size, (batch_size, seq_len))
    output = model(X)

    print(f"Tester:")
    print(f"Input Size: {X.shape}")
    print(f"Output Size: {output.shape}")
    print()
