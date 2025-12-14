#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   lstm4classification.py
# @Desc     :   

from torch import nn, cat, zeros, device, randint, arange, Tensor

WIDTH: int = 64


class LSTMClassifier(nn.Module):
    """ A LSTM model for multi-class classification tasks using PyTorch """

    def __init__(
            self,
            vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
            num_classes: int, dropout_rate: float = 0.3, bid: bool = True,
            accelerator: str = "cpu", task: str = "classification", pad_idx: int = 0
    ):
        super().__init__()
        """ Initialise the CharsRNNModel class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param num_classes: number of output classes
        :param dropout_rate: dropout rate for regularization
        :param bid: bidirectional flag
        :param accelerator: accelerator for PyTorch
        :param task: task type, either "classification" or "generation"
        :param pad_idx: padding index for the embedding layer
        """
        self._L = vocab_size  # Lexicon/Vocabulary size
        self._H = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count
        self._accelerator = accelerator
        self._task = task
        self._pad_idx = pad_idx
        self._factor = 2 if bid else 1
        dropout = dropout_rate if self._C > 1 else 0.0

        self._embed = nn.Embedding(self._L, self._H, padding_idx=pad_idx)
        self._lstm = nn.LSTM(self._H, self._M, self._C, batch_first=True, bidirectional=bid, dropout=dropout)
        self._dropout = nn.Dropout(dropout_rate)
        self._linear = nn.Linear(self._M * self._factor, num_classes)

        self._init_params()

    def _init_params(self):
        """ Initialize model parameters """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias_ih" in name:
                # Forget gate bias initialization to 1
                param.data[self._M:2 * self._M] = 1.0

    def _init_hidden(self, batch_size: int, accelerator: device) -> tuple[Tensor, Tensor]:
        """ Initialize h0 and c0
        :param batch_size: size of the batch
        :param accelerator: device for PyTorch
        :return: h0 and c0 tensors
        """
        shape = (self._C * self._factor, batch_size, self._M)
        h0 = zeros(shape, device=accelerator)
        c0 = zeros(shape, device=accelerator)
        return h0, c0

    def forward(self, X):
        """ Forward pass of the model
        :param X: input tensor, shape (batch_size, sequence_length)
        :return: output tensor and new hidden state tensor, shapes (batch_size, sequence_length, vocab_size) and (num_layers, batch_size, hidden_dim)
        """
        embeddings = self._embed(X)

        batches = X.size(0)
        h0, c0 = self._init_hidden(batches, X.device)
        output, (hidden, cell) = self._lstm(embeddings, (h0, c0))

        last_hidden = None
        match self._task:
            case "classification":
                # Method I, which is better for classification tasks
                if self._factor == 2:
                    forward_hn = hidden[-2]  # [batch_size, hidden_size]
                    backward_hn = hidden[-1]  # [batch_size, hidden_size]
                    last_hidden = cat([forward_hn, backward_hn], dim=1)  # [batch_size, hidden_size*2]
                else:
                    last_hidden = hidden[-1]
            case "generation":
                # Method II, using the last output timestep, which is better for sequence generation tasks
                lengths: int = (X != self._pad_idx).sum(dim=1)  # shape: (batch,)
                batch_idx = arange(batches, device=X.device)

                forward_last = output[batch_idx, lengths - 1, :self._M]
                if self._factor == 2:
                    backward_first = output[batch_idx, 0, self._M:]
                else:
                    backward_first = zeros((batches, self._M), device=X.device)

                last_hidden = cat([forward_last, backward_first], dim=1)
            case _:
                raise ValueError(f"Unsupported task type: {self._task}")

        last_hidden = self._dropout(last_hidden)
        # Fully connected layer, shape (batch_size, num_classes)
        out = self._linear(last_hidden)

        return out

    def summary(self):
        print("=" * WIDTH)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model Summary for {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"- Vocabulary size: {self._L}")
        print(f"- Embedding dim: {self._H}")
        print(f"- Hidden size: {self._M}")
        print(f"- Num layers: {self._C}")
        print(f"- Output classes: {self._linear.out_features}")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        print("=" * WIDTH)
        print()


if __name__ == "__main__":
    vocab_size = 7459
    batch_size = 16
    seq_len = 111

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        num_classes=vocab_size,
        dropout_rate=0.5,
        bid=True,
        accelerator="cpu",
    )
    model.summary()

    # 测试输入
    X = randint(0, vocab_size, (batch_size, seq_len))
    output = model(X)
    print(f"Input Size: {X.shape}")
    print(f"Output Size: {output.shape}")
