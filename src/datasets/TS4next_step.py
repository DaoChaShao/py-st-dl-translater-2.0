#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:57
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   TS4next_step.py
# @Desc     :   

from torch import Tensor, tensor
from torch.utils.data import Dataset


class TimeSeriesTorchDatasetForPredNextStep(Dataset):
    """ A custom PyTorch Dataset class for handling sequential features and labels """

    def __init__(self, sequences: list, seq_max_len: int, mode: str, pad_token: int) -> None:
        """ Initialise the TorchDataset class for sequential data
        :param sequences: the input sequences
        :param seq_max_len: the length of each sequence
        :param mode: the mode of sequential data
        :param pad_token: the padding token to use
        """
        self._sequences = sequences
        self._length = seq_max_len
        self._mode = mode
        self._pad = pad_token

        match mode:
            case "seq2one":
                self._features, self._labels = self._pad_to_seq2one_tensor()
            case "seq2seq":
                self._features, self._labels = self._pad_to_seq2seq_tensor()
            case "slice":
                self._features, self._labels = self._slice_to_tensor()
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def _pad_to_seq2one_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via padding for one-step prediction
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - 1):
            if i < self._length - 1:
                feature = [self._pad] * (self._length - i - 1) + self._sequences[0: i + 1]
            else:
                feature = self._sequences[i - self._length + 1: i + 1]
            label = self._sequences[i + 1]
            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    def _pad_to_seq2seq_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via sequence padding for sequence-to-sequence prediction
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - 1):
            if i < self._length - 1:
                feature = [self._pad] * (self._length - i - 1) + self._sequences[0: i + 1]
                label = [self._pad] * (self._length - i - 1) + self._sequences[1: i + 2]
            else:
                feature = self._sequences[i - self._length + 1: i + 1]
                label = self._sequences[i - self._length + 2: i + 2]

            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    def _slice_to_tensor(self) -> tuple[Tensor, Tensor]:
        """ Convert input data to a PyTorch tensor via sliding window for next-step prediction
        :return: the converted PyTorch tensor
        """
        _features, _labels = [], []
        for i in range(len(self._sequences) - self._length):
            feature = self._sequences[i: i + self._length]
            label = self._sequences[i + self._length]
            _features.append(feature)
            _labels.append(label)

        return tensor(_features), tensor(_labels)

    @property
    def features(self) -> Tensor:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor:
        """ Return the label tensor as a property """
        return self._labels

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: int | slice) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if isinstance(index, slice):
            # Return a batch (for example dataset[:5])
            return self._features[index], self._labels[index]
        elif isinstance(index, int):
            # Return a single sample
            return self._features[index], self._labels[index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __repr__(self):
        """ Return a string representation of the dataset """
        return f"SequentialTorchDataset(features={self._features.shape}, labels={self._labels.shape}, device={self._features.device})"


if __name__ == "__main__":
    pass
