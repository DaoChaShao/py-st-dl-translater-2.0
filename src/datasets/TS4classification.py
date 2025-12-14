#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   lstm4classification.py
# @Desc     :   

from torch import tensor, long, Tensor
from torch.utils.data import Dataset


class TimeSeriesTorchDatasetForClassification(Dataset):
    """ A custom PyTorch Dataset class for handling sequential features and labels """

    def __init__(self, feature_seqs: list, lbl_seqs: list, seq_max_len: int, pad_token: int = 0) -> None:
        """ Initialise the TorchDataset class for sequential data
        :param feature_seqs: the input sequences
        :param lbl_seqs: the label sequences
        :param seq_max_len: the length of each sequence
        :param pad_token: the padding token to use
        """
        self._sequences = feature_seqs
        self._length = seq_max_len
        self._pad = pad_token
        self._features = self._pad_to_fixed_len_tensor()
        self._labels = tensor(lbl_seqs, dtype=long)

    def _pad_to_fixed_len_tensor(self) -> Tensor:
        """ Convert input data to a PyTorch tensor via padding to fixed length
        :return: the converted PyTorch tensor
        """
        _features = []
        for seq in self._sequences:
            if len(seq) < self._length:
                padded_seq = seq + [self._pad] * (self._length - len(seq))
            else:
                padded_seq = seq[:self._length]
            _features.append(padded_seq)

        return tensor(_features)

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
