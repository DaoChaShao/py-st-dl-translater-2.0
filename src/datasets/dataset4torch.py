#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   dataset4torch.py
# @Desc     :

from numpy import ndarray
from pandas import DataFrame, Series
from random import choice
from torch import Tensor, tensor, float32, long
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling label features and labels """

    def __init__(self, features, labels, use_batch_pad: bool = False):
        """ Initialise the TorchDataset class
        :param features: raw features(list/ndarray/DataFrame) or padded Tensor
        :param labels: raw labels(list/ndarray/DataFrame) or padded Tensor
        :param use_batch_pad: if True → keep raw lists, collate_fn will pad them
        """
        self._set_var_len = use_batch_pad

        if self._set_var_len:
            # Do not convert to tensor here; will be handled in collate_fn
            self._features = self._to_list_var_len_tensor(features)
            self._labels = self._to_list_var_len_tensor(labels)
        else:
            self._features: Tensor = self._to_equal_len_tensor(features)
            self._labels: Tensor = self._to_equal_len_tensor(labels)

    @property
    def features(self) -> Tensor | list:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor | list:
        """ Return the label tensor as a property """
        return self._labels

    @staticmethod
    def _to_equal_len_tensor(data: DataFrame | Tensor | ndarray | list) -> Tensor:
        """ Convert input data to a PyTorch tensor on the specified device
        :param data: the input data (DataFrame, ndarray, list, or Tensor)
        :return: the converted PyTorch tensor
        """
        if isinstance(data, (DataFrame, Series)):
            out = tensor(data.values, dtype=float32)
        elif isinstance(data, Tensor):
            out = data.float()
        elif isinstance(data, (ndarray, list)):
            out = tensor(data, dtype=float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return out

    @staticmethod
    def _to_list_var_len_tensor(data: DataFrame | Tensor | ndarray | list) -> list[Tensor]:
        if isinstance(data, list):
            return [tensor(item, dtype=long) for item in data]
        elif isinstance(data, (DataFrame, ndarray)):
            if isinstance(data, DataFrame):
                data = data.values
            return [tensor(row, dtype=long) for row in data]
        elif isinstance(data, Tensor):
            return [data[i].long() for i in range(len(data))]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: int | slice) -> tuple[Tensor, Tensor] | tuple[list, list]:
        """ Return a single or sliced (feature, label) pair """
        return self._features[index], self._labels[index]

    def __repr__(self):
        """ Return a string representation of the dataset """
        if self._set_var_len:
            info4features = f"len={len(self._features)} (unpadded tensor list)"
            info4labels = f"len={len(self._labels)} (unpadded tensor list)"
        else:
            info4features = f"shape={tuple(self._features.shape)}"
            info4labels = f"shape={tuple(self._labels.shape)}"

        return f"TorchDataset(features={info4features}, labels={info4labels})"


if __name__ == "__main__":
    var_len: bool = choice([True, False])
    print(f"{var_len}: Set Var Length") if var_len else print(f"{var_len}: Set Equal Length")

    if var_len:
        cn = [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [9, 10]
        ]
        en = [
            [2, 11, 12, 13, 3],
            [2, 14, 15, 16, 17, 3],
            [2, 18, 19, 20, 3]
        ]

        dataset = TorchDataset(features=cn, labels=en, use_batch_pad=True)

        for i in range(len(dataset)):
            feature, label = dataset[i]
            print(f"Sample {i + 1}: feature={feature}, label={label}")
        """
        True: Set Var Length
        Sample 1: feature=tensor([1., 2., 3.]), label=tensor([ 2., 11., 12., 13.,  3.])
        Sample 2: feature=tensor([4., 5., 6., 7., 8.]), label=tensor([ 2., 14., 15., 16., 17.,  3.])
        Sample 3: feature=tensor([ 9., 10.]), label=tensor([ 2., 18., 19., 20.,  3.])
        """
    else:
        features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        labels = [0, 1, 0]

        dataset = TorchDataset(features, labels, use_batch_pad=False)
        print(dataset)
        """
        False: Set Equal Length
        TorchDataset(features=shape=(3, 3), labels=shape=(3,))
        """
