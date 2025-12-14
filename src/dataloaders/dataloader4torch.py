#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   dataloader4torch.py
# @Desc     :

from torch import Tensor, stack
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from src.datasets.dataset4torch import TorchDataset
from src.utils.highlighter import starts, lines


class TorchDataLoader:
    """ A custom PyTorch DataLoader class for handling TorchDataset """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32, shuffle_state: bool = True,
                 workers: int = 0,
                 use_batch_pad: bool = False, FEATURES_PAD_VALUE: int = 0, LABELS_PAD_VALUE: int = 0,
                 batch_first: bool = True, padding_direction: str = "right"
                 ):
        """ Initialise the TorchDataLoader class
        :param dataset: the TorchDataset or Dataset to load data from
        :param batch_size: the number of samples per batch
        :param shuffle_state: whether to shuffle the data at every epoch
        :param workers: the number of workers to use for data loading
        :param use_batch_pad: whether to pad sequences in the batch
        :param FEATURES_PAD_VALUE: the padding value for sequences
        :param LABELS_PAD_VALUE: the padding value for labels, -100 by default for PyTorch loss functions ignore_index
        :param batch_first: whether to have batch dimension first
        :param padding_direction: side to apply padding ("right" or "left")
        """
        self._dataset: Dataset = dataset
        self._batches: int = batch_size
        self._shuffle: bool = shuffle_state
        self._batch_pad: bool = use_batch_pad
        self._PAD4F = FEATURES_PAD_VALUE
        self._PAD4L = LABELS_PAD_VALUE
        self._first: bool = batch_first
        self._direction: str = padding_direction

        pin_memory: bool = False if workers == 0 else True
        print(f"num_workers={workers} | pin_memory={pin_memory}")
        print()

        pad_in_batch = self._collate_fn if self._batch_pad else None
        self._loader: DataLoader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batches,
            shuffle=self._shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=pad_in_batch
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _collate_fn(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """ Custom collate function to process a batch of data """
        # batch: list[tuple[Tensor, Tensor]]
        unpadded_features: list[Tensor] = [f for f, _ in batch]
        unpadded_labels: list[Tensor] = [l for _, l in batch]

        # Pad features
        features: Tensor = pad_sequence(
            unpadded_features,
            batch_first=self._first, padding_value=self._PAD4F, padding_side=self._direction
        )

        # Check labels
        label_state: int = unpadded_labels[0].dim()
        if label_state > 0:
            # Pad labels
            labels: Tensor = pad_sequence(
                unpadded_labels,
                batch_first=self._first, padding_value=self._PAD4L, padding_side=self._direction
            )
        else:
            # Labels are scalars; stack them directly
            labels: Tensor = stack(unpadded_labels, dim=0)

        return features, labels

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self):
        return (f"TorchDataLoader(dataset={self._dataset}, "
                f"batch_size={self._batches}, "
                f"shuffle={self._shuffle}), "
                f"features_pad_value={self._PAD4F}, "
                f"labels_pad_value={self._PAD4L}, "
                f"batch_first={self._first}, "
                f"padding_side='{self._direction}')")


if __name__ == "__main__":
    cn = [[1, 2, 3], [4, 5, 6, 7], [9, 10], [8]]
    en = [[2, 11, 12, 13, 3], [2, 14, 15, 16, 17, 18, 3], [2, 22, 19, 20, 21, 3], [2, 23, 3]]

    dataset = TorchDataset(features=cn, labels=en, use_batch_pad=True)

    dataloader = TorchDataLoader(
        dataset=dataset,
        batch_size=2,
        use_batch_pad=True,
        FEATURES_PAD_VALUE=0,
        padding_direction="right"
    )

    starts()
    print("Dataloader Representation:")
    lines()
    for idx, (features, labels) in enumerate(dataloader):
        print(f"Batch {idx + 1}:")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Features (padded): {features}")
        print(f"Labels (padded): {labels}")
        lines()
    print("Done!")
    starts()
    """
    ****************************************************************
    Dataloader Representation:
    ----------------------------------------------------------------
    Batch 1:
    Features shape: torch.Size([2, 3])
    Labels shape: torch.Size([2, 5])
    Features (padded): tensor([[8, 0, 0],
            [1, 2, 3]])
    Labels (padded): tensor([[ 2, 23,  3,  0,  0],
            [ 2, 11, 12, 13,  3]])
    ----------------------------------------------------------------
    Batch 2:
    Features shape: torch.Size([2, 4])
    Labels shape: torch.Size([2, 7])
    Features (padded): tensor([[ 9, 10,  0,  0],
            [ 4,  5,  6,  7]])
    Labels (padded): tensor([[ 2, 22, 19, 20, 21,  3,  0],
            [ 2, 14, 15, 16, 17, 18,  3]])
    ----------------------------------------------------------------
    Done!
    ****************************************************************
    """
