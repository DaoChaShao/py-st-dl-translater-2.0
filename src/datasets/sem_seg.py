#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/21 16:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   sem_seg.py
# @Desc     :   

from numpy import array, ndarray
from PIL import Image
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from src.datasets.mask_mapper import mask2index


class TorchDataset4SemanticSegmentation(Dataset):
    """ A custom PyTorch Dataset class for UNet model """

    def __init__(self, image_paths: list[str], mask_paths: list[str], transformer=None):
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._transformer = transformer

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple:
        assert index < len(self._image_paths), f"Index {index} out of range."

        img_image: Image.Image = Image.open(self._image_paths[index]).convert("RGB")
        img_mask: Image.Image = Image.open(self._mask_paths[index])

        image_arr: ndarray = array(img_image)
        mask_arr, _ = mask2index(img_mask)

        if self._transformer:
            augmented = self._transformer(image=image_arr, mask=mask_arr)
            image: Tensor = augmented["image"]
            mask: Tensor = augmented["mask"]
        else:
            image: Tensor = from_numpy(image_arr).permute(2, 0, 1).float() / 255.0
            mask: Tensor = from_numpy(mask_arr).float()

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask


if __name__ == "__main__":
    pass
