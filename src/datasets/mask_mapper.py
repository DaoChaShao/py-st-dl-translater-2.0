#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/23 18:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   mask_mapper.py
# @Desc     :   

from PIL import Image
from numpy import ndarray, array, float32, unique


def mask2index(mask: Image.Image) -> tuple[ndarray, ndarray]:
    """ Convert mask pixel values to class IDs """
    # Transfer PIL Image to numpy array - shape: (H, W, C) or (H, W)
    arr = array(mask)

    # Calculate the frequency of each class in the mask
    # print(f"Mask Distribution:")
    # classes_before, freq_before = unique(arr, return_counts=True)
    # for cls, freq in zip(classes_before, freq_before):
    #     percentage = freq / arr.size * 100
    #     print(f"- class {cls}: {freq} pixels ({percentage:.2f}%)")

    # Convert to binary segmentation: foreground (1) and background (0)
    seg_arr: ndarray = (arr > 0).astype(float32)
    classes = unique(seg_arr)

    # Calculate the frequency of each class after conversion
    # print(f"Converted Mask Distribution:")
    # classes_after, freq_after = unique(seg_arr, return_counts=True)
    # for cls, freq in zip(classes_after, freq_after):
    #     percentage = freq / seg_arr.size * 100
    #     print(f"- class {cls}: {freq} pixels ({percentage:.2f}%)")

    return seg_arr, classes


if __name__ == "__main__":
    pass
