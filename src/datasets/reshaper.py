#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:34
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   reshaper.py
# @Desc     :   

from torch import Tensor, randn


def reshape_to_grayscale(flat_tensor: Tensor, height: int, width: int) -> Tensor:
    """ Reshape flattened tensor to grayscale tensor.
    :param flat_tensor: Input tensor of shape (N, H*W)
    :param height: Height of the output image
    :param width: Width of the output image
    :return: Reshaped tensor of shape (N, 1, H, W)
    """
    channels: int = 1

    print(f"Reshaping tensor from shape {flat_tensor.shape} to (N, {channels}, {height}, {width})")

    return flat_tensor.reshape(-1, channels, height, width)


if __name__ == "__main__":
    # input_tensor = randn(32, 784)
    # output_tensor = reshape_to_grayscale(input_tensor, 28, 28)
    # print(f"Output tensor shape: {output_tensor.shape}")
    pass
