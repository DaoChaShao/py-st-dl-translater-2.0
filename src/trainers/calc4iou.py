#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 16:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   calc4iou.py
# @Desc     :   

from torch import Tensor, sigmoid


def calc_binary_sem_seg_iou(output: Tensor, target: Tensor, threshold: float = 0.5) -> dict[str, float]:
    """ Calculate binary IoU for semantic segmentation.
    :param output: logits, shape [B, 1, H, W] or [B, H, W]
    :param target: 0/1 mask, shape same as outputs
    :param threshold: threshold for converting logits to 0/1
    :return: dict of IoU
    """
    # Ensure same shape
    if output.ndim == 4 and output.shape[1] == 1:
        output = output.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Convert logits to binary mask
    preds = (sigmoid(output) > threshold).float()

    # IoU
    intersection = (preds * target).sum()
    union = ((preds + target) > 0).sum()
    iou = intersection / (union + 1e-8)

    # Pixel Accuracy
    pixel_acc = (preds == target).sum() / target.numel()

    return {
        "iou": iou.item(),
        "pixel_acc": pixel_acc.item()
    }


if __name__ == "__main__":
    pass
