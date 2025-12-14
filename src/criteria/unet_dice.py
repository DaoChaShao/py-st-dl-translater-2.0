#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 04:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_dice.py
# @Desc     :   

from torch import nn, sigmoid


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=None, smooth: float = 1e-6):
        super().__init__()
        self._pos_weight = pos_weight
        self._smooth = smooth

    def forward(self, inputs, targets):
        sig_out = sigmoid(inputs)

        intersection = (sig_out * targets).sum()
        dice = (2. * intersection + self._smooth) / (sig_out.sum() + targets.sum() + self._smooth)
        dice_loss = 1 - dice

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self._pos_weight,
            reduction="mean"
        )

        return bce_loss + dice_loss


if __name__ == "__main__":
    pass
