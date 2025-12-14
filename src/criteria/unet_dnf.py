#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/22 11:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_dnf.py
# @Desc     :   

from torch import nn, sigmoid, exp


class DiceFocalBCELoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=0.8, gamma=2.0, smooth=1e-6, weights_ratio: list = None):
        super().__init__()
        self._pos_weight = pos_weight
        self._alpha = alpha
        self._gamma = gamma
        self._smooth = smooth
        # [dice_weight, bce_weight, focal_weight]
        self._weights = weights_ratio if weights_ratio is not None else [0.4, 0.3, 0.3]

    def forward(self, inputs, targets):
        sig_out = sigmoid(inputs)

        # 1. Dice Loss
        intersection = (sig_out * targets).sum()
        dice = (2. * intersection + self._smooth) / (sig_out.sum() + targets.sum() + self._smooth)
        dice_loss = 1 - dice

        # 2. BCE Loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self._pos_weight, reduction="mean"
        )

        # 3. Focal Loss
        bce_loss_per_pixel = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = exp(-bce_loss_per_pixel)
        foreground_mask = (targets > 0.5).float()
        focal_weight = self._alpha * (1 - pt) ** self._gamma
        focal_weight = focal_weight * (1 + foreground_mask)
        focal_loss = (focal_weight * bce_loss_per_pixel).mean()

        # Combine Losses
        total_loss = (self._weights[0] * dice_loss + self._weights[1] * bce_loss + self._weights[2] * focal_loss)

        return total_loss


if __name__ == "__main__":
    pass
