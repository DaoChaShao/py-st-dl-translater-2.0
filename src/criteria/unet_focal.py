#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/18 01:30
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_focal.py
# @Desc     :   

from torch import nn, exp


class FocalLoss(nn.Module):
    """ Focal Loss with emphasis on foreground pixels for binary segmentation tasks. """

    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, inputs, targets):
        # More focus on foreground pixel (targets == 1)
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = exp(-BCE_loss)

        # Give higher weight to foreground pixels
        foreground_mask = (targets > 0.5).float()
        focal_weight = self._alpha * (1 - pt) ** self._gamma

        # Increase weight for foreground pixels
        focal_weight = focal_weight * (1 + foreground_mask)
        focal_loss = (focal_weight * BCE_loss).mean()

        return focal_loss


if __name__ == "__main__":
    pass
