#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/22 12:57
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet_edge.py
# @Desc     :   

from torch import nn, tensor, abs, no_grad, float32


class EdgeAwareLoss(nn.Module):
    def __init__(self, pos_weight, edge_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.edge_weight = edge_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _compute_edges(self, mask):
        sobel_x = tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float32).view(1, 1, 3, 3)
        sobel_y = tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float32).view(1, 1, 3, 3)

        sobel_x = sobel_x.to(mask.device)
        sobel_y = sobel_y.to(mask.device)

        edge_x = abs(nn.functional.conv2d(mask, sobel_x, padding=1))
        edge_y = abs(nn.functional.conv2d(mask, sobel_y, padding=1))
        edges = (edge_x + edge_y) > 0.1

        return edges.float()

    def forward(self, inputs, targets):
        base_loss = self.bce_loss(inputs, targets)

        with no_grad():
            target_edges = self._compute_edges(targets)
            edge_mask = (target_edges > 0).float()

        edge_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, weight=1.0 + self.edge_weight * edge_mask
        )

        return 0.7 * base_loss + 0.3 * edge_loss


if __name__ == "__main__":
    pass
