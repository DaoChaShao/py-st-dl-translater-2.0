#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Criterion Module - Specialized PyTorch Loss Functions
----------------------------------------------------------------
This module provides a complete set of specialized loss functions
for various machine learning tasks including semantic segmentation
and regression.

Main Categories:
+ DiceBCELoss: Combined Dice coefficient and Binary Cross-Entropy loss
  for segmentation tasks, balancing region overlap and pixel-wise accuracy
+ DiceFocalBCELoss: Multi-component loss combining Dice, Focal, and BCE terms
  for robust segmentation
+ EdgeAwareLoss: Segmentation loss emphasizing object boundaries and edges
+ FocalLoss: Adaptive cross-entropy variant focusing on hard-to-classify pixels
+ regression_log_mse: Logarithmic Mean Squared Error loss for regression tasks

Usage:
+ Direct import of loss functions via:
    - from src.criterion import DiceBCELoss, EdgeAwareLoss, regression_log_mse, etc.
+ Instantiate losses with optional parameters (e.g., pos_weight, alpha, gamma, smooth)
  as needed for your task.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .mlp_log_mse import regression_log_mse
from .unet_edge import EdgeAwareLoss
from .unet_dice import DiceBCELoss
from .unet_dnf import DiceFocalBCELoss
from .unet_focal import FocalLoss

__all__ = [
    "regression_log_mse",
    "DiceBCELoss",
    "DiceFocalBCELoss",
    "EdgeAwareLoss",
    "FocalLoss",
]
