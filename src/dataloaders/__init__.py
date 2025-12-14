#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :   

"""
****************************************************************
Dataloader Module - PyTorch DataLoader Wrapper
----------------------------------------------------------------
This module provides a complete set of specialized PyTorch DataLoader
wrappers for various machine learning tasks including semantic
segmentation, image classification, sequence classification,
and sequence prediction.

Main Categories:
+ TorchDataLoader: Custom PyTorch DataLoader wrapper with simplified
  interface and enhanced accessibility

Usage:
+ Direct import of the DataLoader via:
    - from src.dataloaders import TorchDataLoader
+ Instantiate with any PyTorch Dataset and custom batch or device parameters as needed
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .dataloader4torch import TorchDataLoader

__all__ = [
    "TorchDataLoader",
]
