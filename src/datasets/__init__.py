#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Trainer Module - Neural Network Training Implementations
----------------------------------------------------------------
This module provides a unified set of trainer classes for various
neural network tasks including regression, sequence classification,
and semantic segmentation.

Main Categories:
+ TorchTrainer4Regression: Trainer for regression models with continuous-value prediction
+ TorchTrainer4Seq2Classification: Trainer for sequence classification tasks using RNN-based models
+ TorchTrainer4UNetSemSeg: Trainer for UNet-based semantic segmentation with image–mask supervision

Usage:
+ Direct import of trainers:
    - from src.trainers import TorchTrainer4Regression, TorchTrainer4UNetSemSeg, etc.
+ Instantiate trainers with model, criterion, optimizer, and dataloaders
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .dataset4torch import TorchDataset
from .reshaper import reshape_to_grayscale
from .TS4classification import TimeSeriesTorchDatasetForClassification
from .TS4next_step import TimeSeriesTorchDatasetForPredNextStep
from .mask_mapper import mask2index
from .sem_seg import TorchDataset4SemanticSegmentation

__all__ = [
    "TorchDataset",
    "reshape_to_grayscale",
    "TimeSeriesTorchDatasetForClassification",
    "TimeSeriesTorchDatasetForPredNextStep",
    "mask2index",
    "TorchDataset4SemanticSegmentation",
]
