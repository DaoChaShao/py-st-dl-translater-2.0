#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Trainers Module - Neural Network Training Frameworks
----------------------------------------------------------------
This module provides a complete set of specialized PyTorch trainer
classes for training and evaluating neural network models across
regression, sequence classification, and semantic segmentation tasks.

Main Categories:
+ TorchTrainer4Regression: Trainer for MLP/CNN-based regression models
  with support for continuous target prediction and regression metrics
+ TorchTrainer4Seq2Classification: Trainer for RNN/GRU/LSTM sequence
  classification models with sequence processing and label prediction
+ TorchTrainer4UNetSemSeg: Trainer for UNet-based semantic segmentation
  with image-mask training loops and segmentation evaluation metrics

Usage:
+ Direct import of trainer classes via:
    - from src.trainers import TorchTrainer4Regression, TorchTrainer4UNetSemSeg, etc.
+ Instantiate trainers with model, data loaders, optimizer, and config
  to perform full training and evaluation cycles.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from . import configs
from . import criteria
from . import dataloaders
from . import datasets
from . import nets
from . import trainers
from . import utils

__all__ = [
    "configs",
    "criteria",
    "dataloaders",
    "datasets",
    "nets",
    "trainers",
    "utils",
]
