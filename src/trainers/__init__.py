#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Trainers & Metrics Module - PyTorch Implementations
----------------------------------------------------------------
This module provides a complete set of specialized PyTorch trainer
classes and metric calculators for various machine learning tasks
including regression, sequence classification, and semantic segmentation.

Main Categories:
+ TorchTrainer4Regression: Trainer for regression models such as MLP
  or CNN, providing end-to-end training loops and regression metrics

+ TorchTrainer4Seq2Classification: Trainer for sequence classification
  using RNN/GRU/LSTM architectures with full support for sequential
  data batching and classification evaluation

+ TorchTrainer4UNetSemSeg: Trainer for UNet-based semantic segmentation
  with image-mask training cycles, IoU computation, pixel accuracy,
  and confusion matrix evaluation

Utility Functions:
+ calculator_for_classification: Metrics calculation for classification
+ calculator_for_confusion_metrics: Confusion matrix-based metrics
+ calc_binary_sem_seg_iou: Binary semantic segmentation IoU and pixel accuracy
+ calculator_for_regression: Regression metrics computation

Usage:
+ Direct import of trainer classes via:
    - from src.trainers import TorchTrainer4Regression, TorchTrainer4UNetSemSeg, etc.
+ Instantiate trainer classes with model, data loaders, optimizer, and config
  to perform full supervised training workflows with built-in metrics.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .calc4classification import calculator_for_classification
from .calc4cm import calculator_for_confusion_metrics
from .calc4iou import calc_binary_sem_seg_iou
from .calc4regression import calculator_for_regression
from .calc4seq_text_quilty import TextQualityScorer
from .trainer4sem_seg import TorchTrainer4UNetSemSeg
from .trainer4seq2seq import TorchTrainer4SeqToSeq
from .trainer4torch import TorchTrainer

__all__ = [
    "calculator_for_classification",
    "calculator_for_confusion_metrics",
    "calc_binary_sem_seg_iou",
    "calculator_for_regression",
    "TextQualityScorer",
    "TorchTrainer4UNetSemSeg",
    "TorchTrainer4SeqToSeq",
    "TorchTrainer"
]
