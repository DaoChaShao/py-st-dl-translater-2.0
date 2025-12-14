#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Neural Nets Module - Neural Network Architectures
----------------------------------------------------------------
This module provides a complete set of neural network architectures
for various machine learning tasks including segmentation, classification,
and sequence modeling.

Main Categories:
+ Standard4LayersUNetClassification: 4-layer UNet variant for semantic segmentation
+ Standard5LayersUNetForClassification: 5-layer UNet variant for semantic segmentation
+ LSTMRNNForClassification: Recurrent Neural Network for sequence classification tasks
+ NormalRNNForClassification: Standard RNN for sequence classification tasks

Usage:
+ Direct import of models via:
    - from src.nn import Standard4LayersUNetClassification, LSTMRNNForClassification, etc.
+ Instantiate models with default or custom parameters as needed.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .gru4classification import GRUClassifier
from .seq2seq import SeqToSeqCoder
from .lstm4classification import LSTMClassifier
from .rnn4classification import RNNClassifier
from .unet4layers4sem import Standard4LayersUNetClassification
from .unet5layers4sem import Standard5LayersUNetForClassification

__all__ = [
    "GRUClassifier",
    "SeqToSeqCoder",
    "LSTMClassifier",
    "RNNClassifier",
    "Standard4LayersUNetClassification",
    "Standard5LayersUNetForClassification",
]
