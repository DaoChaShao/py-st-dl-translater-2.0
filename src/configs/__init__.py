#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
ML/Data Processing Configuration Mudules
----------------------------------------------------------------
This package provides comprehensive configurations and utility
modules for machine learning, NLP, CV, and general data processing.

Main Categories:
+ cfg_base       : Basic file paths and general config (CONFIG, FilePaths)
+ cfg_base4dl    : Deep learning base parameters (CONFIG4DL, DataPreprocessor, Hyperparameters)
+ cfg_cnn        : CNN-specific parameters (CONFIG4CNN, CNNParams)
+ cfg_mlp        : MLP/NLP-specific parameters (CONFIG4MLP, NLPParams)
+ cfg_rnn        : RNN-specific parameters (CONFIG4RNN, RNNParams)
+ cfg_unet       : UNet-specific parameters (CONFIG4UNET, UNetParams)
+ eMun: Some necessary options functionality
+ parser         : Parser module

Usage:
+ Access default configuration: e.g., CONFIG4CNN.CNN_PARAMS.OUT_CHANNELS
+ Create new instances for custom configurations:
    - from src.configs import Configuration4CNN, CNNParams
    - my_cnn_config = Configuration4CNN(CNN_PARAMS=CNNParams(OUT_CHANNELS=128))
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .cfg_base import CONFIG, Config, Database, FilePaths, Punctuations
from .cfg_cnn import CONFIG4CNN, Configuration4CNN, CNNParams
from .cfg_dl import CONFIG4DL, Config4DL, DataPreprocessor, Hyperparameters
from .cfg_mlp import CONFIG4MLP, Configuration4MLP, MLPParams
from .cfg_rnn import CONFIG4RNN, Configuration4RNN, RNNParams
from .cfg_unet import CONFIG4UNET, Configuration4UNet, UNetParams
from .cfg_types import Langs, TSSeqSeparate, Seq2SeqNets, Seq2SeqStrategies, Tokens, Tasks
from .parser import set_argument_parser

__all__ = [
    "CONFIG", "Config", "Database", "FilePaths", "Punctuations",
    "CONFIG4DL", "Config4DL", "DataPreprocessor", "Hyperparameters",
    "CONFIG4CNN", "Configuration4CNN", "CNNParams",
    "CONFIG4MLP", "Configuration4MLP", "MLPParams",
    "CONFIG4RNN", "Configuration4RNN", "RNNParams",
    "CONFIG4UNET", "Configuration4UNet", "UNetParams",
    "Langs", "TSSeqSeparate", "Seq2SeqNets", "Seq2SeqStrategies", "Tokens", "Tasks",
    "set_argument_parser"
]
