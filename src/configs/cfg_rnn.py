#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_rnn.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations
from src.configs.cfg_dl import DataPreprocessor, Hyperparameters


@dataclass
class RNNParams:
    BEAM_SIZE: int = 5
    CLASSES: int = 3  # Binary classification is 2
    EMBEDDING_DIM: int = 128
    HIDDEN_SIZE: int = 256
    LAYERS: int = 2
    PAD_FEATURES_IN_BATCH: int = 0  # Padding value for features in batch
    PAD_LABELS_IN_BATCH: int = -100  # Padding value for labels in batch
    TEMPERATURE: float = 1.0


@dataclass
class Configuration4RNN:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: RNNParams = field(default_factory=RNNParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


CONFIG4RNN = Configuration4RNN()
