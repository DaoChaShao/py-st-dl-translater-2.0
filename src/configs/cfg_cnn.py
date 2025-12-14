#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 15:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_cnn.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations
from src.configs.cfg_dl import DataPreprocessor, Hyperparameters


@dataclass
class CNNParams:
    KERNEL_SIZE: int = 3
    OUT_CHANNELS: int = 64
    PAD_SIZE: int = 1
    STRIDE_SIZE: int = 1


@dataclass
class Configuration4CNN:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    PARAMETERS: CNNParams = field(default_factory=CNNParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


CONFIG4CNN = Configuration4CNN()
