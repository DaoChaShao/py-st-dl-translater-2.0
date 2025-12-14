#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:56
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_dl.py
# @Desc     :   

from dataclasses import dataclass, field
from torch import cuda

from src.configs.cfg_base import Database, FilePaths, Punctuations


@dataclass
class DataPreprocessor:
    BATCHES: int = 16
    DROPOUT_RATIO: float = 0.5
    IMAGE_HEIGHT: int = 320
    IMAGE_WIDTH: int = 384
    MAX_SEQUENCE_LEN: int = 5
    PCA_VARIANCE_THRESHOLD: float = 0.95
    RANDOMNESS: int = 27
    SHUFFLE: bool = True
    TEST_SIZE: float = 0.2
    WORKERS: int = 4


@dataclass
class Hyperparameters:
    ACCELERATOR: str = "cuda" if cuda.is_available() else "cpu"
    DECAY: float = 1e-4


@dataclass
class Config4DL:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


CONFIG4DL = Config4DL()
