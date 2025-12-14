#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 15:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_mlp.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import Database, FilePaths, Punctuations
from src.configs.cfg_dl import DataPreprocessor, Hyperparameters


@dataclass
class MLPParams:
    UNITS: int = 128


@dataclass
class Configuration4MLP:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: MLPParams = field(default_factory=MLPParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


CONFIG4MLP = Configuration4MLP()
