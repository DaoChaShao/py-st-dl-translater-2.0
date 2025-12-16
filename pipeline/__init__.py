#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 17:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Data Processing Module - Deep Learning Workflow
----------------------------------------------------------------
This package provides utility modules for processing and preparing
datasets for machine learning, NLP, CV, and general data tasks.

Main Categories:
+ preprocess_data : functions and classes for preprocessing data
+ process_data : functions and classes for processing target data
+ prepare_data : functions and classes for preparing datasets for training and validation dataloaders

Usage:
+ Direct import from this package:
    - from src.configs import process_data, prepare_data
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .preprocessor import preprocess_data
from .processor import process_data
from .prepper import prepare_data

__all__ = [
    "preprocess_data",
    "process_data",
    "prepare_data",
]
