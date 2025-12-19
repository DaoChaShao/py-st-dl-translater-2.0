#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :

"""
****************************************************************
ML/Data Processing Utility Module - Comprehensive Toolkit
----------------------------------------------------------------
This module provides a complete set of utility functions and classes
for machine learning, natural language processing, computer vision,
and general data processing tasks.

Main Categories:
+ apis: API wrappers for OpenAI and DeepSeek
+ decorator: Performance measurement and timing (timer, Timer)
+ helper: Text formatting, timing, and I/O utilities
+ highlighter: Terminal text coloring and formatting functions
+ Logger: Structured logging system
+ NLTK: Natural Language Toolkit tokenizer wrapper
+ nlp: Chinese/English text processing, tokenization, and vocabulary utilities
+ PT: PyTorch tensor operations, device management, and network verification
+ SQL: SQLite database management
+ stats: Statistical analysis, data preprocessing, file I/O, and dimensionality reduction
+ THU: Chinese word segmentation with THULAC

Usage:
+ Direct import of utility functions or classes via:
    - from src.utils import read_file, check_device, record_log, etc.
+ Access default configurations or create custom instances as needed.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.3.0"

from .apis import (OpenAITextCompleter, OpenAIImageCompleter,
                   DeepSeekCompleter)
from .decorator import beautifier, timer
from .helper import (Beautifier, Timer, RandomSeed,
                     read_file, read_files, read_yaml)
from .highlighter import (black, red, green, yellow, blue, purple, cyan, white,
                          bold, underline, invert, strikethrough,
                          starts, lines, sharps)
from .logger import record_log
from .NLTK import NLTKTokenizer
from .nlp import (regular_chinese, regular_english,
                  count_frequency, unique_characters, extract_cn_chars,
                  SpaCyBatchTokeniser,
                  build_word2id_seqs,
                  check_vocab_coverage)
from .PT import (TorchRandomSeed,
                 check_device, get_device,
                 item2tensor, sequences2tensors,
                 balance_imbalanced_weights,
                 TensorLogWriter,
                 verify_seq_net_initialisation)
from .SQL import SQLiteIII
from .stats import (NumpyRandomSeed,
                    load_csv, load_text, summary_dataframe,
                    load_paths, split_paths,
                    create_train_valid_split_byXy, create_full_data_split_byXy, create_full_data_split,
                    save_json, load_json,
                    create_data_transformer, transform_data,
                    pca_importance,
                    get_correlation_among_features,
                    get_categories_corr_ratio, get_correlation_among_Xy)
from .THU import THULACTokeniser

__all__ = [
    "OpenAITextCompleter", "OpenAIImageCompleter",
    "DeepSeekCompleter",

    "beautifier", "timer",

    "Beautifier", "Timer", "RandomSeed",
    "read_file", "read_files", "read_yaml",

    "black", "red", "green", "yellow", "blue", "purple", "cyan", "white",
    "bold", "underline", "invert", "strikethrough",
    "starts", "lines", "sharps",

    "record_log",

    "NLTKTokenizer",

    "regular_chinese", "regular_english",
    "count_frequency", "unique_characters", "extract_cn_chars",
    "SpaCyBatchTokeniser",
    "build_word2id_seqs",
    "check_vocab_coverage",

    "TorchRandomSeed",
    "check_device", "get_device",
    "item2tensor", "sequences2tensors",
    "balance_imbalanced_weights",
    "TensorLogWriter",
    "verify_seq_net_initialisation",

    "SQLiteIII",

    "NumpyRandomSeed",
    "load_csv", "load_text", "summary_dataframe",
    "load_paths", "split_paths",
    "create_train_valid_split_byXy", "create_full_data_split_byXy", "create_full_data_split",
    "save_json", "load_json",
    "create_data_transformer", "transform_data",
    "pca_importance",
    "get_correlation_among_features",
    "get_categories_corr_ratio", "get_correlation_among_Xy",

    "THULACTokeniser",
]
