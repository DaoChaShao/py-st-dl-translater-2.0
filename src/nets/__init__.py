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
+ Base Classes: Fundamental network abstractions
    - BaseRNN: Base class for Recurrent Neural Networks
    - BaseSeqNet: Base class for general sequence networks
    - BaseAttn: Base class for attention mechanisms
+ Attention Modules: Enhanced sequence modeling
    - AdditiveAttention
    - DotProductAttention
    - ScaledDotProductAttention
+ Multi-Task Learning: RNN/LSTM/GRU variants supporting multiple outputs
    - MultiTaskRNN
    - MultiTaskLSTM
    - MultiTaskGRU
+ Sequence-to-Sequence Architectures: Encoder-decoder networks
    - SeqToSeqCoder
    - SeqToSeqRNN
    - SeqToSeqLSTM
    - SeqToSeqGRU
+ Encoder & Decoder Components:
    - SeqEncoder
    - SeqDecoder
    - SeqAttnDecoder
+ UNet Variants for Semantic Segmentation:
    - Standard4LayersUNetClassification
    - Standard5LayersUNetForClassification

Usage:
+ Direct import of models via:
    - from src.nn import Standard4LayersUNetClassification, LSTMRNNForClassification, etc.
+ Instantiate models with default or custom parameters as needed.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.3.0"

from .attentions import AdditiveAttention, DotProductAttention, ScaledDotProductAttention
from .base_attn import BaseAttn
from .base_rnn import BaseRNN
from .base_seq2seq import BaseSeqNet
from .multi_task_gru import MultiTaskGRU
from .multi_task_lstm import MultiTaskLSTM
from .multi_task_rnn import MultiTaskRNN
from .seq2seq import SeqToSeqCoder
from .seq2seq_attn_gru import SeqToSeqGRUWithAttn
from .seq2seq_attn_lstm import SeqToSeqLSTMWithAttn
from .seq2seq_attn_rnn import SeqToSeqRNNWithAttn
from .seq2seq_task_gru import SeqToSeqGRU
from .seq2seq_task_lstm import SeqToSeqLSTM
from .seq2seq_task_rnn import SeqToSeqRNN
from .seq_decoder import SeqDecoder
from .seq_decoder4attn import SeqDecoderWithAttn
from .seq_encoder import SeqEncoder
from .unet4layers4sem import Standard4LayersUNetClassification
from .unet5layers4sem import Standard5LayersUNetForClassification

__all__ = [
    "AdditiveAttention",
    "DotProductAttention",
    "ScaledDotProductAttention",

    "BaseAttn",
    "BaseRNN",
    "BaseSeqNet",

    "MultiTaskGRU",
    "MultiTaskLSTM",
    "MultiTaskRNN",

    "SeqToSeqGRUWithAttn",
    "SeqToSeqLSTMWithAttn",
    "SeqToSeqRNNWithAttn",

    "SeqToSeqCoder",

    "SeqToSeqGRU",
    "SeqToSeqLSTM",
    "SeqToSeqRNN",

    "SeqDecoder",
    "SeqDecoderWithAttn",
    "SeqEncoder",

    "Standard4LayersUNetClassification",
    "Standard5LayersUNetForClassification",
]
