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
+ BaseRNN: Base class for Recurrent Neural Networks
+ BaseSeqNet: Base class for Sequence Networks
+ MultiTask Models: Multi-task learning architectures (RNN, LSTM, GRU)
+ Seq2Seq Models: Encoder-decoder architectures for sequence-to-sequence tasks
+ SeqEncoder/SeqDecoder: Individual encoder and decoder components
+ Attention Mechanisms: Single and multi-head attention modules for enhanced sequence modeling
+ Standard4LayersUNetClassification: 4-layer UNet variant for semantic segmentation
+ Standard5LayersUNetForClassification: 5-layer UNet variant for semantic segmentation

Usage:
+ Direct import of models via:
    - from src.nn import Standard4LayersUNetClassification, LSTMRNNForClassification, etc.
+ Instantiate models with default or custom parameters as needed.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.3.0"

from .attention_multi import MultiHeadAttention
from .attention_single import SingleHeadAttention
from .base_rnn import BaseRNN
from .base_seq import BaseSeqNet
from .multi_task_gru import MultiTaskGRU
from .multi_task_lstm import MultiTaskLSTM
from .multi_task_rnn import MultiTaskRNN
from .seq2seq import SeqToSeqCoder
from .seq2seq_attn_gru import AttnGRUForSeqToSeq
from .seq2seq_attn_lstm import AttnLSTMForSeqToSeq
from .seq2seq_attn_rnn import AttnRNNForSeqToSeq
from .seq2seq_task_gru import SeqToSeqTaskGRU
from .seq2seq_task_lstm import SeqToSeqTaskLSTM
from .seq2seq_task_rnn import SeqToSeqTaskRNN
from .seq_encoder import SeqEncoder
from .seq_decoder import SeqDecoder
from .unet4layers4sem import Standard4LayersUNetClassification
from .unet5layers4sem import Standard5LayersUNetForClassification

__all__ = [
    "MultiHeadAttention",
    "SingleHeadAttention",
    "BaseRNN",
    "BaseSeqNet",
    "MultiTaskGRU",
    "MultiTaskLSTM",
    "MultiTaskRNN",
    "SeqToSeqCoder",
    "AttnGRUForSeqToSeq",
    "AttnLSTMForSeqToSeq",
    "AttnRNNForSeqToSeq",
    "SeqToSeqTaskGRU",
    "SeqToSeqTaskLSTM",
    "SeqToSeqTaskRNN",
    "SeqEncoder",
    "SeqDecoder",
    "Standard4LayersUNetClassification",
    "Standard5LayersUNetForClassification",
]
