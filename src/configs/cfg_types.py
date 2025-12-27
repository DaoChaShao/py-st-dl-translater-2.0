#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 23:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_types.py
# @Desc     :   

from enum import StrEnum, unique


@unique
class AttnHeads(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


@unique
class AttnScorer(StrEnum):
    BAHDANAU = "bahdanau"
    DOT_PRODUCT = "dot"
    SCALED_DOT_PRODUCT = "sdot"


@unique
class Languages(StrEnum):
    CN = "cn"
    EN = "en"


@unique
class SeqMergeMethods(StrEnum):
    CONCAT = "concat"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"


@unique
class SeqNets(StrEnum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"


@unique
class SeqStrategies(StrEnum):
    GREEDY = "greedy"
    BEAM_SEARCH = "beam"


@unique
class Tasks(StrEnum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"


@unique
class Tokens(StrEnum):
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"  # Or, call it BOS (Beginning of Sequence)
    EOS = "<EOS>"
    MASK = "<MASK>"
    BOS = "<BOS>"  # Beginning of Sequence


@unique
class SeqSeparator(StrEnum):
    SEQ2ONE = "seq2one"
    SEQ2SEQ = "seq2seq"
    SEQ_SLICE = "slice"


if __name__ == "__main__":
    out = SeqSeparator.SEQ2ONE
    print(out)
