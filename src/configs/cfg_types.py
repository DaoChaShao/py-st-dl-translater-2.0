#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 23:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_types.py
# @Desc     :   

from enum import StrEnum, unique
from pathlib import Path

from src.configs.cfg_base import CONFIG
from src.utils.stats import load_json


@unique
class Attentions(StrEnum):
    DOT_PRODUCT = "dot"
    GENERAL = "general"
    CONCATENATION = "concat"


@unique
class Langs(StrEnum):
    CN = "cn"
    EN = "en"


@unique
class SeqMergeMethods(StrEnum):
    AVERAGE = "average"
    CONCATENATE = "concat"


@unique
class Seq2SeqNets(StrEnum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"


@unique
class Seq2SeqStrategies(StrEnum):
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
class TSSeqSeparate(StrEnum):
    SEQ2ONE = "seq2one"
    SEQ2SEQ = "seq2seq"
    SEQ_SLICE = "slice"


if __name__ == "__main__":
    out = TSSeqSeparate.SEQ2ONE
    print(out)

    dic: Path = Path(CONFIG.FILEPATHS.DICTIONARY)
    dictionary: dict = load_json(dic)
    print(dictionary[Tokens.PAD])
