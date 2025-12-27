#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:36
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_base.py
# @Desc     :   

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class FilePaths:
    API_KEY: Path = BASE_DIR / "data/api_keys.yaml"
    DATA4ALL: Path = BASE_DIR / "data/cmn.txt"
    DATA4TRAIN: Path = BASE_DIR / "data/train/"
    DATA4TEST: Path = BASE_DIR / "data/test/"
    DICTIONARY: Path = BASE_DIR / "data/dictionary.json"
    DICTIONARY_CN: Path = BASE_DIR / "data/dictionary_cn.json"
    DICTIONARY_EN: Path = BASE_DIR / "data/dictionary_en.json"
    LOGS: Path = BASE_DIR / "logs/"
    SAVED_NET: Path = BASE_DIR / "models/model.pth"

    NET_FALSE_GRU_BEAM_CONCAT: Path = BASE_DIR / "models/model-false-gru-beam-concat.pth"
    NET_FALSE_GRU_BEAM_MEAN: Path = BASE_DIR / "models/model-false-gru-beam-mean.pth"
    NET_FALSE_GRU_GREEDY_CONCAT: Path = BASE_DIR / "models/model-false-gru-greedy-concat.pth"
    NET_FALSE_GRU_GREEDY_MEAN: Path = BASE_DIR / "models/model-false-gru-greedy-mean.pth"
    NET_FALSE_LSTM_BEAM_CONCAT: Path = BASE_DIR / "models/model-false-lstm-beam-concat.pth"
    NET_FALSE_RNN_BEAM_CONCAT: Path = BASE_DIR / "models/model-false-rnn-beam-concat.pth"
    NET_TRUE_GRU_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-gru-beam-concat.pth"
    NET_TRUE_GRU_BEAM_MEAN: Path = BASE_DIR / "models/model-true-gru-beam-mean.pth"
    NET_TRUE_GRU_GREEDY_CONCAT: Path = BASE_DIR / "models/model-true-gru-greedy-concat.pth"
    NET_TRUE_GRU_GREEDY_MEAN: Path = BASE_DIR / "models/model-true-gru-greedy-mean.pth"
    NET_TRUE_GRU_WITH_ATTN_BAHDANAU_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-gru-with-attn-bahdanau-beam-concat.pth"
    NET_TRUE_GRU_WITH_ATTN_DOT_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-gru-with-attn-dot-beam-concat.pth"
    NET_TRUE_GRU_WITH_ATTN_SDOT_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-gru-with-attn-sdot-beam-concat.pth"
    NET_TRUE_GRU_WITHOUT_ATTN_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-gru-without-attn-beam-concat.pth"
    NET_TRUE_LSTM_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-lstm-beam-concat.pth"
    NET_TRUE_LSTM_WITH_ATTN_DOT_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-lstm-with-attn-dot-beam-concat.pth"
    NET_TRUE_LSTM_WITHOUT_ATTN_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-lstm-without-attn-beam-concat.pth"
    NET_TRUE_RNN_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-rnn-beam-concat.pth"
    NET_TRUE_RNN_WITH_ATTN_DOT_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-rnn-with-attn-dot-beam-concat.pth"
    NET_TRUE_RNN_WITHOUT_ATTN_BEAM_CONCAT: Path = BASE_DIR / "models/model-true-rnn-without-attn-beam-concat.pth"

    SPACY_MODEL_EN: Path = BASE_DIR / "models/spacy/en_core_web_md"
    SPACY_MODEL_CN: Path = BASE_DIR / "models/spacy/zh_core_web_md"
    SQLITE: Path = BASE_DIR / "data/sqlite3.db"


@dataclass
class Database:
    USER: str = ""
    PASSWORD: str = ""
    HOST: str = ""
    PORT: str = ""


@dataclass
class Punctuations:
    CN = [
        "，", "。", "？", "！", "、", "；", "：", "「", "」", "『", "』",
        "《", "》", "（", "）", "【", "】", "｛", "｝", "－", "～", "·",
        "…", "——", "〝", "〞", "＂", "＇", "＇", "‘", "’", "“", "”",
        "〈", "〉", "〖", "〗", "〔", "〕", "〘", "〙", "〚", "〛"
    ]
    EN = [
        ",", ".", "?", "!", ";", ":", "'", '"', "(", ")", "[", "]",
        "{", "}", "-", "~", "`", "@", "#", "$", "%", "^", "&", "*",
        "_", "+", "=", "<", ">", "/", "\\", "|"
    ]


@dataclass
class Config:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    PUNCTUATIONS: Punctuations = field(default_factory=Punctuations)


CONFIG = Config()
