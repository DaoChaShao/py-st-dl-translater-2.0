#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 19:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   processor.py
# @Desc     :   

from functools import cache
from itertools import chain
from pathlib import Path
from random import randint
from tqdm import tqdm

from src.configs.cfg_dl import CONFIG4DL
from src.configs.cfg_types import Languages, Tokens
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines
from src.utils.nlp import SpaCyBatchTokeniser, count_frequency, build_word2id_seqs, check_vocab_coverage
from src.utils.stats import create_full_data_split, save_json
from src.utils.SQL import SQLiteIII


@cache
def connect_db() -> list[tuple]:
    """ Connect to the Database and Return the Connection Object """
    table: str = "translater"
    cols: dict = {"en": str, "cn": str}
    path: Path = Path(CONFIG4DL.FILEPATHS.SQLITE)

    with SQLiteIII(table, cols, path) as db:
        data = db.fetch_all(col_names=[col for col in cols.keys()])

    return data


def process_data() -> tuple[list, list, list, list]:
    """ Main Function """
    with Timer("Process Data"):
        # Get the data from the database
        data: list[tuple] = connect_db()
        # pprint(data[:3])
        # print(len(data))

        # Separate the data
        data4train, data4valid, _ = create_full_data_split(data)
        # print(data4train[:3])
        # print(data4valid[:3])

        # Set a dictionary
        # amount: int | None = 100
        amount: int | None = None
        batches: int = 16 if amount else 128
        cn4train: list[str] = [c for _, c in data4train]
        en4train: list[str] = [e for e, _ in data4train]
        cn4valid: list[str] = [c for _, c in data4valid]
        en4valid: list[str] = [e for e, _ in data4valid]
        if amount is None:
            with SpaCyBatchTokeniser(Languages.CN, batches=batches, strict=False) as tokeniser:
                train_cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4train)
                valid_cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4valid)
            with SpaCyBatchTokeniser(Languages.EN, batches=batches, strict=False) as tokeniser:
                train_en_items: list[list[str]] = tokeniser.batch_tokenise(en4train)
                valid_en_items: list[list[str]] = tokeniser.batch_tokenise(en4valid)
        else:
            with SpaCyBatchTokeniser(Languages.CN, batches=batches, strict=False) as tokeniser:
                train_cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4train[:amount])
                valid_cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4valid[:amount])
            with SpaCyBatchTokeniser(Languages.EN, batches=batches, strict=False) as tokeniser:
                train_en_items: list[list[str]] = tokeniser.batch_tokenise(en4train[:amount])
                valid_en_items: list[list[str]] = tokeniser.batch_tokenise(en4valid[:amount])
        # print(train_cn_items[:3])
        # print(train_en_items[:3])
        # print(valid_cn_items[:3])
        # print(valid_en_items[:3])

        # Count the frequency of the words
        # Method I
        # all_cn_items: list[str] = [item for items in train_cn_items for item in items] + [item for items in valid_cn_items for item in items]
        # all_en_items: list[str] = [item for items in train_en_items for item in items] + [item for items in valid_en_items for item in items]
        # Method II
        cn_items: list[str] = list(chain.from_iterable(train_cn_items + valid_cn_items))
        en_items: list[str] = list(chain.from_iterable(train_en_items + valid_en_items))
        # print(all_cn_items[:10])
        # print(all_en_items[:10])
        cn_tokens, _ = count_frequency(cn_items, top_k=10, freq_threshold=1)
        en_tokens, _ = count_frequency(en_items, top_k=10, freq_threshold=1)
        # print(cn_tokens[:10])
        # print(en_tokens[:10])

        # Build a Chinese dictionary
        special: list[str] = [Tokens.PAD, Tokens.UNK, Tokens.SOS, Tokens.EOS]
        # print(special)
        dictionary_cn: dict[str, int] = {
            word: i for i, word in
            tqdm(enumerate(special + cn_tokens), total=len(special + cn_tokens), desc=f"Building {Languages.CN} dictionary")
        }
        save_json(dictionary_cn, CONFIG4DL.FILEPATHS.DICTIONARY_CN)
        dic: Path = Path(CONFIG4DL.FILEPATHS.DICTIONARY_CN)
        print(f"{Languages.CN} Dictionary Saved Successfully!") if dic.exists() else print("Dictionary NOT Saved!")
        print()
        # Build an English dictionary
        dictionary_en: dict[str, int] = {
            word: i for i, word in
            tqdm(enumerate(special + en_tokens), total=len(special + en_tokens), desc=f"Building {Languages.EN} dictionary")
        }
        save_json(dictionary_en, CONFIG4DL.FILEPATHS.DICTIONARY_EN)
        dic: Path = Path(CONFIG4DL.FILEPATHS.DICTIONARY_EN)
        print(f"{Languages.EN} Dictionary Saved Successfully!") if dic.exists() else print("Dictionary NOT Saved!")
        print()

        # Build sequence for train
        train_cn_seq: list[list[int]] = build_word2id_seqs(train_cn_items, dictionary_cn, add_sos_eos=False)
        train_en_seq: list[list[int]] = build_word2id_seqs(train_en_items, dictionary_en, add_sos_eos=True)
        # idx4train: int = randint(0, len(train_cn_seq) - 1)
        # print(train_cn_seq[idx4train])
        # print(train_en_seq[idx4train])
        # Build sequence for valid
        valid_cn_seq: list[list[int]] = build_word2id_seqs(valid_cn_items, dictionary_cn, add_sos_eos=False)
        valid_en_seq: list[list[int]] = build_word2id_seqs(valid_en_items, dictionary_en, add_sos_eos=True)
        # idx4valid: int = randint(0, len(valid_cn_seq) - 1)
        # print(valid_cn_seq[idx4valid])
        # print(valid_en_seq[idx4valid])
        # Build sequence for all
        seq4cn: list[list[int]] = train_cn_seq + valid_cn_seq
        seq4en: list[list[int]] = train_en_seq + valid_en_seq
        # idx4all: int = randint(0, len(seq4cn) - 1)
        # print(seq4cn[idx4all])
        # print(seq4en[idx4all])

        # Get the train dataset sentences description
        lengths: list[int] = [len(seq) for seq in seq4cn]
        max_len: int = max(lengths)
        min_len: int = min(lengths)
        avg_len: float = sum(lengths) / len(lengths)
        # Check the coverage of train data
        check_vocab_coverage([item for items in train_cn_items for item in items], dictionary_cn)
        # Check the coverage of valid data
        check_vocab_coverage([item for items in valid_cn_items for item in items], dictionary_cn)

        starts()
        print("Data Processing Summary:")
        lines()
        print(f"Train dataset: {len(data4train)} Samples")
        print(f"Valid dataset: {len(data4valid)} Samples")
        print(f"Chinese Dictionary Size: {len(dictionary_cn)}")
        print(f"English Dictionary Size: {len(dictionary_en)}")
        print(f"The min length of the sequence: {min_len}")
        print(f"The average length of the sequence: {avg_len:.2f}")
        print(f"The max length of the sequence: {max_len}")
        starts()
        print()
        """
        ****************************************************************
        Data Processing Summary:
        ----------------------------------------------------------------
        Train dataset: 20408 Samples
        Valid dataset: 4373 Samples
        Chinese Dictionary Size: 5235
        English Dictionary Size: 3189
        The min length of the sequence: 1
        The average length of the sequence: 6.93
        The max length of the sequence: 28
        ****************************************************************
        """

        return train_cn_seq, valid_cn_seq, train_en_seq, valid_en_seq


if __name__ == "__main__":
    process_data()
