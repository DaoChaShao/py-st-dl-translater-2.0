#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :   

from numpy import array, ndarray
from pandas import DataFrame, read_csv
from pprint import pprint
from pathlib import Path
from random import randint

from src.configs.cfg_base import CONFIG
from src.utils.helper import Timer
from src.utils.SQL import SQLiteIII
from src.utils.stats import save_json


def preprocess_data() -> None:
    """ Main Function """
    with Timer("Preprocess Data"):
        path: Path = Path(CONFIG.FILEPATHS.DATA4ALL)

        if path.exists():
            print(f"Bingo! {path.name} exists!")
            print()

            # Get raw structural data
            raw: DataFrame = read_csv(path, sep="\t", header=None, names=["en", "cn"], usecols=[0, 1], encoding="utf-8")
            print(raw.head())
            print()
            idx: int = randint(0, raw.shape[0] - 1)
            pprint(raw.iloc[idx])
            print(type(raw.iloc[idx]), raw.shape)
            print()

            # Store the preprocessed data into sqlit 3 database
            table: str = "translater"
            en = raw["en"].tolist()
            cn = raw["cn"].tolist()
            cols: dict[str, type[int | str]] = {"en": str, "cn": str}
            data: dict[str, list[int | str]] = {"en": en, "cn": cn}
            with SQLiteIII(table, cols, CONFIG.FILEPATHS.SQLITE) as db:
                db.insert(data)
        else:
            print(f"{path.name} does NOT exist!")


if __name__ == "__main__":
    preprocess_data()
