#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prepper.py
# @Desc     :   

from pathlib import Path
from random import randint
from tqdm import tqdm

from src.configs import Tokens
from src.configs.cfg_dl import CONFIG4DL
from src.datasets.dataset4torch import TorchDataset
from src.dataloaders.dataloader4torch import TorchDataLoader
from src.utils.stats import load_json

from pipeline.processor import process_data


def prepare_data() -> tuple[TorchDataLoader, TorchDataLoader]:
    """ Prepare data """
    # Get data
    features4train, features4valid, labels4train, labels4valid = process_data()
    assert len(features4train) == len(labels4train), "Training features and labels length mismatch."
    assert len(features4valid) == len(labels4valid), "Validation features and labels length mismatch."

    # Load dictionary
    dic_cn: Path = Path(CONFIG4DL.FILEPATHS.DICTIONARY_CN)
    dictionary_cn: dict = load_json(dic_cn) if dic_cn.exists() else print("Dictionary file not found.")
    dic_en: Path = Path(CONFIG4DL.FILEPATHS.DICTIONARY_EN)
    dictionary_en: dict = load_json(dic_en) if dic_en.exists() else print("Dictionary file not found.")

    # Set dataset
    dataset4train = TorchDataset(features4train, labels4train, use_batch_pad=True)
    dataset4valid = TorchDataset(features4valid, labels4valid, use_batch_pad=True)
    # idx4train: int = randint(0, len(dataset4train) - 1)
    # print(dataset4train[idx4train])
    # idx4valid: int = randint(0, len(dataset4valid) - 1)
    # print(dataset4valid[idx4valid])
    # print()

    # Set up dataloader
    dataloader4train = TorchDataLoader(
        dataset4train,
        batch_size=CONFIG4DL.PREPROCESSOR.BATCHES,
        shuffle_state=CONFIG4DL.PREPROCESSOR.SHUFFLE,
        workers=CONFIG4DL.PREPROCESSOR.WORKERS,
        use_batch_pad=True,
        FEATURES_PAD_VALUE=dictionary_cn[Tokens.PAD],
        LABELS_PAD_VALUE=dictionary_cn[Tokens.PAD],
    )
    dataloader4valid = TorchDataLoader(
        dataset4valid,
        batch_size=CONFIG4DL.PREPROCESSOR.BATCHES,
        shuffle_state=CONFIG4DL.PREPROCESSOR.SHUFFLE,
        workers=CONFIG4DL.PREPROCESSOR.WORKERS,
        use_batch_pad=True,
        FEATURES_PAD_VALUE=dictionary_cn[Tokens.PAD],
        LABELS_PAD_VALUE=dictionary_cn[Tokens.PAD],
    )
    # for feature, label in tqdm(
    #         dataloader4train._loader,
    #         total=len(dataloader4train),
    #         desc="Sample a batch from training data"
    # ):
    #     print(feature, label)
    #     break

    return dataloader4train, dataloader4valid


if __name__ == "__main__":
    prepare_data()
