#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from functools import cache
from pathlib import Path
from random import randint, choice
from torch import Tensor, load, device, no_grad

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Lang, Tokens, Seq2SeqNet, Seq2SeqStrategies
from src.nets.seq2seq import SeqToSeqCoder
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines, red, green, blue
from src.utils.NLTK import bleu_score
from src.utils.nlp import SpaCyBatchTokeniser, build_word2id_seqs
from src.utils.PT import item2tensor
from src.utils.SQL import SQLiteIII
from src.utils.stats import create_full_data_split, load_json


@cache
def connect_db() -> list[tuple]:
    """ Connect to the Database and Return the Connection Object """
    table: str = "translater"
    cols: dict = {"en": str, "cn": str}
    path: Path = Path(CONFIG4RNN.FILEPATHS.SQLITE)

    with SQLiteIII(table, cols, path) as db:
        data = db.fetch_all(col_names=[col for col in cols.keys()])

    return data


def main() -> None:
    """ Main Function """
    # Get the data from the database
    database: list[tuple] = connect_db()
    # pprint(data[:3])
    # print(len(data))

    with Timer("Next Word Prediction"):
        # Separate the data
        _, _, data = create_full_data_split(database)

        # Tokenise the data
        # amount: int | None = 100
        amount: int | None = None
        batches: int = 16 if amount else 128
        cn4prove: list[str] = [c for _, c in data]
        en4prove: list[str] = [e for e, _ in data]
        assert len(cn4prove) == len(en4prove), "Chinese and English data length mismatch."
        if amount is None:
            with SpaCyBatchTokeniser(Lang.CN, batches=batches, strict=False) as tokeniser:
                cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4prove)
            with SpaCyBatchTokeniser(Lang.EN, batches=batches, strict=False) as tokeniser:
                en_items: list[list[str]] = tokeniser.batch_tokenise(en4prove)
        else:
            with SpaCyBatchTokeniser(Lang.CN, batches=batches, strict=False) as tokeniser:
                cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4prove[:amount])
            with SpaCyBatchTokeniser(Lang.EN, batches=batches, strict=False) as tokeniser:
                en_items: list[list[str]] = tokeniser.batch_tokenise(en4prove[:amount])
        # print(cn_items[:3])
        # print(en_items[:3])

        # Load dictionary
        dic_cn: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_CN)
        dictionary_cn: dict = load_json(dic_cn) if dic_cn.exists() else print("Dictionary file not found.")
        dic_en: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_EN)
        dictionary_en: dict = load_json(dic_en) if dic_en.exists() else print("Dictionary file not found.")
        reversed_dict: dict = {idx: word for word, idx in dictionary_en.items()}
        # print(reversed_dict)

        starts()
        print("Data Preprocessing Summary:")
        lines()
        print(f"Chinese Dictionary Size: {len(dictionary_cn)} Samples")
        print(f"English Dictionary Size: {len(dictionary_en)} Samples")
        starts()
        """
        ****************************************************************
        Data Preprocessing Summary:
        ----------------------------------------------------------------
        Chinese Dictionary Size: 5235 Samples
        English Dictionary Size: 3189 Samples
        ****************************************************************
        """

        # Load the save model parameters
        options: list[str] = [CONFIG4RNN.FILEPATHS.TRAINED_NET_GREEDY, CONFIG4RNN.FILEPATHS.TRAINED_NET_BEAM]
        selection: str = choice(options)
        params: Path = Path(selection)
        if params.exists():
            print(f"Model {green(params.name)} Exists!")

            # Set up a model and load saved parameters
            model = SeqToSeqCoder(
                len(dictionary_cn),
                len(dictionary_en),
                embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
                bid=True,
                pad_idx4input=dictionary_cn[Tokens.PAD],
                pad_idx4output=dictionary_en[Tokens.PAD],
                net_category=Seq2SeqNet.GRU,
                SOS=dictionary_cn[Tokens.SOS],
                EOS=dictionary_en[Tokens.EOS],
            )
            dict_state: dict = load(params, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
            model.load_state_dict(dict_state)
            model.eval()
            print("Model Loaded Successfully!")

            # Convert sentences to sequence using dictionary
            sequences: list[list[int]] = build_word2id_seqs(cn_items, dictionary_cn, UNK=Tokens.UNK)
            # print(sequences[:3])

            # Randomly select a data point for prediction
            assert len(sequences) == len(en_items), "src and truth tgt length mismatch."
            idx: int = randint(0, len(sequences) - 1)
            seq: list[int] = sequences[idx]
            # Convert the token to a tensor
            src: Tensor = item2tensor(seq, embedding=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
            # Add batch size
            src = src.unsqueeze(0)
            # print(src.shape, src)

            # Prediction
            with no_grad():
                strategy: str = (
                    Seq2SeqStrategies.GREEDY
                    if params.name.split(".")[0].split("-")[2] in str(CONFIG4RNN.FILEPATHS.TRAINED_NET_GREEDY)
                    else Seq2SeqStrategies.BEAM_SEARCH
                )

                out: Tensor = model.generate(src)
                pred = [reversed_dict.get(idx, Tokens.UNK) for idx in out.squeeze().tolist()]
                hypothesis: list[str] = [word.strip() for word in pred if word != Tokens.EOS]
                # Get the relevant reference
                reference: list[str] = en_items[idx]

                bleu = bleu_score(reference, hypothesis)
                starts()
                print(f"Evaluation Results for {strategy} Model:")
                lines()
                print(f"Selected Data Index for Prediction: {red(str(idx))}")
                print(f"Input Sentence (CN):                {cn4prove[idx]}")
                print(f"Reference (EN):                     {reference}")
                print(f"Predation (EN):                     {hypothesis}")
                print(f"BLEU Score:                         {blue(f"{bleu:.4f}")}")
                starts()
                """
                ****************************************************************
                Evaluation Results for beam Model:
                ----------------------------------------------------------------
                Selected Data Index for Prediction: 2102
                Input Sentence (CN):                他喜欢听收音机。
                Reference (EN):                     ['he', 'like', 'listen', 'to', 'the', 'radio', '.']
                Predation (EN):                     ['he', 'like', 'to', 'listen', 'to', 'the', 'radio', '.']
                BLEU Score:                         0.5946
                ****************************************************************
                """
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
