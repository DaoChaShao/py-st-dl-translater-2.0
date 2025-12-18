#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   evaluator.py
# @Desc     :

from functools import cache
from numpy import mean as np_mean, max as np_max, min as np_min, median as np_median
from pathlib import Path
from torch import Tensor, tensor, load, device, no_grad, long
from tqdm import tqdm

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Languages, Tokens, SeqMergeMethods, SeqStrategies
from src.nets.seq2seq_task_gru import SeqToSeqTaskGRU
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines, green
from src.utils.NLTK import bleu_score
from src.utils.nlp import SpaCyBatchTokeniser, build_word2id_seqs
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
            with SpaCyBatchTokeniser(Languages.CN, batches=batches, strict=False) as tokeniser:
                cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4prove)
            with SpaCyBatchTokeniser(Languages.EN, batches=batches, strict=False) as tokeniser:
                en_items: list[list[str]] = tokeniser.batch_tokenise(en4prove)
        else:
            with SpaCyBatchTokeniser(Languages.CN, batches=batches, strict=False) as tokeniser:
                cn_items: list[list[str]] = tokeniser.batch_tokenise(cn4prove[:amount])
            with SpaCyBatchTokeniser(Languages.EN, batches=batches, strict=False) as tokeniser:
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
        params: Path = Path(CONFIG4RNN.FILEPATHS.TRAINED_NET_BEAM)
        if params.exists():
            print(f"Model {green(params.name)} Exists!")

            # Set up a model and load saved parameters
            model = SeqToSeqTaskGRU(
                vocab_size_src=len(dictionary_cn),
                vocab_size_tgt=len(dictionary_en),
                embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
                bidirectional=True,
                accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
                PAD_SRC=dictionary_cn[Tokens.PAD],
                PAD_TGT=dictionary_en[Tokens.PAD],
                SOS=dictionary_cn[Tokens.SOS],
                EOS=dictionary_cn[Tokens.EOS],
                merge_method=SeqMergeMethods.CONCATENATE,
            )
            # dict_state: dict = load(params, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
            # model.load_state_dict(dict_state)
            model.load_model(params, strict=True)
            model.eval()
            print("Model Loaded Successfully!")

            # Convert sentences to sequence using dictionary
            sequences: list[list[int]] = build_word2id_seqs(cn_items, dictionary_cn, UNK=Tokens.UNK)
            # print(sequences[:3])

            # Predict and Evaluate
            with no_grad():
                strategy: str = (
                    SeqStrategies.GREEDY
                    if params.name.split(".")[0].split("-")[2] in str(CONFIG4RNN.FILEPATHS.TRAINED_NET_GREEDY)
                    else SeqStrategies.BEAM_SEARCH
                )

                assert len(sequences) == len(en4prove), "src and truth tgt length mismatch."
                results: list[tuple[list[str], list[str], float]] = []
                for seq, reference in tqdm(zip(sequences, en_items), total=len(sequences), desc=f"{strategy} Model:"):
                    src: Tensor = tensor(seq, dtype=long, device=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                    out = model.generate(src.unsqueeze(dim=0))
                    pred = [reversed_dict.get(idx, Tokens.UNK) for idx in out.squeeze().tolist()]
                    hypothesis: list[str] = [word.strip() for word in pred if word != Tokens.EOS]
                    # print(hypothesis)
                    # print(reference)
                    bleu = bleu_score(reference, hypothesis)
                    # print(f"BLEU Score: {bleu:.4f}")
                    results.append((reference, hypothesis, bleu))
                starts()
                print(f"Evaluation Results for {strategy} Model:")
                scores: list[float] = [score for _, _, score in results]
                lines()
                print(f"Sentence Amount: {len(scores)}")
                print(f"Mean BLEU:       {np_mean(scores):.4f}")
                print(f"Highest BLEU:    {np_max(scores):.4f}")
                print(f"Lowest BLEU:     {np_min(scores):.4f}")
                print(f"Median BLEU:     {np_median(scores):.4f}")
                starts()
                """
                ****************************************************************
                Evaluation Results for beam Model using concatenation:
                ----------------------------------------------------------------
                Sentence Amount: 4374
                Mean BLEU:       0.1446
                Highest BLEU:    1.0000
                Lowest BLEU:     0.0000
                Median BLEU:     0.0700
                ****************************************************************
                Normal
                ****************************************************************
                Evaluation Results for greedy Model:
                ----------------------------------------------------------------
                Sentence Amount: 4374
                Mean BLEU:       0.1637
                Highest BLEU:    1.0000
                Lowest BLEU:     0.0000
                Median BLEU:     0.0845
                ****************************************************************
                """
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
