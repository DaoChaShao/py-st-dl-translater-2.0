#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from functools import cache
from pathlib import Path
from random import randint
from torch import Tensor, no_grad

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Languages, Tokens, SeqMergeMethods, SeqStrategies, AttnScorer
from src.nets.seq2seq_attn_gru import SeqToSeqGRUWithAttn
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
        params: Path = Path(CONFIG4RNN.FILEPATHS.NET_TRUE_GRU_WITH_ATTN_BAHDANAU_BEAM_CONCAT)
        if params.exists():
            print(f"Model {green(params.name)} Exists!")

            # Set up a model and load saved parameters
            model = SeqToSeqGRUWithAttn(
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
                merge_method=SeqMergeMethods.CONCAT,
                teacher_forcing_ratio=0.5,
                use_attention=True,
                attn_scorer=AttnScorer.BAHDANAU
            )
            model.load_model(params, strict=True)
            model.eval()
            print("Model Loaded Successfully!")

            # Convert sentences to sequence using dictionary
            sequences: list[list[int]] = build_word2id_seqs(cn_items, dictionary_cn, UNK=Tokens.UNK)
            # print(sequences[:3])

            # Randomly select a data point for prediction
            assert len(sequences) == len(en_items), "src and truth tgt length mismatch."
            idx: int = randint(0, len(sequences) - 1)
            # ----------------------------------------------------------------
            seq: list[int] = sequences[idx]
            # seq: list[int] = sequences[2102]
            # seq: list[int] = sequences[3280]
            # ----------------------------------------------------------------
            # Convert the token to a tensor
            src: Tensor = item2tensor(seq, embedding=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
            # Add batch size
            src = src.unsqueeze(0)
            # print(src.shape, src)

            # Prediction
            with no_grad():
                strategy: str = params.name.split("-")[6]

                out: Tensor = model.generate(src)
                pred = [reversed_dict.get(idx, Tokens.UNK) for idx in out.squeeze().tolist()]
                hypothesis: list[str] = [word.strip() for word in pred if word != Tokens.EOS]
                # Get the relevant reference
                # ----------------------------------------------------------------
                reference: list[str] = en_items[idx]
                # reference: list[str] = en_items[2102]
                # reference: list[str] = en_items[3280]
                # ----------------------------------------------------------------

                bleu = bleu_score(reference, hypothesis)
                starts()
                print(f"Evaluation Results for {strategy} Model:")
                lines()
                # ----------------------------------------------------------------
                print(f"Selected Data Index for Prediction: {red(str(idx))}")
                print(f"Input Sentence (CN):                {cn4prove[idx]}")
                # print(f"Selected Data Index for Prediction: {red(str(2102))}")
                # print(f"Input Sentence (CN):                {cn4prove[2102]}")
                # print(f"Selected Data Index for Prediction: {red(str(3280))}")
                # print(f"Input Sentence (CN):                {cn4prove[3280]}")
                # ----------------------------------------------------------------
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
                ****************************************************************
                Evaluation Results for beam Model without attention:
                ----------------------------------------------------------------
                Selected Data Index for Prediction: 3280
                Input Sentence (CN):                你裁切纸张了吗？
                Reference (EN):                     ['do', 'you', 'cut', 'the', 'paper', '?']
                Predation (EN):                     ['do', 'you', 'have', 'any', '<UNK>', 'on', 'your', 'shoe', '?']
                BLEU Score:                         0.0700
                ****************************************************************
                ****************************************************************
                Evaluation Results for beam Model with bahdanau attention:
                ----------------------------------------------------------------
                Selected Data Index for Prediction: 3280
                Input Sentence (CN):                你裁切纸张了吗？
                Reference (EN):                     ['do', 'you', 'cut', 'the', 'paper', '?']
                Predation (EN):                     ['be', 'you', '<UNK>', 'on', 'the', '?']
                BLEU Score:                         0.0495
                ****************************************************************
                """
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
