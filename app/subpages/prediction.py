#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prediction.py
# @Desc     :   

from functools import cache
from pathlib import Path
from random import randint
from streamlit import (empty, sidebar, subheader, session_state,
                       button, container, rerun, columns, caption,
                       markdown, write, selectbox, data_editor, text_input, )
from torch import load, device, Tensor, no_grad

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Languages, Tokens, SeqNets, SeqStrategies, SeqMergeMethods, AttnScorer
from src.nets.seq2seq_attn_gru import SeqToSeqGRUWithAttn
from src.utils.helper import Timer
from src.utils.NLTK import bleu_score
from src.utils.nlp import SpaCyBatchTokeniser, build_word2id_seqs
from src.utils.PT import item2tensor
from src.utils.SQL import SQLiteIII
from src.utils.stats import load_json, create_full_data_split


@cache
def connect_db() -> list[tuple]:
    """ Connect to the Database and Return the Connection Object """
    table: str = "translater"
    cols: dict = {"en": str, "cn": str}
    path: Path = Path(CONFIG4RNN.FILEPATHS.SQLITE)

    with SQLiteIII(table, cols, path) as db:
        data = db.fetch_all(col_names=[col for col in cols.keys()])

    return data


def map_bleu_to_benchmarks(bleu: float) -> list[tuple[str, str]]:
    BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
        "news": {
            "excellent": (0.35, 0.45),
            "good": (0.25, 0.35),
            "acceptable": (0.15, 0.25),
            "poor": (0.0, 0.15)
        },
        "documentation": {
            "excellent": (0.40, 0.50),
            "good": (0.30, 0.40),
            "acceptable": (0.20, 0.30),
            "poor": (0.0, 0.20)
        },
        "conversation": {
            "excellent": (0.25, 0.35),
            "good": (0.15, 0.25),
            "acceptable": (0.05, 0.15),
            "poor": (0.0, 0.05)
        }
    }

    rating: list[tuple[str, str]] = []
    for domain, benchmarks in BENCHMARKS.items():
        for quality, (low, high) in benchmarks.items():
            if low <= bleu < high:
                rating.append((domain, quality))
                break
            else:
                continue
    return rating


display4bar: container = container(width="stretch")
empty_messages: empty = empty()
interpreter: empty = empty()
original, prediction = columns(2, gap="medium", vertical_alignment="center", width="stretch")
display4data_title: empty = empty()
display4data = container(border=1, width="stretch")
dict4en, dict4cn = columns(2, gap="medium", vertical_alignment="center", width="stretch")

session4init: list[str] = ["model", "cn4prove", "cn_items", "en_items", "timer4init"]
for session in session4init:
    session_state.setdefault(session, None)
session4pick: list[str] = ["src", "reference", "idx", "timer4pick"]
for session in session4pick:
    session_state.setdefault(session, None)
session4pred: list[str] = ["hypothesis", "timer4pred"]
for session in session4pred:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Translater Settings")

    # Load model parameters
    params4beam: Path = Path(CONFIG4RNN.FILEPATHS.NET_TRUE_GRU_WITH_ATTN_BAHDANAU_BEAM_CONCAT)
    # Load dictionary
    dic_cn: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_CN)
    dictionary_cn: dict[str, int] = load_json(dic_cn) if dic_cn.exists() else print("Dictionary file not found.")
    dic_en: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_EN)
    dictionary_en: dict[str, int] = load_json(dic_en) if dic_en.exists() else print("Dictionary file not found.")
    reversed_dict: dict[int, str] = {idx: word for word, idx in dictionary_en.items()}
    # print(reversed_dict)
    if params4beam.exists() and dic_cn.exists() and dic_en.exists():
        empty_messages.warning("The model & dictionary file already exists. You can initialise model first.")

        if session_state["model"] is None:
            # Set model selection
            model: str = selectbox(
                "Select a Model",
                options=[SeqNets.RNN, SeqNets.LSTM, SeqNets.GRU], index=2,
                disabled=True,
                width="stretch"
            )
            caption(f"You selected **{model}** for translation.")
            # Set strategy selection
            selection: str = selectbox(
                "Select a Strategy to translate",
                options=[SeqStrategies.GREEDY, SeqStrategies.BEAM_SEARCH], index=1,
                disabled=True,
                width="stretch"
            )
            caption(f"You selected **{selection} search** strategy for translation.")

            if button("Initialise Model & Dictionary & Data", type="primary", width="stretch"):
                with Timer("Initialisation") as session_state["timer4init"]:
                    # Initialise a model and load saved parameters
                    session_state["model"] = SeqToSeqGRUWithAttn(
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
                    dict_state: dict = load(params4beam, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                    session_state["model"].load_state_dict(dict_state)
                    session_state["model"].eval()
                    print("Model Loaded Successfully!")

                    # Initialise the test data from sqlite database
                    database: list[tuple] = connect_db()
                    # pprint(data[:3])
                    # print(len(data))
                    # Separate the data
                    _, _, data = create_full_data_split(database)

                    # Tokenise the data
                    with display4bar:
                        # Tokenise the data
                        # amount: int | None = 100
                        amount: int | None = None
                        batches: int = 16 if amount else 128
                        session_state["cn4prove"]: list[str] = [c for _, c in data]
                        en4prove: list[str] = [e for e, _ in data]
                        if amount is None:
                            with SpaCyBatchTokeniser(Languages.CN, batches=batches, strict=False) as tokeniser:
                                session_state["cn_items"]: list[list[str]] = tokeniser.batch_tokenise(
                                    session_state["cn4prove"], streamlit_bar=True
                                )
                            with SpaCyBatchTokeniser(Languages.EN, batches=batches, strict=False) as tokeniser:
                                session_state["en_items"]: list[list[str]] = tokeniser.batch_tokenise(
                                    en4prove, streamlit_bar=True
                                )
                        else:
                            with SpaCyBatchTokeniser(Languages.CN, batches=batches, strict=False) as tokeniser:
                                session_state["cn_items"]: list[list[str]] = tokeniser.batch_tokenise(
                                    session_state["cn4prove"][:amount], streamlit_bar=True
                                )
                            with SpaCyBatchTokeniser(Languages.EN, batches=batches, strict=False) as tokeniser:
                                session_state["en_items"]: list[list[str]] = tokeniser.batch_tokenise(
                                    en4prove[:amount], streamlit_bar=True
                                )
                        # print(cn_items[:3])
                        # print(en_items[:3])
                        rerun()
        else:
            empty_messages.info(f"Initialisation completed! {session_state["timer4init"]} Pick up a data to test.")

            with dict4en:
                markdown(f"**English Dictionary {len(dictionary_en)}**")
                data_editor(dictionary_en, hide_index=False, disabled=True, width="stretch")
            with dict4cn:
                markdown(f"**Chinese Dictionary {len(dictionary_cn)}**")
                data_editor(dictionary_cn, hide_index=False, disabled=True, width="stretch")

            if session_state["src"] is None and session_state["reference"] is None:
                if button("Pick up a Data", type="primary", width="stretch"):
                    with Timer("Pick a piece of data") as session_state["timer4pick"]:
                        # Pick up a random sequence token converting a random sentence to sequence using dictionary
                        sequences: list[list[int]] = build_word2id_seqs(
                            session_state["cn_items"], dictionary_cn, UNK=Tokens.UNK
                        )
                        # print(sequences[:3])
                        # Randomly select a data point for prediction
                        assert len(sequences) == len(session_state["en_items"]), "src and truth tgt length mismatch."
                        session_state["idx"]: int = randint(0, len(sequences) - 1)
                        seq: list[int] = sequences[session_state["idx"]]

                        # Convert the token to a tensor
                        src: Tensor = item2tensor(
                            seq, embedding=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR
                        )
                        # Add batch size
                        session_state["src"] = src.unsqueeze(0)
                        # print(session_state["src"])
                        # Get the relevant label
                        session_state["reference"]: list[str] = session_state["en_items"][session_state["idx"]]
                        # print(session_state["reference"])
                        rerun()

                if button("Reselect Model & Strategy", type="secondary", width="stretch"):
                    for key in session4init:
                        session_state[key] = None
                    for key in session4pick:
                        session_state[key] = None
                    for key in session4pred:
                        session_state[key] = None
                    rerun()
            else:
                empty_messages.warning(
                    f"You selected a data for prediction. {session_state['timer4pick']} You can repick if needed."
                )

                with original:
                    markdown(f"**The data you selected**")
                    write(session_state["idx"])
                    write(session_state["cn4prove"][session_state["idx"]])
                    write(session_state["reference"])
                    write(session_state["src"])

                if session_state["hypothesis"] is None:
                    if button("Predict", type="primary", width="stretch"):
                        with Timer("Predict") as session_state["timer4pred"]:
                            session_state["model"].eval()
                            with no_grad():
                                out: Tensor = session_state["model"].generate(session_state["src"])
                                pred = [reversed_dict.get(idx, Tokens.UNK) for idx in out.squeeze().tolist()]
                                session_state["hypothesis"]: list[str] = [
                                    word.strip() for word in pred if word != Tokens.EOS
                                ]
                                rerun()


                else:
                    empty_messages.success(
                        f"Prediction Completed! {session_state["timer4pred"]} You can repredict or repick."
                    )

                    # Calculate BLEU Score
                    bleu = bleu_score(session_state["reference"], session_state["hypothesis"])
                    with prediction:
                        markdown(f"**The Prediction Result**")
                        write(session_state["hypothesis"])
                        write(bleu)
                    display4data_title.markdown(f"**The rating of the prediction**")
                    with display4data:
                        for domain, quality in map_bleu_to_benchmarks(bleu):
                            write(f"Domain: {domain}, Quality: {quality}")

                if button("Repick Data", type="secondary", width="stretch"):
                    for key in session4pick:
                        session_state[key] = None
                    for key in session4pred:
                        session_state[key] = None
                    rerun()
    else:
        empty_messages.error("The model & dictionary file does NOT exist.")
