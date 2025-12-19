#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   nlp.py
# @Desc     :   

from collections import Counter
from pathlib import Path
from re import compile, sub
from pandas import DataFrame
from spacy import load
from stqdm import stqdm
from tqdm import tqdm
from typing import Literal

from src.configs.cfg_base import CONFIG
from src.utils.decorator import timer


# @timer
def regular_chinese(words: list[str]) -> list[str]:
    """ Retain only Chinese characters in the list of words
    :param words: list of words to process
    :return: list of words containing only Chinese characters
    """
    pattern = compile(r"[\u4e00-\u9fa5]+")

    chinese = [word for word in words if pattern.fullmatch(word)]

    # print(f"Retained {len(chinese)} Chinese words from the original {len(words)} words.")

    return chinese


@timer
def regular_english(words: list[str]) -> list[str]:
    """ Retain only English characters in the list of words
    :param words: list of words to process
    :return: list of words containing only English characters
    """
    pattern = compile(r"^[A-Za-z]+$")

    english: list[str] = [word.lower() for word in words if pattern.fullmatch(word)]

    print(f"Retained {len(english)} English words from the original {len(words)} words.")

    return english


@timer
def count_frequency(words: list[str], top_k: int = 10, freq_threshold: int = 3) -> tuple[list, DataFrame]:
    """ Get frequency of Chinese words
    :param words: list of words to process
    :param top_k: number of top frequent words to return
    :param freq_threshold: frequency threshold to separate high and low frequency words
    :return: DataFrame containing words and their frequencies
    """
    # Get word frequency using Counter
    counter = Counter(words)
    words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    high_freq: list[str] = [word for word, count in words if count > freq_threshold]
    low_freq: list[str] = [word for word, count in words if count <= freq_threshold]

    cols: list[str] = ["word", "frequency"]
    sorted_freq = words[:top_k]
    df: DataFrame = DataFrame(sorted_freq, columns=cols)
    sorted_df = df.sort_values(by="frequency", ascending=False)

    print(f"Word Frequency Results:\n{sorted_df}")
    print(f"{len(low_freq)}, {len(low_freq) / len(counter):.2%} low frequency words has been filtered.")

    return high_freq, sorted_df


@timer
def unique_characters(content: str) -> list[str]:
    """ Get unique words from the list
    :param content: text content to process
    :return: list of unique words
    """
    chars: list[str] = list(content)
    # Get unique words by converting the list to a set and back to a sorted list
    # - sort based on Unicode code point order
    unique: list[str] = list(sorted(set(chars)))

    print(f"Extracted {len(unique)} unique words from the original {len(chars)} words.")

    return unique


@timer
def extract_cn_chars(filepath: str | Path, pattern: str = r"[^\u4e00-\u9fa5]") -> tuple[list, list]:
    """ Get Chinese characters from the text content
    :param filepath: path to the text file
    :param pattern: regex pattern to remove unwanted characters
    :return: list of Chinese characters
    """
    chars: list[str] = []
    lines: list[str] = []
    with open(str(filepath), "r", encoding="utf-8") as file:
        for line in file:
            line = sub(pattern, "", line).strip()
            if not line:
                continue
            lines.append(line)
            for word in list(line):
                chars.append(word)

    print(f"Total number of Chinese characters: {len(chars)}")
    print(f"Total number of lines in the Chinese content: {len(lines)}")

    return chars, lines


def select_bar(streamlit_bar: bool = False):
    """ Select progress bar based on environment
    :param streamlit_bar: whether running in Streamlit environment
    :return: appropriate progress bar function
    """
    if streamlit_bar:
        return stqdm
    else:
        return tqdm


class SpaCyBatchTokeniser:
    """ "" SpaCy NLP Processor for a batch of English texts or a single text """

    def __init__(self, lang: str | Literal["cn", "en"] = "en", batches: int = 100, strict: bool = False) -> None:
        """ Initialize the SpaCy Batch Tokeniser
        :param lang: language code for the texts (default is 'en' for English, 'cn' for Chinese)
        :param batches: number of texts to process in each batch
        :param strict: whether to enforce strict token filtering (default is False)
        """
        self._lang = lang
        self._batches = batches
        self._strict = strict
        self._nlp = None

    def __enter__(self):
        """ Enter the context manager """
        match self._lang:
            case "cn":
                self._nlp = load(CONFIG.FILEPATHS.SPACY_MODEL_CN)
                # print("SpaCy Chinese Model initialized.")
            case "en":
                self._nlp = load(CONFIG.FILEPATHS.SPACY_MODEL_EN)
                # print(f"SpaCy English Model initialized.")
            case _:
                raise ValueError(f"Unsupported language: {self._lang}")

        # print(f"{self._nlp.pipe_names} loaded.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Exit the context manager """
        if self._nlp:
            self._nlp = None

        # print(f"SpaCy {self._lang} Model released.")

    def batch_tokenise(self, contents: list[str], streamlit_bar: bool = False) -> list[list[str]]:
        """ Tokenise a batch of texts
        :param contents: list of text contents to process
        :param streamlit_bar: whether to use Streamlit progress bar
        :return: list of tokenized texts
        """
        if self._nlp is None:
            raise RuntimeError("Model not loaded. Use within 'with' statement.")

        words: list[list[str]] = []
        for doc in select_bar(streamlit_bar)(
                self._nlp.pipe(contents, batch_size=self._batches),
                total=len(contents),
                desc=f"SpaCy {self._lang} Tokeniser"
        ):
            tokens = self._process_doc(doc)

            words.append(tokens)

        # avg_len = sum(len(w) for w in words) / len(words)
        # print(f"Average length is {avg_len:.2f} words per content.")

        return words

    def single_tokenise(self, content: str) -> list[str]:
        """ Tokenise a single text
        :param content: a text content to process
        :return: list of tokens
        """
        if self._nlp is None:
            raise RuntimeError("Model not loaded. Use within 'with' statement.")

        doc = self._nlp(content)
        tokens = self._process_doc(doc)

        # print(f"The {len(tokens)} words has been segmented using SpaCy Tokeniser.")

        return tokens

    def _process_doc(self, doc):
        """ Process a SpaCy Doc object and extract tokens based on language and strictness
        :param doc: SpaCy Doc object
        :return: list of processed tokens
        """
        match self._lang:
            case "cn":
                if self._strict:
                    return [
                        token.text for token in doc
                        if token.text.strip()
                           and not token.is_stop
                           and not token.is_punct
                           and any(c.isalnum() for c in token.text)
                    ]
                else:
                    return [
                        token.text for token in doc
                        if token.text.strip()
                    ]
            case "en":
                if self._strict:
                    return [
                        token.lemma_.lower() for token in doc
                        if not token.is_stop
                           and token.text.strip()
                           and any(c.isalnum() for c in token.text)
                    ]
                else:
                    return [
                        token.lemma_.lower() for token in doc
                        if token.text.strip()
                    ]
            case _:
                raise ValueError(f"Unsupported language: {self._lang}")


def build_word2id_seqs(
        contents: list[list[str]], dictionary: dict[str, int], add_sos_eos: bool = False,
        UNK: str = "<UNK>", SOS: str = "<SOS>", EOS: str = "<EOS>"
) -> list[list[int]]:
    """ Build word2id sequences from contents using the provided dictionary
    :param contents: list of texts to convert
    :param dictionary: word2id mapping dictionary
    :param add_sos_eos: whether to add start-of-sequence and end-of-sequence tokens
    :param UNK: token for unknown words
    :param SOS: token for start of sequence
    :param EOS: token for end of sequence
    :return: list of word2id sequences
    """
    sequences: list[list[int]] = []
    for content in contents:
        sequence: list[int] = [dictionary[SOS]] if add_sos_eos else []
        for word in content:
            if word in dictionary:
                sequence.append(dictionary[word])
            else:
                sequence.append(dictionary[UNK])
        if add_sos_eos:
            sequence.append(dictionary[EOS])
        sequences.append(sequence)

    return sequences


@timer
def check_vocab_coverage(words: list[str], dictionary: dict[str, int]) -> float:
    """ Check the vocab coverage
    :param words: list of words
    :param dictionary: word2id mapping dictionary
    :return: vocab coverage
    """
    counter: int = sum(1 for word in words if word in dictionary)
    coverage: float = counter / len(words)

    if coverage >= 0.95:
        rating = "Perfect"
    elif coverage >= 0.90:
        rating = "Good"
    elif coverage >= 0.85:
        rating = "Enough"
    else:
        rating = "Bad"

    print(f"The coverage of vocabs in the sentences is {coverage:.2%}, and {rating}.")

    return coverage


if __name__ == "__main__":
    text_en = "Don't you miss anything?"
    text_cn = "我爱北京天安门。"

    with SpaCyBatchTokeniser(lang="en", strict=False) as tokenizer:
        tokens = tokenizer.single_tokenise(text_en)
        print("Tokens:", tokens)

    with SpaCyBatchTokeniser(lang="en", strict=True) as tokenizer:
        tokens = tokenizer.single_tokenise(text_en)
        print("Tokens with strict filtering:", tokens)

    with SpaCyBatchTokeniser(lang="cn", strict=False) as tokenizer:
        tokens = tokenizer.single_tokenise(text_cn)
        print("Tokens:", tokens)

    with SpaCyBatchTokeniser(lang="cn", strict=True) as tokenizer:
        tokens = tokenizer.single_tokenise(text_cn)
        print("Tokens with strict filtering:", tokens)
