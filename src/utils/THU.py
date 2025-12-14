#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   THU.py
# @Desc     :   

from thulac import thulac


class THULACTokeniser:

    def __init__(self, with_pos: bool = False):
        self._with_pos = with_pos
        self._thu = None

    def __enter__(self):
        match self._with_pos:
            case True:
                self._thu = thulac(seg_only=False)
                # print("THUCLAC model loaded. The text will be cut with POS tags.")
            case False:
                self._thu = thulac(seg_only=True)
                # print(f"THUCLAC model loaded. The text will be cut ONLY.")
            case _:
                raise ValueError("Invalid value for with_pos. Must be True or False.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._thu:
            self._thu = None

            # print("THUCLAC model released.")

    def tokenise(self, text: str) -> list[str] | list[tuple[str, str]]:
        if not self._thu:
            raise RuntimeError("THUCLAC model is not loaded. Please use 'with' statement to load the model.")

        match self._with_pos:
            case True:
                words_tag: list[tuple[str, str]] = self._thu.cut(text)
                return words_tag
            case False:
                words: list[str] = self._thu.cut(text)
                words = [word for word, _ in words]
                return words
            case _:
                raise ValueError("Invalid value for with_pos. Must be True or False.")


if __name__ == "__main__":
    text = "我爱北京天安门。"

    with THULACTokeniser(with_pos=False) as thu:
        tokens = thu.tokenise(text)
        print("Tokens:", tokens)
