#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/12 14:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   calc4seq_text_quilty.py
# @Desc     :   

from torch import Tensor, tensor, device
from torchmetrics.text import BLEUScore, ROUGEScore


class TextQualityScorer:
    """ Class for calculating text quality scores like BLEU and ROUGE """

    def __init__(self, accelerator: str = "cpu", idx2word_dictionary: dict = None) -> None:
        """ Initialise the TextQualityScorer class
        :param accelerator: device to use for calculations ("cpu", "cuda", "auto", etc.)
        """
        self._accelerator = device(accelerator)
        self._dictionary = idx2word_dictionary

        # Initialize metrics
        self._bleu = BLEUScore().to(self._accelerator)
        self._rouge = ROUGEScore().to(self._accelerator)
        """
        Expected score ranges for typical text generation tasks:
        BLEU：0.25-0.45 (0.0 - 0.6+)
        ROUGE：0.45-0.65 (0.0 - 1.0)
        """

    def __enter__(self):
        """ Context manager entry """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Context manager exit """
        pass

    def _convert_idx_tokens_to_text(self, idx_tokens: list[int]) -> str:
        """ Convert index tokens to text string
        :param idx_tokens: list of token indices
        :return: corresponding text string
        """
        if self._dictionary is None:
            return " ".join([str(token) for token in idx_tokens])

        words = []
        for token in idx_tokens:
            if token in self._dictionary:
                words.append(self._dictionary[token])
            else:
                words.append(f"<UNK_{token}>")

        return " ".join(words)

    def calculate(self, predictions: list[list[int]], references: list[list[int]]) -> dict[str, float]:
        """ Convert index tokens to text and calculate text quality scores
        :param predictions: list of predicted token indices
        :param references: list of reference token indices
        :return: a dict containing the calculated scores
        """
        if not predictions or not references:
            raise ValueError("predictions and references must not be empty")

        assert len(predictions) == len(references), "predictions and references must have the same length"

        # Convert index tokens to text strings
        preds: list[str] = [self._convert_idx_tokens_to_text(pred) for pred in predictions]
        refs: list[list[str]] = [[self._convert_idx_tokens_to_text(ref)] for ref in references]

        # Calculate scores, passing lists of strings
        bleu: float = self._bleu(preds, refs)
        rouge: dict = self._rouge(preds, refs)

        # Add extra stats info
        avg_pred_len: float = sum(len(pred) for pred in predictions) / len(predictions)

        dps: int = 4
        return {
            "bleu": round(float(bleu), dps),
            "rouge": round(float(rouge["rouge1_fmeasure"].item()), dps),
            "avg_pred_len": round(float(avg_pred_len), dps),
        }


if __name__ == "__main__":
    pass
