#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 16:06
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   calc4classification.py
# @Desc     :   

from numpy import ndarray, array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculator_for_classification(outputs: list[int], targets: list[int]) -> dict[str, float]:
    """ calculate accuracy, precision, recall, f1 score based on outputs and targets
    :param outputs: the predicted outputs
    :param targets: the ground truth outputs
    :return: dict of outputs and targets
    """
    outputs: ndarray = array(outputs)
    targets: ndarray = array(targets)

    dps: int = 4

    return {
        "accuracy": round(float(accuracy_score(targets, outputs)), dps),
        "precision": round(float(precision_score(targets, outputs, average="weighted", zero_division=0)), dps),
        "recall": round(float(recall_score(targets, outputs, average="weighted", zero_division=0)), dps),
        "f1_score": round(float(f1_score(targets, outputs, average="weighted", zero_division=0)), dps),
    }


if __name__ == "__main__":
    pass
