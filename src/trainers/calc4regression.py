#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 16:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   calc4regression.py
# @Desc     :   

from numpy import ndarray, sqrt, array
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculator_for_regression(outputs: list[int], targets: list[int]) -> dict[str, float]:
    """ Calculates the mean squared error and mean absolute error
    :param outputs: the predicted outputs
    :param targets: the ground truth outputs
    :return: a dict containing the mean squared error and mean absolute error
    """
    outputs: ndarray = array(outputs)
    targets: ndarray = array(targets)

    dps: int = 4

    rMse: float = sqrt(((outputs - targets) ** 2).mean())
    mse: float = mean_squared_error(outputs, targets)
    mae: float = mean_absolute_error(outputs, targets)
    r2: float = r2_score(targets, outputs)

    return {
        "rMse": round(rMse, dps),
        "mse": round(mse, dps),
        "mae": round(mae, dps),
        "r2": round(r2, dps),
    }


if __name__ == "__main__":
    pass
