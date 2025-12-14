#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 16:29
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   calc4cm.py
# @Desc     :   


from numpy import ndarray, array
from sklearn.metrics import confusion_matrix


def calculator_for_confusion_metrics(outputs: list[int], targets: list[int]) -> dict[str, float]:
    """ Calculate confusion metrics based on outputs and targets
    :param outputs: the predicted outputs
    :param targets: the ground truth outputs
    :return: a dict containing the confusion metrics
    """
    outputs: ndarray = array(outputs)
    targets: ndarray = array(targets)

    cm = confusion_matrix(targets, outputs)

    metrics: dict[str, float] = {}
    if cm.shape == (2, 2):
        # Binary classification
        TN, FP, FN, TP = cm.ravel()
        metrics.update({
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN)
        })
    else:
        # Multi-class classification
        num_classes = cm.shape[0]
        for i in range(num_classes):
            TP = int(cm[i, i])
            FP = int(cm[:, i].sum() - TP)
            FN = int(cm[i, :].sum() - TP)
            TN = int(cm.sum() - (TP + FP + FN))
            metrics.update({
                f"class_{i}_TP": TP,
                f"class_{i}_FP": FP,
                f"class_{i}_FN": FN,
                f"class_{i}_TN": TN
            })
    return metrics


if __name__ == "__main__":
    pass
