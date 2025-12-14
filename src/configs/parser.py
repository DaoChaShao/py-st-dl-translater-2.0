#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 18:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   parser.py
# @Desc     :   

from argparse import ArgumentParser


def set_argument_parser():
    """ Set command line arguments """
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training times")
    args = parser.parse_args()
    """
    Display hyperparameters You need to set before training using command lines:
    1.Running: python trainer.py -h
    2.Or running: python trainer.py --help
    """

    return args


if __name__ == "__main__":
    set_argument_parser()
