#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   decorator.py
# @Desc     :   

from time import perf_counter
from functools import wraps

WIDTH: int = 64


def beautifier(func):
    """ The decorator for beautifying function output
    :param func: The function to be decorated
    :return: The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print("*" * WIDTH)
        print(f"The function named {func.__name__!r} is starting:")
        print("-" * WIDTH)
        result = func(*args, **kwargs)
        print("-" * WIDTH)
        print(f"The function named {func.__name__!r} has completed.")
        print("*" * WIDTH)
        print()
        return result

    return wrapper


def timer(func):
    """ The decorator for timing functions
    :param func: The function to be decorated
    :return: The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print("*" * WIDTH)
        print(f"The function named {func.__name__!r} is starting:")
        print("-" * WIDTH)
        time_start = perf_counter()
        result = func(*args, **kwargs)
        time_end = perf_counter()
        time_elapsed = time_end - time_start
        print("-" * WIDTH)
        print(f"The function named {func.__name__!r} took {time_elapsed:.4f} seconds to complete.")
        print("*" * WIDTH)
        print()

        return result

    return wrapper


if __name__ == "__main__":
    pass
