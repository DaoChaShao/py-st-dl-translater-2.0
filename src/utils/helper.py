#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :   

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import seed as rnd_seed, getstate, setstate
from time import perf_counter
from typing import Literal, final
from yaml import safe_load

from src.utils.decorator import timer

WIDTH: int = 64


class Beautifier(object):
    """ beautifying code blocks using a context manager """

    def __init__(self, description: str = None):
        """ Initialise the Beautifier class
        :param description: the description of a beautifier
        """
        self._description: str = description

    def __enter__(self):
        """ Start the beautifier """
        print("*" * WIDTH)
        print(f"The block named {self._description!r} is starting:")
        print("-" * WIDTH)
        return self

    def __exit__(self, *args):
        """ Stop the beautifier """
        print("-" * WIDTH)
        print(f"The block named {self._description!r} has completed.")
        print("*" * WIDTH)
        print()


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("*" * WIDTH)
        print(f"{self._description} has started.")
        print("-" * WIDTH)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

        print("-" * WIDTH)
        print(f"{self._description} took {self._elapsed:.{self._precision}f} seconds.")
        print("*" * WIDTH)

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."

        return f"{self._description} has NOT started."


class RandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27, tick_tock: bool = False) -> None:
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        :param tick_tock: whether to time the random seed setting
        """
        self._description: str = description
        self._current_seed: int = seed
        self._previous_seed = None

        self._tick: bool = tick_tock
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Set the random seed """
        if self._tick:
            self._start = perf_counter()

        # Save the previous random seed state
        self._previous_seed = getstate()

        # Set the new random seed
        rnd_seed(self._current_seed)

        print("*" * WIDTH)
        print(f"{self._description} has been set to {self._current_seed}.")
        print("-" * WIDTH)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_seed is not None:
            setstate(self._previous_seed)

        # Calculate elapsed time if measuring
        if self._tick:
            self._end = perf_counter()
            self._elapsed = self._end - self._start

        print("-" * WIDTH)
        print(f"{self._description!r} has been restored to previous randomness.")
        if self._tick:
            elapsed_time: str = self._format_time(self._elapsed)
            print(f"{self._description} took {elapsed_time}.")
        print("*" * WIDTH)
        print()

        return False

    @staticmethod
    def _format_time(seconds: float) -> str:
        """ Format time breakdown from seconds to days, hours, minutes, and seconds
        :param seconds: time in seconds
        :return: formatted time breakdown string
        """
        days: int = int(seconds // 86400)
        hours: int = int((seconds % 86400) // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: float = seconds % 60

        parts: list[str] = []
        if days > 0:
            parts.append(f"{days} days")
        if hours > 0:
            parts.append(f"{hours} hours")
        if minutes > 0:
            parts.append(f"{minutes} minutes")
        if secs > 0 or not parts:
            parts.append(f"{secs:.2f} seconds")

        return " ".join(parts)

    def __repr__(self):
        """ Return a string representation of the random seed """
        base: str = f"PythonRandomSeed({self._description}, seed={self._current_seed})"
        if self._tick and self._elapsed > 0.0:
            base += f", Elapsed Time: {self._elapsed:.2f}s"

        return base


def read_file(file_path: str | Path) -> str:
    """ Read content from a file
    :param file_path: path to the file
    :return: content read from the file
    """
    with open(str(file_path), "r", encoding="utf-8") as file:
        content = file.read()
        # print(f"The content:\n{content}")

    return content


def read_files(file_paths: list[str | Path], workers: int = 10) -> list[str]:
    """ Read multiple files in parallel
    :param file_paths: list of file paths
    :param workers: number of parallel workers
    :return: list of contents read from the files
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        contents = list(executor.map(read_file, file_paths))

    return contents


def read_yaml(file_path: str | Path) -> dict:
    """ Read YAML file and return as dict
    :param file_path: path to the file
    :return: dict representation of the YAML file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return safe_load(f)


if __name__ == "__main__":
    pass
