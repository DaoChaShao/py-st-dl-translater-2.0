#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 17:25
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   logger.py
# @Desc     :   

from datetime import datetime
from json import dumps
from logging import (Logger, getLogger, INFO, WARNING, ERROR, CRITICAL, DEBUG,
                     FileHandler, StreamHandler,
                     Formatter)
from pathlib import Path
from random import randint, uniform

from src.configs.cfg_base import CONFIG


def record_log(name: str, log_dir: str = str(CONFIG.FILEPATHS.LOGS), mode: str = "w", level: str = "info") -> Logger:
    """ Create a logger that records logs to a txt file
    :param name: name of the log file (without extension)
    :param log_dir: directory to save the log file
    :param mode: file mode, "w" for write (overwrite), "a" for append
    :param level: logging level, e.g., "info", "debug"
    :return: configured Logger object
    """
    LOG_DIR: Path = Path(log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # print(LOG_DIR)

    # Set the level
    level_map = {
        "info": INFO,
        "warning": WARNING,
        "error": ERROR,
        "critical": CRITICAL,
        "debug": DEBUG
    }
    lvl = level_map.get(level.lower(), INFO)

    # Initialize logger
    logger: Logger = getLogger(name)
    # Set log level for global
    logger.setLevel(lvl)
    logger.propagate = False

    # Set a file handler separately (w: write, a: append)
    handler4file = FileHandler(LOG_DIR / f"{name}.log", mode=mode, encoding="utf-8")
    handler4file.setLevel(lvl)
    # print(handler4file)

    # Set Console handler separately
    handler4console = StreamHandler()
    handler4console.setLevel(INFO)
    # print(handler4console)

    # Formatter
    formatter = Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler4file.setFormatter(formatter)
    handler4console.setFormatter(formatter)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(handler4file)
        logger.addHandler(handler4console)

    return logger


if __name__ == "__main__":
    # Get the current date and time
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Initialize logger
    logger: Logger = record_log(f"train@{dt}")

    # Log some test messages
    epochs: int = randint(0, 10)
    for epoch in range(epochs):
        alpha: float = uniform(0, 0.01)
        train_loss: float = uniform(0.0, 1.0)
        valid_loss: float = uniform(0.0, 1.0)
        logger.info(dumps({
            "epoch": epoch + 1,
            "alpha": alpha,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }))
