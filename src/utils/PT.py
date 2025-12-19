#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:38
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   PT.py
# @Desc     :   

from datetime import datetime
from numpy import ndarray, array, random as np_random, number
from pandas import DataFrame, Series
from pathlib import Path
from random import seed as rnd_seed, getstate, setstate
from sklearn.utils.class_weight import compute_class_weight
from time import perf_counter
from torch import (cuda, backends, Tensor, tensor, float32, long,
                   manual_seed, get_rng_state, set_rng_state,
                   nn, unique, no_grad)
from typing import Literal

from torch.utils.tensorboard import SummaryWriter
from src.utils.decorator import timer

WIDTH: int = 64


class TorchRandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27, tick_tock: bool = False) -> None:
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        :param tick_tock: whether to print timing information
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_py_seed = None
        self._previous_pt_seed = None
        self._previous_np_seed = None

        self._tick: bool = tick_tock
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Set the random seed """
        if self._tick:
            self._start = perf_counter()

        # Save the previous random seed state
        self._previous_py_seed = getstate()
        self._previous_pt_seed = get_rng_state()
        self._previous_np_seed = np_random.get_state()

        # Set the new random seed
        rnd_seed(self._seed)
        manual_seed(self._seed)
        np_random.seed(self._seed)

        print("*" * WIDTH)
        print(f"{self._description} has been set to {self._seed}.")
        print("-" * WIDTH)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_py_seed is not None:
            setstate(self._previous_py_seed)
        if self._previous_pt_seed is not None:
            set_rng_state(self._previous_pt_seed)
        if self._previous_np_seed is not None:
            np_random.set_state(self._previous_np_seed)

        # Calculate elapsed time if measuring
        if self._tick:
            self._end = perf_counter()
            self._elapsed = self._end - self._start

        print("-" * WIDTH)
        print(f"{self._description} has been restored to previous randomness.")
        if self._tick:
            elapsed_time: str = self._format_time(self._elapsed)
            print(f"{self._description} took {elapsed_time}.")
        print("*" * WIDTH)
        print()

        # Return False to propagate exceptions, True to suppress them
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
        base: str = f"TorchRandomSeed({self._description}, seed={self._seed})"
        if self._tick and self._elapsed > 0.0:
            base += f", Elapsed Time: {self._elapsed:.2f} s"

        return base


@timer
def check_device() -> None:
    """ Check Available Device (CPU, GPU, MPS)
    :return: dictionary of available devices
    """

    # CUDA (NVIDIA GPU)
    if cuda.is_available():
        count: int = cuda.device_count()
        print(f"Detected {count} CUDA GPU(s):")
        for i in range(count):
            print(f"GPU {i}: {cuda.get_device_name(i)}")
            print(f"- Memory Usage:")
            print(f"- Allocated:   {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
            print(f"- Cached:      {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")

    # MPS (Apple Silicon GPU)
    elif backends.mps.is_available():
        print("Apple MPS device detected.")

    # Fallback: CPU
    else:
        print("Due to GPU or MPS unavailable, using CPU.")


@timer
def get_device(accelerator: str | Literal["auto", "cuda", "cpu"] = "auto", cuda_mode: int = 0) -> str:
    """ Get the appropriate device based on the target device string
    :param accelerator: the target device string ("auto", "cuda", "mps", "cpu")
    :param cuda_mode: the CUDA device index to use (if applicable)
    :return: the appropriate device string
    """
    match accelerator:
        case "auto":
            if cuda.is_available():
                count: int = cuda.device_count()
                print(f"Detected {count} CUDA GPU(s):")
                if cuda_mode < count:
                    for i in range(count):
                        print(f"GPU {i}: {cuda.get_device_name(i)}")
                        print(f"- Memory Usage:")
                        print(f"- Allocated:   {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
                        print(f"- Cached:      {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
                    print(f"The current accelerator is set to cuda:{cuda_mode}.")
                    return f"cuda:{cuda_mode}"
                else:
                    print(f"CUDA device index {cuda_mode} is out of range. Using 'cuda:0' instead.")
                    return "cuda:0"
            elif backends.mps.is_available():
                print("Apple MPS device detected.")
                return "mps"
            else:
                print("Due to GPU or MPS unavailable, using CPU ).")
                return "cpu"
        case "cuda":
            if cuda.is_available():
                count: int = cuda.device_count()
                print(f"Detected {count} CUDA GPU(s):")
                if cuda_mode < count:
                    for i in range(count):
                        print(f"GPU {i}: {cuda.get_device_name(i)}")
                        print(f"- Memory Usage:")
                        print(f"- Allocated:   {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
                        print(f"- Cached:      {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
                    print(f"The current accelerator is set to cuda:{cuda_mode}.")
                    return f"cuda:{cuda_mode}"
                else:
                    print(f"CUDA device index {cuda_mode} is out of range. Using 'cuda:0' instead.")
                    return "cuda:0"
            else:
                print("Due to GPU unavailable, using CPU.")
                return "cpu"
        case "mps":
            if backends.mps.is_available():
                print("Apple MPS device detected.")
                return "mps"
            else:
                print("Due to MPS unavailable, using CPU.")
                return "cpu"
        case "cpu":
            print("Using CPU as target device.")
            return "cpu"

        case _:
            print("Due to GPU unavailable, using CPU.")
            return "cpu"


def item2tensor(
        data: DataFrame | ndarray | list | int | float | number,
        embedding: bool = False,
        accelerator: str | Literal["cuda", "cpu"] = "cpu", is_grad: bool = False
) -> Tensor:
    """ Convert data to a PyTorch tensor
    :param data: data to be converted
    :param embedding: whether the data is embedded or not
    :param accelerator: the target device string ("cpu", "cuda", "mps")
    :param is_grad: whether the tensor requires gradient computation
    :return: the converted PyTorch tensor
    """
    # Convert DataFrame or list to ndarray
    if isinstance(data, DataFrame):
        arr = data.values
    elif isinstance(data, ndarray):
        arr = data
    elif isinstance(data, list):
        arr = array(data, dtype=float)
    elif isinstance(data, (int, float, number)):
        arr = array([data], dtype=float)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Convert to tensor with appropriate dtype
    if embedding:
        t = tensor(arr, dtype=long, device=accelerator, requires_grad=is_grad)
    else:
        t = tensor(arr, dtype=float32, device=accelerator, requires_grad=is_grad)

    # print(f"The tensor shape is {t.shape}, and its dtype is {t.dtype}.")

    return t


def sequences2tensors(sequences: list[list[int]], max_len: int) -> Tensor:
    """ Convert sequences to PyTorch Tensor
    :param sequences: list of word2id sequences
    :param max_len: maximum sequence length
    :return: PyTorch Tensor of sequences
    """
    padded: list[list[int]] = []

    for seq in sequences:
        if len(seq) > max_len:
            new = seq[:max_len]
        else:
            new = seq + [0] * (max_len - len(seq))
        padded.append(new)

    return tensor(padded, dtype=long)


@timer
def balance_imbalanced_weights(labels: list[int], ordered_categories: list[int]) -> ndarray:
    """ Balance the weights of different classes
    :param labels: array of labels
    :param ordered_categories: array of unique classes
    :return: array of balanced weights
    """
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=array(ordered_categories),
        y=Series(labels).to_numpy(),
    )

    print(f"Balanced weights for classes {ordered_categories}: {balanced_weights}")

    return balanced_weights


class TensorLogWriter:
    """ Tensor log writer """

    def __init__(self, log_dir: Path, log_name: str | None) -> None:
        if not log_dir.exists():
            raise FileNotFoundError(f"{log_dir} not found.")

        self._log_dir: Path = log_dir
        self._exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if log_name is None else log_name

    def __enter__(self):
        self._writer = SummaryWriter(f"{self._log_dir}/{self._exp_name}")
        return self._writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer:
            self._writer.close()


@timer
def verify_seq_net_initialisation(
        model: nn.Module, src: Tensor, tgt: Tensor,
        *,
        spare_model: nn.Module | None = None, spare_src: Tensor | None = None, spare_tgt: Tensor | None = None,
) -> None:
    if spare_model is None:
        print("*" * 64)
        print("Torch Network Initialization Verification")
        print("-" * 64)
        # Checking embedding initialization
        std: float | None = None
        for name, param in model.named_parameters():
            if any(key in name for key in ["embed", "embedder"]):
                print(f"Model - {name}: mean={param.mean():.6f}, std={param.std():.6f}, "
                      f"range=[{param.min():.6f}, {param.max():.6f}]")
                std = param.std().item()
        print("-" * 64)

        # Bias initialization check
        for name, param in model.named_parameters():
            if "bias" in name:
                unique_vals = unique(param.data)
                print(f"Model- {name}: shape={param.shape}, unique values={len(unique_vals)}", end=" ")
                if len(unique_vals) <= 3:
                    print(f"value={unique_vals.tolist()}")
        print("-" * 64)

        # Forward pass check
        model.eval()
        with no_grad():
            logits = model(src, tgt)
            print(f"logits shape: {logits.shape}, "
                  f"logits range=[{logits.min():.4f}, {logits.max():.4f}], "
                  f"logits mean±std=[{logits.mean():.4f} ± {logits.std():.4f}]")
        print("-" * 64)

        # Final verdict
        if abs(std - 0.01) > 0.005:
            print("x The embedding initialisation error！The std should be close to 0.01!")
        else:
            print("o The model's embedding initialisation is correct.")
        print("*" * 64)
    else:
        print("*" * 64)
        print("Network Initialization Verification Comparison")
        print("-" * 64)
        # Checking embedding initialization
        model_std, spare_std = float, float
        for name, param in model.named_parameters():
            if any(key in name for key in ["embed", "embedder"]):
                print(f"Main Model - {name}: mean={param.mean():.6f}, std={param.std():.6f}, "
                      f"range=[{param.min():.6f}, {param.max():.6f}]")
                model_std = param.std().item()
        for name, param in spare_model.named_parameters():
            if any(key in name for key in ["embed", "embedder"]):
                print(f"Spare Model - {name}: mean={param.mean():.6f}, std={param.std():.6f}, "
                      f"range=[{param.min():.6f}, {param.max():.6f}]")
                spare_std = param.std().item()
        print(f"Embedding std Comparison: {abs(model_std - spare_std):.6f}")
        print("-" * 64)

        # Bias initialization check
        for name, param in model.named_parameters():
            if "bias" in name:
                unique_vals = unique(param.data)
                print(f"Main Model - {name}: shape={param.shape}, unique values={len(unique_vals)}", end=" ")
                if len(unique_vals) <= 3:
                    print(f"value={unique_vals.tolist()}")
        for name, param in spare_model.named_parameters():
            if "bias" in name:
                unique_vals = unique(param.data)
                print(f"Spare Model - {name}: shape={param.shape}, unique values={len(unique_vals)}", end=" ")
                if len(unique_vals) <= 3:
                    print(f"value: {unique_vals.tolist()}")
        print("-" * 64)

        # Forward pass check
        model.eval()
        with no_grad():
            model_logits = model(src, tgt)
            print(f"Main Model: logits shape: {model_logits.shape}, "
                  f"logits range=[{model_logits.min():.4f}, {model_logits.max():.4f}], "
                  f"logits mean±std=[{model_logits.mean():.4f} ± {model_logits.std():.4f}]")
        spare_model.eval()
        with no_grad():
            spare_logits = spare_model(spare_src, spare_tgt)
            print(f"Spare Model: logits shape: {spare_logits.shape}, "
                  f"logits range=[{spare_logits.min():.4f}, {spare_logits.max():.4f}], "
                  f"logits mean±std=[{spare_logits.mean():.4f} ± {spare_logits.std():.4f}]")
        # Comparing logits differences
        print(f"logits diffs in Mean: {abs(model_logits.mean() - spare_logits.mean()):.6f}")
        print(f"logits diffs in std: {abs(model_logits.std() - spare_logits.std()):.6f}")
        print("-" * 64)

        # Final verdict
        if abs(model_std - 0.01) > 0.005:
            print("x The main model embedding initialisation error！The std should be close to 0.01!")
        else:
            print("o The main model's embedding initialisation is correct.")
        if abs(spare_std - 0.01) > 0.005:
            print("x The spare model's embedding initialisation error！The std should be close to 0.01!")
        else:
            print("o The spare model's embedding initialisation is correct.")
        print("*" * 64)


if __name__ == "__main__":
    pass
