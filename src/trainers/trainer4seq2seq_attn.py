#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/21 21:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer4seq2seq_attn.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from datetime import datetime
from json import dumps
from torch import nn, optim, no_grad, save, device, tensor, exp, Tensor
from typing import Literal

from src.configs.cfg_types import SeqStrategies
from src.dataloaders import TorchDataLoader
from src.trainers.calc4seq_text_quilty import TextQualityScorer
from src.utils.logger import record_log
from src.utils.PT import get_device

WIDTH: int = 64


class TorchTrainer4SeqToSeq(QObject):
    """ Trainer class for managing seq2seq training process """
    losses: Signal = Signal(int, float, float)

    def __init__(self,
                 vocab_size_tgt: int,
                 model: nn.Module, optimiser, criterion, scheduler=None,
                 PAD: int = 0, SOS: int = 2, EOS: int = 3,
                 *,
                 decode_strategy: str | SeqStrategies | Literal["greedy", "beam"] = "greedy", beam_width: int = 5,
                 accelerator: str | Literal["auto", "cuda", "cpu"] = "auto",
                 ) -> None:
        """ Initialise the TorchTrainer4SeqToSeq class
        :param vocab_size_tgt: size of the targert vocabulary
        :param model: the seq2seq model to be trained
        :param optimiser: the optimiser for training
        :param criterion: the loss function
        :param scheduler: learning rate scheduler (optional)
        :param PAD: padding token index
        :param SOS: start-of-sequence token index
        :param EOS: end-of-sequence token index
        :param decode_strategy: decoding strategy for generation ("greedy", "beam_search", etc.)
        :param beam_width: beam width for beam search decoding
        :param accelerator: device to use for training ("cpu", "cuda", "auto", etc.)
        """
        super().__init__()
        self._vocab_size = vocab_size_tgt
        self._PAD = PAD
        self._SOS = SOS
        self._EOS = EOS
        self._S = decode_strategy
        self._beam_size = beam_width

        self._accelerator = get_device(accelerator)
        self._model = model.to(device(self._accelerator))
        self._optimiser = optimiser
        self._criterion = criterion
        self._scheduler = scheduler

    def _epoch_train(self, dataloader: TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _total: float = 0.0
        for src, tgt in dataloader:
            src, tgt = src.to(device(self._accelerator)), tgt.to(device(self._accelerator))

            self._optimiser.zero_grad()

            # Teacher Forcing
            logits = self._model(src, tgt)
            # print(logits.shape)

            # Calculate loss
            targets = tgt[:, 1:]
            loss = self._criterion(
                logits.reshape(-1, self._vocab_size),  # [batch*(tgt_len-1), vocab]
                targets.reshape(-1)  # [batch*(tgt_len-1)]
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimiser.step()

            _loss += loss.item() * src.size(0)
            _total += src.size(0)

        return _loss / _total

    def _epoch_valid(self, dataloader: TorchDataLoader) -> tuple[float, dict[str, float]]:
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _total: float = 0.0

        _predictions: list[list[int]] = []
        _references: list[list[int]] = []
        with no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(device(self._accelerator)), tgt.to(device(self._accelerator))

                logits = self._model(src, tgt)
                # print(logits.shape)

                # Calculate loss as same as training
                targets = tgt[:, 1:]
                loss = self._criterion(
                    logits.reshape(-1, self._vocab_size),  # [batch*(tgt_len-1), vocab]
                    targets.reshape(-1)  # [batch*(tgt_len-1)]
                )

                _loss += loss.item() * src.size(0)
                _total += src.size(0)

                # Collect predictions and references for evaluation
                src_lengths: Tensor = (src != self._PAD).sum(dim=1)
                dynamic_lens = (src_lengths.float() * 1.5).long().clamp(min=10, max=100)
                MAX_LEN = dynamic_lens.max().item()
                generated = self._model.generate(src, max_len=MAX_LEN, strategy=self._S, beam_width=self._beam_size)
                for i in range(len(generated)):
                    _pred = generated[i].cpu().tolist()
                    _ref = tgt[i, 1:].cpu().tolist()

                    # Truncate at EOS token
                    if self._EOS in _pred:
                        _pred = _pred[:_pred.index(self._EOS)]
                    if self._EOS in _ref:
                        _ref = _ref[:_ref.index(self._EOS)]

                    _predictions.append(_pred)
                    _references.append(_ref)

        with TextQualityScorer(self._accelerator) as scorer:
            _metrics: dict[str, float] = {
                **scorer.calculate(_predictions, _references),
            }

        avg_val_loss = _loss / _total if _total > 0 else float("inf")
        dps: int = 4
        _metrics["perplexity"] = round(float(exp(tensor(avg_val_loss)).item()), dps)
        _metrics["avg_val_loss"] = round(float(avg_val_loss), dps)

        return _loss / _total, _metrics

    def fit(self,
            train_loader: TorchDataLoader, valid_loader: TorchDataLoader,
            epochs: int, model_save_path: str | None = None, log_name: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :param log_name: name for the training log file
        :return: None
        """
        # Initialize logger
        timer = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        logger = record_log(f"train_at_{timer}-{log_name}")

        _best_valid_loss = float("inf")
        _min_delta = 5e-4
        _patience = 5
        _patience_counter = 0
        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, _metrics = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss)

            # Log epoch results
            dps: int = 4
            logger.info(dumps({
                "epochs": epochs,
                "epoch": epoch + 1,
                "strategy": self._S,
                "alpha": self._optimiser.param_groups[0]["lr"],
                "train_loss": round(float(train_loss), dps),
                "valid_loss": round(float(valid_loss), dps),
                **_metrics,
            }))

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss - _min_delta:
                _patience_counter = 0
                _best_valid_loss = valid_loss

                if model_save_path:
                    save(self._model.state_dict(), model_save_path)
                    print(f"√ Model's parameters saved to {model_save_path}\n")
            else:
                _patience_counter += 1
                print(f"× No improvement [{_patience_counter}/{_patience}]\n")
                if _patience_counter >= _patience:
                    print("*" * WIDTH)
                    print("Early Stopping Triggered")
                    print("-" * WIDTH)
                    print(f"Early stopping at epoch {epoch}, the best value is {_best_valid_loss:.4f}.")
                    print("*" * WIDTH)
                    print()
                    break

            if self._scheduler is not None:
                if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(valid_loss)
                else:
                    self._scheduler.step()

        if _patience_counter < _patience:
            print(f"Training completed after {epochs} epochs.")


if __name__ == "__main__":
    pass
