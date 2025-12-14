#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/17 23:06
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   sem_seg.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from datetime import datetime
from json import dumps
from torch import device, nn, no_grad, Tensor, cat, save
from torch.utils.data import DataLoader

from src.utils.logger import record_log
from src.utils.PT import get_device

WIDTH: int = 64


class TorchTrainer4UNetSemSeg(QObject):
    """ Trainer class for managing training process """
    losses: Signal = Signal(int, float, float)

    def __init__(self, model: nn.Module, optimiser, criterion, scheduler=None, accelerator: str = "auto") -> None:
        super().__init__()
        self._accelerator = get_device(accelerator)
        self._model = model.to(device(self._accelerator))
        self._optimiser = optimiser
        self._criterion = criterion
        self._scheduler = scheduler

    def _epoch_train(self, dataloader: DataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _tatal: float = 0.0
        for features, labels in dataloader:
            features = features.to(device(self._accelerator))
            labels = labels.to(device(self._accelerator))

            # Ensure the shape dimension of labels is right [B, 1, H, W]
            if len(labels.shape) == 3:  # [B, H, W]
                labels = labels.unsqueeze(1)  # -> [B, 1, H, W]

            self._optimiser.zero_grad()
            outputs = self._model(features)  # [B, 1, H, W]
            # print(outputs.shape, masks.shape)
            loss = self._criterion(outputs, labels)

            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimiser.step()

            _loss += loss.item()
            _tatal += 1.0

        return _loss / _tatal

    def _epoch_valid(self, dataloader):
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _total: int = 0

        _results: list[Tensor] = []
        _targets: list[Tensor] = []
        _metrics: dict[str, float] = {}
        with no_grad():
            for features, labels in dataloader:
                features = features.to(device(self._accelerator))
                labels = labels.to(device(self._accelerator))

                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                outputs = self._model(features)
                loss = self._criterion(outputs, labels)

                _loss += loss.item() * features.size(0)
                _total += features.size(0)

                _results.append(outputs)
                _targets.append(labels)

            epoch_outputs = cat(_results, dim=0)
            epoch_targets = cat(_targets, dim=0)

        return _loss / _total, _metrics

    def fit(self,
            train_loader: DataLoader, valid_loader: DataLoader,
            epochs: int, model_save_path: str | None = None, log_name: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :return: None
        """
        # Initialize logger
        timer = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger = record_log(f"train_at_{timer}-{log_name}")

        _best_mIoU: float = 0.0
        _patience: int = 5
        _patience_counter: int = 0

        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, _metrics = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss)

            # Log epoch results
            logger.info(dumps({
                "epochs": epochs,
                "epoch": epoch + 1,
                "alpha": self._optimiser.param_groups[0]["lr"],
                "train_loss": round(float(train_loss), 4),
                "valid_loss": round(float(valid_loss), 4),
                **_metrics,
            }))

            # Save the model if it has the best validation loss so far
            if _metrics:
                if _metrics["iou"] > _best_mIoU + 1e-4:
                    _best_mIoU = _metrics["iou"]
                    _patience_counter = 0
                    if model_save_path is not None:
                        save(self._model.state_dict(), model_save_path)
                        print(f"√ Model's parameters saved to {model_save_path}\n")
                else:
                    _patience_counter += 1
                    print(f"× No improvement [{_patience_counter}/{_patience}]\n")
                    if _patience_counter >= _patience:
                        print("*" * WIDTH)
                        print("Early Stopping Triggered")
                        print("-" * WIDTH)
                        print(f"Early stopping at epoch {epoch}, best mIoU: {_best_mIoU * 100:.2f}%")
                        print("*" * WIDTH)
                        print()
                        break

            if self._scheduler is not None:
                self._scheduler.step(_metrics["iou"])


if __name__ == "__main__":
    pass
