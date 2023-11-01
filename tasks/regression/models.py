import lightning as L
import torch
from torch import nn, Tensor
import torchvision
from pathlib import Path
from typing import *
from torchmetrics import MetricCollection, MeanAbsolutePercentageError, PearsonCorrCoef
from .models import BaseModel

Batch       = Dict[str, Tensor]
StepOutput  = Dict[str, Tensor]


class Regressor(BaseModel):
    def __init__(
        self, model: nn.Module, target: str = "Thickness", lr: float = 1e-4, weight_decay: float = 1e-4
    ) -> None:
        super().__init__(model, lr, weight_decay)

        self.loss = nn.MSELoss()

    def step(self, batch: Batch, batch_idx: int, name: str = "") -> StepOutput:
        x, y = batch["x"], batch["y"]

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        outs = {
            "loss": loss,
            "y_hat": y_hat,
            "y": y,
        }

        return outs

    def build_metrics(self) -> MetricCollection:
        metrics = MetricCollection(
            [
                MeanAbsolutePercentageError(),
                PearsonCorrCoef(),
            ]
        )

        return metrics
