import lightning as L
import torch
from torch import nn, Tensor
import torchvision
from pathlib import Path
from typing import *
from torchmetrics import MetricCollection, Accuracy, Precision, AUROC, ConfusionMatrix
from ..models import BaseModel, Naive
import argparse

Batch 		= Tuple[Tensor, Dict[str, Tensor]]
StepOutput 	= Dict[str, Tensor]

class Classifier(BaseModel):

    def __init__(
        self,
        model: nn.Module,
        target: str = "Phase index",
        n_classes: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        
        super().__init__(model, target, lr, weight_decay)

        self.n_classes = n_classes
        self.loss = nn.CrossEntropyLoss()

        self.train_metrics  = self.build_metrics()
        self.val_metrics    = self.build_metrics()
        self.test_metrics   = self.build_metrics()

    def step(self, batch: Batch, batch_idx: int, name: str = "") -> StepOutput:
        """
        Performs one step of optimization on a batch of data. Common interface for training, validation and testing steps.
        
        Args:
            batch (Batch): A batch of data
            batch_idx (int): Index of the batch 
            name (str): Name of the step (train, val, test)
            
        Returns:
            StepOutput: A dictionary containing the loss and the predictions
        """

        x, y = batch[0], batch[1][self.target]

        y_hat_logit = self(x)
        y_hat = torch.softmax(y_hat_logit, dim=1)

        loss = self.loss(y_hat, y)

        outs = {
            "loss": loss.detach(),
            "y_hat": y_hat,
            "y": y,
        }

        return outs

    def build_metrics(self) -> MetricCollection:

        if self.n_classes > 2:
            task = "multiclass"
        else:
            task = "binary"

        metrics = MetricCollection({
            "accuracy": Accuracy(task=task, num_classes=self.n_classes),
            "precision": Precision(task=task, num_classes=self.n_classes, average="macro"),
            "auroc": AUROC(task=task, num_classes=self.n_classes),
            # "cm": ConfusionMatrix(task=task, num_classes=self.n_classes),
        })

        return metrics

def build_model(args: argparse.Namespace, n_classes: int = 4) -> nn.Module:
    
    # To ensure we have three initial channels, as prescribed by most models
    pre_conv = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride = 1, padding = 0),
            nn.ReLU(),
        )
    
    if "resnet" in args.backbone:
        backbone = torchvision.models.__dict__[args.backbone](pretrained=args.pretrained)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
    elif "naive" in args.backbone:
        backbone = Naive(n_classes=n_classes)
    else:
        raise NotImplementedError
    
    backbone = nn.Sequential(pre_conv, backbone)

    # try:
        # backbone = torch.compile(backbone, mode="reduce-overhead")
    # except:
        # pass

    model = Classifier(backbone, target=args.target, lr = args.lr, weight_decay = args.weight_decay)

    return model