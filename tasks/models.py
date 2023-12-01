import lightning as L
import torch
from torch import nn, Tensor
from pathlib import Path
from typing import *
from torchmetrics import MetricCollection

Batch = Dict[str, Tensor]
StepOutput = Dict[str, Tensor]

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation = nn.ELU()):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = 0)
        self.activation = activation
        self.pooling = nn.AvgPool2d((2, 2))
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv1(x)
        # x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pooling(x)

        return x

class Naive(nn.Module):

    def __init__(self, n_classes: int = 4):
        super().__init__()
        
        self.n_classes = n_classes
        self.height = 256
        self.width = 256

        self.model = nn.Sequential(
            ConvBlock(3, 16, 3),
            *[ConvBlock(16*(2**i), 16*(2**(i+1)), 3) for i in range(5)],
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.n_classes),
        )

    def forward(self, x):
            
            return self.model(x)

class BaseModel(L.LightningModule):

    def __init__(self, 
                 model: nn.Module,
                 target: str,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4
                 ) -> None:
        
        super().__init__()

        self.model = model
        self.target = target
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_metrics  = None
        self.val_metrics    = None
        self.test_metrics   = None

    def forward(self, x: Tensor) -> Tensor:
        
        return self.model(x)
    
    def step(self, batch: Batch, batch_idx: int, name: str = "") -> StepOutput:
        raise NotImplementedError
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        outs = self.step(batch, batch_idx, name="train")

        ms = self.train_metrics(outs["y_hat"].detach(), outs["y"].detach())

        self.log_dict({f"train_{k}": v for k, v in ms.items()}, on_step=True, on_epoch=False)
             
        return outs["loss"]
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:

        outs = self.step(batch, batch_idx, name="val")

        # Detach the outputs to avoid memory leaks
        outs = {k: v.detach() for k, v in outs.items()}

        ms = self.val_metrics(outs["y_hat"], outs["y"])

        self.log_dict({f"val_{k}": v for k, v in ms.items()}, on_step=False, on_epoch=True)

        return outs
    
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        outs = self.step(batch, batch_idx, name="test")

        ms = self.test_metrics(outs["y_hat"], outs["y"])

        self.log_dict({f"test_{k}": v for k, v in ms.items()}, on_step=False, on_epoch=True)

        return outs
    
    def configure_optimizers(self) -> Any:
        
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
              },
        ]
        
        optimizer = torch.optim.SGD(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }
    
    def build_metrics(self) -> MetricCollection:
            
            raise NotImplementedError