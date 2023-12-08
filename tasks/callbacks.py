import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import lightning as L
import matplotlib.pyplot as plt
from .models import BaseModel
from pathlib import Path

class PlotImageCallback(callbacks.Callback):

    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel, name: str) -> None:

        root = Path(trainer.logger.log_dir) / "images" / name

        if not Path(root).exists():
            Path(root).mkdir(parents=True, exist_ok=True)

        f, ax = plt.subplots(1, 4, figsize=(16, 4))

        for i in range(4):

            ax[i].imshow(pl_module.samples['images'][i, 0], cmap="gray")
            ax[i].set_title(f"y: {pl_module.samples['y'][i].item():.2f}\ny_hat: {pl_module.samples['y_hat'][i].argmax().item():.2f}")
            ax[i].axis("off")
            ax[i].set_aspect("equal")
            ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.savefig(root / f"epoch_{trainer.current_epoch}.png")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel) -> None:

        self.on_epoch_end(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel) -> None:

        self.on_epoch_end(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: BaseModel) -> None:

        self.on_epoch_end(trainer, pl_module, "test")

    

