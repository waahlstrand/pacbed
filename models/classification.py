from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
import pytorch_lightning.loggers as pl_loggers
import torchvision.models as models
from skimage.draw import disk
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import *
import utils

class RadialNormalization(nn.Module):

    def __init__(self, n_pixels_original: int, n_pixels_target: int) -> None:
        super().__init__()

        self.n_pixels_original = n_pixels_original
        self.n_pixels_target = n_pixels_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Should be radial matrix
        # TODO
        R2 = self.n_pixels_target // 2
    
        a = torch.pow(10, -7 + 2 * torch.rand(1, device=x.device))
        b = (5 + 95 * torch.rand(1, device=x.device)) * (self.n_pixels_target / self.n_pixels_original)

        p = torch.rand(1)

        if p > 0.5:

            x = x + a * torch.exp(R2/b**2)
        else:
            x = x + a * (1-torch.exp(-R2/b**2))


        return x


class Normalize(nn.Module):

    def __init__(self, scaling="linear"):

        super().__init__()

        self.scaling = scaling

    def forward(self, x: torch.Tensor):

        # Randomly uniform scaling between 0.15 and 0.35
        scale = torch.rand(1, device=x.device) * 0.2 + 0.15
        x = scale * x / torch.mean(torch.abs(x))

        if self.scaling == "linear":
            # Absolute value of intensity.
            x = torch.abs(x)
        else:
            # Square root of absolute value of intensity.
            x = torch.sqrt(torch.abs(x))

        return x

class AddNoise(nn.Module):

    def __init__(self, eta):

        super().__init__()

        self.eta = eta

    def forward(self, x: torch.Tensor):

        return self.add_noise(x, self.eta)

    def add_noise(self, x: torch.Tensor, eta: float):

        device = x.device

        # Noise model parameters.
        c1_log_intensity    = torch.tensor(-7.60540384562294, device = device)
        c0_log_intensity    = torch.tensor(28.0621318839493, device = device)
        c2_log_s_signal     = torch.tensor(0.0349329915028267, device = device)
        c1_log_s_signal     = torch.tensor(-0.304984702105353, device = device)
        c0_log_s_signal     = torch.tensor(6.86126419242947, device = device)
        c1_log_s_background = torch.tensor(-6.99617784594964, device = device)
        c0_log_s_background = torch.tensor(12.448421647627, device = device)

        # Rescale intensity. It is assumed that before this operation, sum(x) = 1.
        x = x + 1e-6
        x = torch.exp(c1_log_intensity * eta + c0_log_intensity) * x

        # Add lognormal noise.
        log_x           = torch.log(x)
        cv_signal       = torch.exp(c2_log_s_signal * log_x * log_x + c1_log_s_signal * log_x + c0_log_s_signal) / x
        log_ratio_s2_m2 = torch.log(1.0 + cv_signal * cv_signal)
        mu_signal       = log_x - 0.5 * log_ratio_s2_m2
        sigma_signal    = torch.sqrt(log_ratio_s2_m2)
        x               = torch.exp(torch.normal(mu_signal, sigma_signal))

        # Add normal noise.
        m_background    = torch.zeros(x.shape, device = device)
        s_background    = torch.exp(c1_log_s_background * eta + c0_log_s_background) * torch.ones(x.shape, device = device)
        x               = x + torch.normal(m_background, s_background)

        # print(x-input)

        return x


class AnnulusOcclusion(nn.Module):

    def __init__(self, n_pixels: int, invert: bool = False):

        super().__init__()

        self.invert = invert
        self.n_pixels = n_pixels
        self.mask   = torch.zeros((n_pixels, n_pixels), dtype=torch.uint8)

    def occlude(self, x: torch.Tensor, invert: bool, max_radius_ratio: float = 0.6, min_radius_ratio: float = 0.1):

        random_offset = torch.randint(-2, 0, (2,)).numpy() # Handle center not being... centered.
        disk_center = (self.n_pixels // 2 + random_offset[0], self.n_pixels // 2 + random_offset[1])

        max_radius = int((self.n_pixels // 2 - 1) * max_radius_ratio)
        min_radius = int((self.n_pixels // 2 - 1) * min_radius_ratio)

        large_radius = torch.randint(min_radius, max_radius, (1,)).item()
        small_radius = torch.randint(0, large_radius, (1,)).item()
        
        large_rr, large_cc = disk(disk_center, large_radius)
        small_rr, small_cc = disk(disk_center, small_radius)

        mask = self.mask

        mask[large_rr, large_cc] = 1
        mask[small_rr, small_cc] = 0

        if invert:
            y = (1-mask)*x
        else:
            y = mask*x

        return y.view(x.shape)

    def forward(self, x):

        return self.occlude(x, self.invert)
    

class CenterCropWithRandomOffset(nn.Module):

    def __init__(self, n_pixels: int, crop: int, offset: float = 0.3):

        super().__init__()

        self.crop = crop
        self.offset = offset
        self.n_pixels = n_pixels

    def forward(self, x: torch.Tensor):

        # Randomly select a crop offset.
        random_offset = torch.randint(-int(self.n_pixels * self.offset), int(self.n_pixels * self.offset), (2,))
        crop_center = (self.crop + random_offset[0].item(), self.crop + random_offset[1].item())
        cc = transforms.CenterCrop(crop_center)

        # Crop.
        y = cc(x)

        return y


class Augmenter(nn.Module):

    def __init__(self, 
                 n_pixels_original: int, 
                 n_pixels_target: int,
                 crop: int, 
                 eta: float, 
                 scaling="linear", 
                 translate: Union[float, Tuple[float, float]] = (0.05, 0.05),
                 scale: Union[float, Tuple[float, float]] = (0.95, 1.05),
                 offset: float = 0.05,
                 p_occlusion: float = 0.4):

        super().__init__()

        self.n_pixels_original = n_pixels_original
        self.n_pixels_target = n_pixels_target
        self.crop = crop
        self.eta = eta
        self.scaling = scaling
        self.translate = translate
        self.scale = scale
        self.offset = offset

        self.augment = nn.Sequential(
            # RadialNormalization(n_pixels_original, n_pixels_target),
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees = 0, translate = translate, scale = scale, shear = 0),
            CenterCropWithRandomOffset(n_pixels_original, crop, offset = offset),
            transforms.Resize((n_pixels_target, n_pixels_target)),
            AddNoise(eta),
            Normalize(scaling),
            transforms.RandomApply([
            transforms.RandomChoice([
                AnnulusOcclusion(n_pixels_target, invert = False),
                AnnulusOcclusion(n_pixels_target, invert = True)
            ], p = [0.5, 0.5]),
            ], p = p_occlusion)
   
        )

    def forward(self, x):

        return self.augment(x)


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

class CustomPACBEDBackbone(nn.Module):

    def __init__(self, ):

        super().__init__()

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
            nn.Linear(256, 1),
        )

    def forward(self, x):
            
            return self.model(x)

class PhaseClassifier(pl.LightningModule):

    def __init__(self, 
                 backbone: nn.Module, 
                 loss: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 n_pixels: int, 
                 optimizer_params: Dict[str, Any] = {},
                 training_params: Dict[str, Any] = {},
                 *args: Any, 
                 **kwargs: Any,):

        super().__init__()

        self.n_pixels   = n_pixels
        
        self.optimizer_params = optimizer_params
        self.training_params = training_params
        self.kwargs     = kwargs
        self.args       = args

        self.save_hyperparameters(
            "optimizer_params",
            "training_params",
        )

        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride = 1, padding = 0),
            nn.ReLU(),
        )
    
        self.model = backbone
        self.loss = loss
        self.optimizer = optimizer
        
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_step_outputs: List[Dict[str, torch.Tensor]] = []

    def forward(self, x):

        return self.model(self.pre_conv(x))
    
    def configure_optimizers(self):

        optimizer = self.optimizer(self.parameters(), **self.hparams.optimizer_params)

        return optimizer
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):

        x, y = batch
        y_hat = self(x)
        # print(y_hat.shape, y)
        loss = self.loss(y_hat, y.squeeze().long())

        self.log("train_loss", loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):

        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.squeeze().long())

        self.validation_step_outputs.append({"loss": loss, "x": x, "y_true": y, "y_pred": y_hat})

        return {"loss": loss, "x": x, "y_true": y, "y_pred": y_hat}
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
    
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.squeeze().long())

        self.test_step_outputs.append({"loss": loss, "x": x, "y_true": y, "y_pred": y_hat})

        return {"loss": loss, "x": x, "y_true": y, "y_pred": y_hat}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def on_validation_epoch_end(self) -> None:
        
        loss = torch.stack([l['loss'] for l in self.validation_step_outputs]).mean()

        self.log("val_loss", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        x = torch.cat([_['x'] for _ in self.validation_step_outputs])
        y_true = torch.cat([_['y_true'] for _ in self.validation_step_outputs])
        y_pred = torch.cat([_['y_pred'] for _ in self.validation_step_outputs])

        # if self.current_epoch % 50 == 0:
        #     try:
        #         self.log_validation_scatter(y_true, y_pred, name="val")
        #     except Exception as e:
        #         print("\n", e, "\n")
        #         pass

        # Sample 4 random indices
        # and select corresponding images with true and predicted labels
        idx = np.random.choice(x.shape[0], 4, replace = False)
        x_sample = x[idx]
        y_true_sample = y_true[idx]
        y_pred_sample = y_pred[idx]

        # if self.current_epoch % 50 == 0:
            # self.log_image_w_predictions(x_sample, y_true_sample, y_pred_sample)

        self.validation_step_outputs = []
    
    def on_test_epoch_end(self) -> None:
        
        #x = torch.cat([_['x'] for _ in outputs])
        
        y_true = torch.cat([_['y_true'].view(self.kwargs['batch_size'], 1) for _ in self.test_step_outputs])
        y_pred = torch.cat([_['y_pred'] for _ in self.test_step_outputs])

        # try:
        #     self.log_validation_scatter(y_true, y_pred, name="test")
        # except Exception as e:
        #     print("\n", e, "\n")

        #     pass


    def log_validation_scatter(self, y_true: torch.Tensor, y_pred: torch.Tensor, name: str = "") -> None:

        fig, ax, ax_histx, ax_histy = utils.scatter_hist(
            y_true.cpu().detach().exp().numpy(), 
            y_pred.cpu().detach().exp().numpy()
        )

        if not os.path.exists(f"{self.logger.log_dir}/images/scatter"):
            os.makedirs(f"{self.logger.log_dir}/images/scatter")

        fig.savefig(f"{self.logger.log_dir}/images/scatter/{name}_epoch_{self.current_epoch}.png")


    def log_image_w_predictions(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:

        x       = x.cpu().detach().numpy()
        y_true  = y_true.cpu().detach().numpy()
        y_pred  = y_pred.cpu().detach().numpy()

        # Random image
        # Plot a 2x2 grid of images x
        fig, ax = plt.subplots(2, 2, figsize = (10, 10))
        for i in range(4):
            ax[i//2, i%2].imshow(x[i,0], cmap = "gray")
            ax[i//2, i%2].set_title(f"d={np.exp(y_true[i,0]):.1f} | y={y_true[i, 0]:.3f} | ŷ={y_pred[i, 0]:.3f}")
            ax[i//2, i%2].axis("off")

        if not os.path.exists(f"{self.logger.log_dir}/images"):
            os.mkdir(f"{self.logger.log_dir}/images")

        plt.savefig(f"{self.logger.log_dir}/images/epoch_{self.current_epoch}.png")