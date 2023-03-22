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

    def occlude(self, x: torch.Tensor, invert: bool, max_radius_ratio: float = 0.7, min_radius_ratio: float = 0.2):

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
                 p_occlusion: float = 0.5):

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

    def forward(self, x):

        x = self.conv1(x)
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
            nn.Linear(1024, 1),
        )

    def forward(self, x):
            
            return self.model(x)

class PACBED(pl.LightningModule):

    def __init__(self, backbone: str, n_pixels: int, lr: float = 1e-3):

        super().__init__()

        self.n_pixels   = n_pixels
        self.lr         = lr
        self.backbone   = backbone
        self.save_hyperparameters()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride = 1, padding = 0),
            nn.ReLU(),
        )

        if backbone == "custom":
            self.model = CustomPACBEDBackbone()
        elif backbone == "efficientnet_b5":
            self.model = models.efficientnet_b5(pretrained=False)
            self.model.classifier[1] = nn.Linear(1280, 1)
        elif backbone == "resnet34":
            self.model = models.resnet34(pretrained=False)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 1)
        else:
            raise ValueError("Backbone not implemented.")

    def forward(self, x):

        return self.model(self.pre_conv(x))
    
    def configure_optimizers(self):

        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log images
        if batch_idx % 50 == 0:
            self.log_image_w_predictions(x, y, y_hat, batch_idx)

        return loss

    def log_image_w_predictions(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, batch_idx: int) -> None:

        x       = x.cpu().detach().numpy()
        y_true  = y_true.cpu().detach().numpy()
        y_pred  = y_pred.cpu().detach().numpy()


        # Random image
        i = np.random.randint(0, x.shape[0])
        img = x[i, 0, :, :]

        # Plot
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))
        ax.imshow(img.squeeze(), cmap = "gray")
        ax.set_title(f"d={np.exp(y_true[i,0]):.1f} | y={y_true[i, 0]:.3f} | ŷ={y_pred[i, 0]:.3f}")
        ax.axis("off")

        if not os.path.exists(f"{self.logger.log_dir}/images"):
            os.mkdir(f"{self.logger.log_dir}/images")

        plt.savefig(f"{self.logger.log_dir}/images/epoch_{self.current_epoch}__batch__{batch_idx}__{int(np.exp(y_true[i,0]))}.png")