import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
import pytorch_lightning.loggers as pl_loggers
from torchvision.models import resnet18

class Normalize(nn.Module):

    def __init__(self, scaling="linear"):

        super().__init__()

        self.scaling = scaling

    def forward(self, x):

        x = 0.25 * x / torch.mean(torch.abs(x))

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

    def forward(self, x):

        return self.add_noise(x, self.eta)

    def add_noise(self, input: torch.Tensor, eta):

        device = input.device

        # Noise model parameters.
        c1_log_intensity    = torch.tensor(-7.60540384562294, device = device)
        c0_log_intensity    = torch.tensor(28.0621318839493, device = device)
        c2_log_s_signal     = torch.tensor(0.0349329915028267, device = device)
        c1_log_s_signal     = torch.tensor(-0.304984702105353, device = device)
        c0_log_s_signal     = torch.tensor(6.86126419242947, device = device)
        c1_log_s_background = torch.tensor(-6.99617784594964, device = device)
        c0_log_s_background = torch.tensor(12.448421647627, device = device)

        # Rescale intensity. It is assumed that before this operation, sum(x) = 1.
        input = input + 1e-6
        x = torch.exp(c1_log_intensity * eta + c0_log_intensity) * input

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
        

class Augmenter(nn.Module):

    def __init__(self, n_pixels: int, crop: int, eta: float, scaling="linear"):

        super().__init__()

        self.n_pixels = n_pixels
        self.crop = crop
        self.eta = eta
        self.scaling = scaling

        self.augment = nn.Sequential(
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(crop),
            transforms.Resize(n_pixels),
            AddNoise(eta),
            Normalize(scaling)
   
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


class PACBED(pl.LightningModule):

    def __init__(self, n_pixels: int, lr: float = 1e-3):

        super().__init__()

        self.n_pixels   = n_pixels
        self.lr         = lr

        self.model = nn.Sequential(
            # augmenter,
            ConvBlock(1, 16, 3),
            *[ConvBlock(16*(2**i), 16*(2**(i+1)), 3) for i in range(2)],
            nn.Flatten(),
            nn.Linear(1048576, 256),
            nn.ELU(),
            nn.Linear(256, 192),
            nn.ELU(),
            nn.Linear(192, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        # self.model = resnet18(pretrained=False)

    def forward(self, x):

        return self.model(x)
    
    def configure_optimizers(self):

        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)
        if batch_idx % 10 and isinstance(self.trainer.logger, pl_loggers.TensorBoardLogger): # Log every 10 batches
            self.log_tb_images(x, y, y_hat, batch_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("val_loss", loss)
        return loss
    
    def log_tb_images(self, x, y_trues, y_preds, batch_idx) -> None:

        x = x.cpu().detach().numpy()
        y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()
         
         # Get tensorboard logger
        tb_logger: pl_loggers.TensorBoardLogger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
       
        # Log the images (Give them different names)
        for img_idx, (image, y_true, y_pred) in enumerate(zip(x, y_trues, y_preds)):
            tb_logger.add_image(f"Image/{batch_idx}_{img_idx}/{y_true[0]}_{y_pred[0]}", image, 0)
            # tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
            # tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0)