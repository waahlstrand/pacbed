import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
import pytorch_lightning.loggers as pl_loggers
import torchvision.models as models
from skimage.draw import disk

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


class AnnulusOcclusion(nn.Module):

    def __init__(self, n_pixels: int, invert: bool = False):

        super().__init__()

        self.invert = invert
        self.n_pixels = n_pixels
        self.mask   = torch.zeros((n_pixels, n_pixels), dtype=torch.uint8)

    def occlude(self, input: torch.Tensor, invert: bool, max_radius_ratio: float = 0.7, min_radius_ratio: float = 0.2):

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

        if self.invert:
            y = (1-mask)*input[0,:,:]
        else:
            y = mask*input[0,:,:]

        return y.view(input.shape)

    def forward(self, x):

        return self.occlude(x, self.invert)


class Augmenter(nn.Module):

    def __init__(self, n_pixels: int, crop: int, eta: float, scaling="linear", p_occlusion: float = 0.5):

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
            Normalize(scaling),
            transforms.RandomApply([
            transforms.RandomChoice([
                AnnulusOcclusion(n_pixels, invert = False),
                AnnulusOcclusion(n_pixels, invert = True)
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

        self.model = nn.Sequential(
            ConvBlock(3, 16, 3),
            *[ConvBlock(16*(2**i), 16*(2**(i+1)), 3) for i in range(7)],
            nn.Flatten(),
            nn.Linear(8192, 2048),
            nn.ELU(),
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
    
    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % 100 and isinstance(self.trainer.logger, pl_loggers.TensorBoardLogger): # Log every 100 batches
            self.log_tb_images(x, y, y_hat, batch_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
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