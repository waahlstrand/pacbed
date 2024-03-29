from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional
from skimage.draw import disk
from typing import *

class RadialNormalization(nn.Module):

    def __init__(self, n_pixels_original: int, n_pixels_crop: int, n_pixels_target: int) -> None:
        super().__init__()

        self.n_pixels_original = n_pixels_original
        self.n_pixels_target = n_pixels_target
        self.n_pixels_crop = n_pixels_crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        d = torch.linspace(-self.n_pixels_original // 2, self.n_pixels_original // 2, self.n_pixels_original, device=x.device)
        X, Y = torch.meshgrid(d, d)
        R2 = X**2 + Y**2
        R2 = R2.view_as(x)
        
        a = torch.pow(10, -7 + 2 * torch.rand(1, device=x.device))
        b = (5 + 95 * torch.rand(1, device=x.device)) * (self.n_pixels_crop / self.n_pixels_target)

        p = torch.rand(1)

        if p > 0.5:

            noise = a * torch.exp(-R2/b**2)
        else:
            noise = a * (1-torch.exp(-R2/b**2))

        return x + noise


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

    def __init__(self, eta: float | str):

        super().__init__()

        self.eta = eta

    def forward(self, x: torch.Tensor):

        if self.eta == "random":
            eta = torch.rand(1, device=x.device)
        else:
            eta = self.eta

        return self.add_noise(x, eta)

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


class RandomGaussianBlur(nn.Module):

    def __init__(self, min_var: float, max_var: float, kernel_size: int = 3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.min_var_log10 = torch.log10(torch.tensor(min_var))
        self.max_var_log10 = torch.log10(torch.tensor(max_var))
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        var = torch.rand(1).item() * (self.max_var_log10 - self.min_var_log10) + self.min_var_log10
        var = torch.pow(10, var)
        y = functional.gaussian_blur(x, self.kernel_size, var.item())

        return y


class Augmenter(nn.Module):

    def __init__(self, 
                 n_pixels_original: int, 
                 n_pixels_target: int,
                 crop: int, 
                 eta: float | str = 0.1, 
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
            RadialNormalization(n_pixels_original, self.crop, n_pixels_target),
            transforms.RandomApply([
                RandomGaussianBlur(0.1, 2.0),
            ], p = 0.5),
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