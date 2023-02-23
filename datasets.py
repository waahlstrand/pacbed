#%%
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import *
import numpy as np
import pytorch_lightning as pl
from models import Augmenter

def process_pacbed_from_file(file: str, n_samples: int, n_pixels: int):
    """
    Reads a binary file containing a PACBED dataset and returns a tuple (x, y) where
    x is a numpy array of shape (n_samples, n_pixels, n_pixels, 1) and y is a numpy array
    of shape (n_samples, 1) containing the thickness of each sample.

    Args:
        file (str): Path to the binary file
        n_samples (int): Number of samples in the file
        n_pixels (int): Number of pixels per sample

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (x, y) where x is a numpy array of shape (n_samples, n_pixels, n_pixels, 1)
    """

    x = np.fromfile(file, dtype = np.float32)
    x = np.reshape(x, (n_pixels, n_pixels, n_samples, 1)) # Saved data format
    x = np.swapaxes(x, 1, 3) # Pytorch requires a (B, C, H, W) format
    x = np.swapaxes(x, 0, 2)

    # Target is simply the thickness, equivalent to index + 1
    y = np.arange(1, n_samples+1).reshape(n_samples, 1)

    return x, y

def process_multiple_pacbed_from_file(files: List[str], n_samples: int, n_pixels: int):
    """
    Reads a list of binary files containing PACBED datasets and returns a tuple (x, y) where
    x is a numpy array of shape (n_samples, n_pixels, n_pixels, 1) and y is a numpy array
    of shape (n_samples, 1) containing the thickness of each sample.

    Args:
        files (List[str]): List of paths to the binary files
        n_samples (int): Number of samples in each file
        n_pixels (int): Number of pixels per sample dimensions

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (x, y) where x is a numpy array of shape (n_samples, n_pixels, n_pixels, 1)
    """

    data = [process_pacbed_from_file(file, n_samples, n_pixels) for file in files]
    xs, ys = list(zip(*data))

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    return x, y

class PACBEDDataset(Dataset):
    """
    Dataset class for PACBED data. The dataset is a list of tuples (x, y) where x is a tensor
    of shape (n_pixels, n_pixels, 1) and y is a tensor of shape (1,).
    
    Args:
        files (List[str] | str): List of paths to the binary files or a single path to a binary file
        n_samples (int): Number of samples in each file
        n_pixels (int): Number of pixels per sample dimensions
        device (str): Device to load the data to
        
    Returns:
            Dataset: A dataset object
            
    """

    def __init__(self, files: List[str] | str, n_samples: int, n_pixels: int, device: str = "cpu", transforms = Augmenter) -> None:
        super().__init__()

        if isinstance(files, List) and len(files) > 1:
            x, y = process_multiple_pacbed_from_file(files, n_samples, n_pixels)
        elif isinstance(files, str):
            x, y = process_pacbed_from_file(files, n_samples, n_pixels)
        else:
            raise ValueError()

        self.transforms = transforms
        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.y = torch.tensor(y, device=device, dtype=torch.float32)
        # self.y = torch.log(torch.tensor(y, device=device, dtype=torch.float32))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):

        if self.transforms:
            return self.transforms(self.x[idx, :, :, :]), torch.log(self.y[idx])
        else:

            return self.x[idx, :, :, :], self.y[idx]


def conv_output_size(n_pixels, kernel_size, dilation, stride, padding):
    return np.floor((n_pixels + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    