#%%
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
from typing import *
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def process_pacbed_from_file(file: str, n_pixels: int, class_idx: int):
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
    x = np.reshape(x, (n_pixels, n_pixels, -1, 1)) # Saved data format
    x = np.swapaxes(x, 1, 3) # Pytorch requires a (B, C, H, W) format
    x = np.swapaxes(x, 0, 2)

    # Target is simply the thickness, equivalent to index + 1
    y = np.ones((x.shape[0], 1), dtype=np.int64) * class_idx

    return x, y

def process_multiple_pacbed_from_file(files: List[Path], n_pixels: List[int], class_idxs: List[int]):
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

    data = [process_pacbed_from_file(file, n, class_idx) for file, n, class_idx in zip(files, n_pixels, class_idxs)]
    xs, ys = list(zip(*data))

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    return x, y

class PACBEDPhaseDataset(Dataset):
    """
    Dataset class for PACBED data. The dataset is a list of tuples (x, y) where x is a tensor
    of shape (n_pixels, n_pixels, 1) and y is a tensor of shape (1,).

    Args:
        files (List[str]): List of paths to the binary files
        source (pd.DataFrame): A dataframe containing source data and metadata for the dataset
        device (str, optional): Device to store the data on. Defaults to "cpu".
        transforms (Callable, optional): A callable that transforms the data. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (x, y) where x is a tensor of shape (n_pixels, n_pixels, 1)
            
    """

    def __init__(self, source: pd.DataFrame, device: str = "cpu", transforms = None) -> None:
        super().__init__()

        files       = source["Filename"].to_list()
        class_idxs  = source["Class index"].to_list()
        n_pixels_x  = source["DimX"].to_list()

        x, y = process_multiple_pacbed_from_file(files, n_pixels_x, class_idxs)

        self.transforms = transforms
        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.y = torch.tensor(y, device=device, dtype=torch.float32)


    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):

        if self.transforms:
            return self.transforms(self.x[idx, :, :, :]), self.y[idx]
        else:

            return self.x[idx, :, :, :], self.y[idx]


class InMemoryPACBEDPhaseDataset(Dataset):
    """
    Dataset class for PACBED data to be stored completely in memory. The dataset is a list of tuples (x, y) where x is a tensor
    of shape (n_pixels, n_pixels, 1) and y is a tensor of shape (1,).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
            
        return self.x[idx,:,:,:], self.y[idx,:]
    
    @classmethod
    def from_dataloader(cls, dataloader: DataLoader) -> "InMemoryPACBEDPhaseDataset":
        """
        Creates an InMemoryPACBEDDataset from a dataloader. This is useful for creating a dataset from a dataloader
        that has been created from a PACBEDDataset. 
        
        Args:
            dataloader (DataLoader): A dataloader created from a PACBEDDataset

        Returns:
            InMemoryPACBEDDataset: A dataset object
        """
        x, y = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            x.append(batch[0])
            y.append(batch[1])
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return cls(x, y)

def conv_output_size(n_pixels, kernel_size, dilation, stride, padding):
    return np.floor((n_pixels + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    