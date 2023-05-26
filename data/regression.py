#%%
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
from typing import *
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm

def save_torch_pacbed_data(name: str, x: np.ndarray, y: np.ndarray, root: Path):
    """
    Saves a PACBED dataset to a binary file.

    Args:
        x (np.ndarray): A numpy array of shape (n_samples, n_pixels, n_pixels, 1)
        y (np.ndarray): A numpy array of shape (n_samples, 1)
        file (str): Path to the binary file
    """

    # Save data as (B, C, H, W) format
    x_path = root / Path(f"{name}_x")
    y_path = root / Path(f"{name}_y")
    np.save(x_path, x, allow_pickle=False)
    np.save(y_path, y, allow_pickle=False)

def read_torch_pacbed_data(name: str, root: Path):
    """
    Reads a binary file containing a PACBED dataset and returns a tuple (x, y) where
    x is a numpy array of shape (n_samples, n_pixels, n_pixels, 1) and y is a numpy array
    of shape (n_samples, 1) containing the thickness of each sample.

    Args:
        file (str): Path to the binary file

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (x, y) where x is a numpy array of shape (n_samples, n_pixels, n_pixels, 1)
    """

    x_path = root / Path(f"{name}_x.npy")
    y_path = root / Path(f"{name}_y.npy")

    x = np.load(x_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')

    return x, y

def generate_test_dataset(files: List[Path], n_samples: int, n_pixels: int, n_samples_per_file: int, augmenter: None, n_workers: int = 8):
    """
    Generates a test dataset. This function is a generator that yields a tuple (x, y) where
    x is a tensor of shape (1, n_pixels, n_pixels, 1) and y is a tensor of shape (1, 1) containing
    the thickness of the sample.

    Args:
        files (List[Path]): List of paths to the binary files containing the PACBED dataset
        n_samples (int): Number of samples in the dataset
        n_pixels (int): Number of pixels in the PACBED
        n_samples_per_file (int): Number of samples in each binary file
        augmenter (Augmenter): Augmenter object
        n_workers (int, optional): Number of workers to use. Defaults to 8.

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (x, y) where x is a tensor of shape (1, n_pixels, n_pixels, 1) and y is a tensor of shape (1, 1) containing
        the thickness of the sample.      
    """

    dataset   = PACBEDDataset(files = files, n_samples = n_samples_per_file, n_pixels=n_pixels, transforms=augmenter)
    sampler   = RandomSampler(dataset, replacement=True, num_samples=n_samples)
    loader    = DataLoader(dataset, batch_size=1, num_workers=n_workers, sampler=sampler)

    for x, y in tqdm(loader, total=n_samples):
        yield x, y

def generate_test_dataset_into_directory(files: List[Path], target_dir: Path, n_samples: int, n_pixels: int, n_samples_per_file: int, augmenter: None, n_workers: int = 8):
    """
    Generates a test dataset into a directory.
    
    Args:
        files (List[Path]): List of paths to the binary files containing the PACBED dataset
        target_dir (Path): Path to the directory where the dataset will be saved
        n_samples (int): Number of samples in the dataset
        n_pixels (int): Number of pixels in the PACBED images
        n_samples_per_file (int): Number of samples in each binary file
        n_workers (int, optional): Number of workers to use for the data loader. Defaults to 8.
    """

    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    else:
        for file in target_dir.glob("*"):
            file.unlink()

    for i, (x, y) in enumerate(generate_test_dataset(files, n_samples, n_pixels, n_samples_per_file, augmenter, n_workers)):
        save_torch_pacbed_data(f"{i}", x.numpy(), y.numpy(), target_dir)


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

    def __init__(self, files: Union[List[str],str], n_samples: int, n_pixels: int, device: str = "cpu", transforms = None) -> None:
        super().__init__()

        if isinstance(files, List) and len(files) > 1:
            x, y = process_multiple_pacbed_from_file(files, n_samples, n_pixels)
        elif isinstance(files, str):
            x, y = process_pacbed_from_file(files, n_samples, n_pixels)
        elif isinstance(files, List) and len(files) == 1:
            x, y = process_pacbed_from_file(files[0], n_samples, n_pixels) 
        else:
            raise ValueError()

        self.transforms = transforms
        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.y = torch.tensor(y, device=device, dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):

        if self.transforms:
            return self.transforms(self.x[idx, :, :, :]), torch.log(self.y[idx])
        else:

            return self.x[idx, :, :, :], self.y[idx]


class FixedPACBEDDataset(Dataset):
    """
    Dataset class for PACBED data written to files in a directory. The dataset is a list of tuples (x, y) where x is a tensor
    of shape (n_pixels, n_pixels, 1) and y is a tensor of shape (1,).

    Args:
        root (Path): Path to the root directory containing the data

    Returns:
        Dataset: A dataset object
    """
    
    def __init__(self, root: Path) -> None:
        super().__init__()

        self.root = root
        self.x_files = list(root.glob("*_x.npy"))
        self.y_files = list(root.glob("*_y.npy"))

    def __len__(self):
        return len(self.x_files)
    
    def __getitem__(self, idx):
            
        x, y = read_torch_pacbed_data(str(idx), self.root)
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        return x[0,:,:,:], y[0,:]
    

class InMemoryPACBEDDataset(Dataset):
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
    def from_dataloader(cls, dataloader: DataLoader) -> "InMemoryPACBEDDataset":
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
    