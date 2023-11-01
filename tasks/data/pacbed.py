import lightning as L
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
from pathlib import Path
from typing import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from .. import utils
from .augmentation import Augmenter

def process_pacbed_from_file(file: str, shape: Tuple[int, int, int]):
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
    x = np.reshape(x, (*shape, 1)) # Saved data format
    x = np.swapaxes(x, 1, 3) # Pytorch requires a (B, C, H, W) format
    x = np.swapaxes(x, 0, 2)

    return x

def process_multiple_pacbed_from_metadata(
        df: pd.DataFrame,
        ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
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

    metadata_keys = ["Phase index","Phase","Energy index","Energy","Convergence angle index","Convergence angle"]
    xs = []
    ys = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading simulated data..."):

        # Extract nz images from single file
        file = row["Filename"]
        nx, ny, nz = row["DimX"], row["DimY"], row["DimZ"]

        # Get PACBED image batch from single file
        image_batch = process_pacbed_from_file(file, (nx, ny, nz))

        # All images in a file have the same metadata
        metadata = [{key: row[key] for key in metadata_keys} for _ in range(nz)]

        # Thickness is the z-index of the sample x. Add to the metadata
        for m, d in zip(metadata, range(nz)):
            m["Thickness"] = d+1

        xs.append(image_batch)
        ys.extend(metadata)

    x = np.concatenate(xs)

    return x, ys

class ExperimentalDataset(Dataset):

    def __init__(self, metadata_path: Path, src_path: Path, target: str = "Phase index") -> None:
        super().__init__()

        self.metadata_path = metadata_path
        self.src_path = src_path

        # Read the metadata and add the root path to the filenames
        self.metadata = pd.read_csv(metadata_path)
        self.metadata["Filename"] = self.metadata["Filename"].apply(lambda x: src_path / x)

        # Filter out if target is not in metadata
        self.metadata = self.metadata[self.metadata[target].notna()]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> Tuple[Tensor, Dict[str, Tensor]]:

        x = torch.tensor(
            utils.bin_to_image(self.metadata["Filename"].iloc[idx])
        ).unsqueeze(0)

        y = self.metadata.iloc[idx].to_dict()

        # Remove string values from metadata
        y = {key: value for key, value in y.items() if not (isinstance(value, str) or isinstance(value, Path)) }

        return x, y

class PACBEDDataset(Dataset):

    def __init__(self, x: np.array, y: List[Dict[str, Any]], transforms = None, n_classes: int = 0) -> None:
        super().__init__()


        self.transforms = transforms
        self.n_classes  = n_classes

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = torch.tensor(self.x[idx, :, :, :])
        y = {key: torch.tensor(value) for key, value in self.y[idx].items() if not isinstance(value, str)}

        if self.transforms:
            return self.transforms(x), y
        else:
            return x, y
        
    @classmethod
    def from_files(cls, metadata_path: Path, src_path: Path, transforms = None, target: str = "Phase index") -> "PACBEDDataset":
        

        # Read the metadata and add the root path to the filenames
        metadata = pd.read_csv(metadata_path)
        metadata["Filename"] = metadata["Filename"].apply(lambda x: src_path / x)
        n_classes = len(metadata[target].unique())

        # Read the PACBED images and metadata
        x, y = process_multiple_pacbed_from_metadata(metadata)

        return cls(x, y, transforms, n_classes)


        
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
            
        return self.x[idx], self.y[idx]
    
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
        x, ys = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc="Creating in-memory dataset..."):
            # Batch has shape 
            # x: (batch_size, 1, n_pixels, n_pixels)
            # y: Dict[str, Tensor]

            x.append(batch[0])

            # Transform into a list of dicts
            y = [{key: value[i] for key, value in batch[1].items()} for i in range(len(batch[0]))]
            ys.extend(y)
            

        x = torch.cat(x, dim=0)
        return cls(x, ys)

class PACBEDDataModule(L.LightningDataModule):

    def __init__(self, 
                 simulated_metadata_path: Path, 
                 simulated_src_path: Path,
                 experimental_metadata_path: Path,
                 experimental_src_path: Path,
                 target: str = "Phase index",
                 train_transforms = None,
                 val_transforms = None,
                 batch_size: int = 32,
                 n_workers: int = 32,
                 n_train_samples: int = 1000,
                 n_val_samples: int = 1000,
                 n_test_samples: int = 1000,
                 ) -> None:
        super().__init__()

        self.simulated_metadata_path = simulated_metadata_path
        self.simulated_src_path = simulated_src_path
        self.experimental_metadata_path = experimental_metadata_path
        self.experimental_src_path = experimental_src_path

        self.target = target
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms


    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == "fit" or stage is None:
            # Create a dataset from the generated data, augmented with train transforms
            self.train_set              = PACBEDDataset.from_files(self.simulated_metadata_path, self.simulated_src_path, self.train_transforms, self.target)
            self.train_sampler          = RandomSampler(self.train_set, replacement=True, num_samples=self.n_train_samples)
            
            # Create a dataset from the generated data, augmented with val transforms
            # This dataset is continuously augmented and generated, thus not fixed
            self.realistic_set          = PACBEDDataset(self.train_set.x, self.train_set.y, self.val_transforms, self.target)
            self.realistic_sampler      = RandomSampler(self.realistic_set, replacement=True, num_samples=self.n_val_samples)
            self.realistic_loader       = DataLoader(self.realistic_set, batch_size=self.batch_size, num_workers=self.n_workers, sampler=self.realistic_sampler, pin_memory=True)

            # Create a fixed dataset to be reused between epochs
            self.val_set                = InMemoryPACBEDPhaseDataset.from_dataloader(self.realistic_loader)
            self.val_sampler            = RandomSampler(self.val_set, replacement=True)

        elif stage == "test":

            # Create a fixed test dataset to be used at the end of training
            self.test_initial_loader    = DataLoader(self.realistic_set, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True)
            self.test_sampler           = RandomSampler(self.test_initial_loader, replacement=True, num_samples=self.n_test_samples)
            self.test_set               = InMemoryPACBEDPhaseDataset.from_dataloader(self.test_initial_loader)
        
            # Create a test set from experimental data
            self.experimental_set       = ExperimentalDataset(self.experimental_metadata_path, self.experimental_src_path)

        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
            
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.n_workers, sampler=self.train_sampler, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.n_workers, sampler=self.val_sampler, pin_memory=True)

    def test_dataloader(self) -> DataLoader:

        return [
            DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.n_workers, sampler=self.test_sampler, pin_memory=True),
            DataLoader(self.experimental_set, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True)
            ]



def build_datamodule(args) -> PACBEDDataModule:

    train_transforms = Augmenter(
        n_pixels_original=args.original_size, 
        n_pixels_target=args.target_size, 
        crop=args.crop, 
        eta=args.eta,
        translate=(0.01, 0.01),
        p_occlusion=args.p_occlusion
        )
    
    val_transforms = Augmenter(
        n_pixels_original=args.original_size, 
        n_pixels_target=args.target_size, 
        crop=args.crop, 
        eta=args.eta,
        translate=(0.01, 0.01),
        p_occlusion=0
        )

    dm = PACBEDDataModule(
        simulated_metadata_path     = Path(args.simulated_metadata_file),
        simulated_src_path          = Path(args.simulated_src_path),
        experimental_metadata_path  = Path(args.experimental_metadata_file),
        experimental_src_path       = Path(args.experimental_src_path),
        target                      = args.target,
        batch_size                  = args.batch_size,
        n_workers                   = args.n_workers,
        n_train_samples             = args.n_train_samples,
        n_val_samples               = args.n_valid_samples,
        n_test_samples              = args.n_test_samples,
        train_transforms            = train_transforms,
        val_transforms              = val_transforms,
    )

    return dm
