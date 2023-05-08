import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from typing import *

import os
from pathlib import Path
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import json
from models import Augmenter, PACBED
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def scatter_hist(x: np.ndarray, y: np.ndarray, left: float = 0.1, width: float = 0.65, bottom: float = 0.1, height: float = 0.65, spacing: float = 0.005) -> Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes, plt.Axes]:

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=0.8)

    # Compute domain
    xmin = np.min(x)
    xmax = np.max(x)

    # Perfect line
    ax.plot([xmin, xmin], [xmax,xmax], 'k')

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    return fig, ax, ax_histx, ax_histy


def preprocess(x: np.ndarray, scaling=0.25, size=256):

    # Reshape 
    x = x.reshape((size, size))

    x = np.abs(scaling * x / np.mean(np.abs(x)))

    return x


def bin_to_image(file: Path):

    # Read binary file
    with open(file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)

    # Preprocess
    data = preprocess(data)

    return data


def experimental_dataloader(data_dir: Path, results_file: Path, **kwargs):

    with open(results_file, "r") as f:
        results = json.load(f)

    x  = torch.stack([torch.tensor(bin_to_image(data_dir / result['file'])) for result in results['data']]).unsqueeze(1)
    y  = torch.tensor([result['thickness'] for result in results['data']]).log()

    dataset = TensorDataset(x, y)

    return DataLoader(dataset, **kwargs)
